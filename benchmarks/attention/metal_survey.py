#!/usr/bin/env python3
"""Apple Metal attention baseline survey (MLX fast SDPA vs MPSGraph SDPA).

Benchmarks the causal prefill attention operator on the local Metal device
across a grid of sequence lengths, head dims and dtypes, using these Apple
baselines:

  * MLX mx.fast.scaled_dot_product_attention(mask="causal")          (default dispatch)
  * MLX same op with force_fused=True                                 (forced fused)
  * MLX unfused expression baseline: QK^T -> float32 softmax -> PV    (labeled expr)
  * MPSGraph scaledDotProductAttention via a compiled Swift helper

Every output is validated against an independent numpy float64 stable causal
reference computed once per case from the dtype-rounded inputs. Frozen
numerical tolerances: fp32 atol=1e-4 rtol=1e-3; fp16 atol=2e-3 rtol=2e-2.
Raw per-invocation timings are saved.

The Swift helper is compiled once (xcrun swiftc -O) into a build directory
outside the tracked tree. Requires mlx and numpy.

Run from anywhere:
  python3 benchmarks/attention/metal_survey.py \
      --output metal_survey.json --build-dir /tmp/metal-survey-build \
      --backends mlx_fast,mlx_fused,mlx_expr,mpsgraph
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    import mlx.core as mx
    MLX_OK = True
except Exception as exc:  # noqa: BLE001  (mlx may not be installed yet)
    mx = None
    MLX_OK = False
    MLX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPES = ["float32", "float16"]

NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
}

MX_DTYPE = {
    "float32": None,
    "float16": None,
}

if MLX_OK:
    MX_DTYPE["float32"] = mx.float32
    MX_DTYPE["float16"] = mx.float16

TOLERANCES = {
    "float32": {"atol": 1.0e-4, "rtol": 1.0e-3},
    "float16": {"atol": 2.0e-3, "rtol": 2.0e-2},
}

BACKENDS = ["mlx_fast", "mlx_fused", "mlx_expr", "mpsgraph"]
MLX_BACKENDS = {"mlx_fast", "mlx_fused", "mlx_expr"}

DEFAULT_SEQS = "256,1024"
DEFAULT_DIMS = "64,128,192,256"
DEFAULT_DTYPES = "float32,float16"

# Rough cap on the reference peak memory (bytes): the float64 causal softmax
# materializes several (B,H,N,N) intermediates (~6*8*B*H*N*N) plus the linear
# Q/K/V/reference arrays (~4*8*B*H*N*D).
MEMORY_CAP_BYTES = 20 * (2**30)  # 20 GiB


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_int_list(text, what, minimum=1):
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError(f"--{what} must be a non-empty comma list")
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--{what} value {p!r} is not an integer")
        if v < minimum:
            raise argparse.ArgumentTypeError(f"--{what} value {v} must be >= {minimum}")
        out.append(v)
    return out


def _parse_dtypes(text):
    parts = [p.strip().lower() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError("--dtypes must be a non-empty comma list")
    for p in parts:
        if p not in DTYPES:
            raise argparse.ArgumentTypeError(f"--dtypes value {p!r} not in {DTYPES}")
    return parts


def _parse_backends(text):
    parts = [p.strip().lower() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError("--backends must be a non-empty comma list")
    for p in parts:
        if p not in BACKENDS:
            raise argparse.ArgumentTypeError(f"--backends value {p!r} not in {BACKENDS}")
    return parts


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Apple Metal attention baseline survey (MLX fast SDPA and "
                    "MPSGraph SDPA vs numpy float64 reference).")
    p.add_argument("--output", default="metal_survey.json",
                   help="output JSON path (incremental writes after each case)")
    p.add_argument("--seqs", default=DEFAULT_SEQS,
                   help="comma-separated sequence lengths")
    p.add_argument("--dims", default=DEFAULT_DIMS,
                   help="comma-separated head dims")
    p.add_argument("--dtypes", default=DEFAULT_DTYPES,
                   help="comma-separated dtypes")
    p.add_argument("--backends", default=",".join(BACKENDS),
                   help="comma-separated backends to run (default all)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mem-cap-bytes", type=int, default=MEMORY_CAP_BYTES,
                   help="refuse cases whose float64 reference peak exceeds this")
    p.add_argument("--swift-src", default=os.path.join(here, "mpsgraph_survey.swift"),
                   help="path to the MPSGraph Swift helper source")
    p.add_argument("--build-dir", default=None,
                   help="directory (outside tracked files) for the compiled "
                        "Swift helper binary; created if needed")
    p.add_argument("--temp-dir", default=None,
                   help="directory for per-case input/output binaries; a "
                        "temporary dir is used if not given")
    p.add_argument("--keep-temp", action="store_true",
                   help="do not delete the temporary binary directory")
    p.add_argument("--no-mlx", action="store_true",
                   help="drop all MLX backends")
    p.add_argument("--no-mps", action="store_true",
                   help="drop the MPSGraph backend")
    args = p.parse_args(argv)

    try:
        args.seqs = _parse_int_list(args.seqs, "seqs")
        args.dims = _parse_int_list(args.dims, "dims")
        args.dtypes = _parse_dtypes(args.dtypes)
        args.backends = _parse_backends(args.backends)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    for name in ("heads", "batch", "warmup", "samples", "inner"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be >= 1 (got {getattr(args, name)})")
    if args.mem_cap_bytes < 1:
        p.error("--mem-cap-bytes must be >= 1")
    if args.build_dir is None:
        p.error("--build-dir is required (compiled Swift binary output dir, "
                "outside tracked files)")

    # Apply --no-mlx / --no-mps to the selected backend list.
    if args.no_mlx:
        args.backends = [b for b in args.backends if b not in MLX_BACKENDS]
    if args.no_mps:
        args.backends = [b for b in args.backends if b != "mpsgraph"]
    if not args.backends:
        p.error("no backends selected after --no-mlx/--no-mps filtering")
    return args


# ---------------------------------------------------------------------------
# Swift helper compilation
# ---------------------------------------------------------------------------

def compile_swift(swift_src, build_dir):
    """Compile the Swift helper once into build_dir; returns the binary path."""
    os.makedirs(build_dir, exist_ok=True)
    binary = os.path.join(build_dir, "mpsgraph_survey")
    src = os.path.abspath(swift_src)
    if not os.path.isfile(src):
        sys.exit(f"Swift source not found: {src}")
    # Rebuild only when the source is newer than the binary.
    if os.path.isfile(binary) and os.path.getmtime(binary) >= os.path.getmtime(src):
        return binary

    cmd = [
        "xcrun", "swiftc", "-O",
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
        src,
        "-o", binary,
    ]
    print(f"[compile] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("Swift compilation failed:\n" + proc.stdout + proc.stderr)
    if not os.path.isfile(binary):
        sys.exit("Swift compilation reported success but produced no binary")
    return binary


def swift_toolchain_version():
    try:
        proc = subprocess.run(["xcrun", "swiftc", "--version"],
                              capture_output=True, text=True, timeout=30)
        return (proc.stdout or proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Reference computation (numpy float64, stable causal softmax)
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, scale):
    """float64 causal prefill attention from (B,H,N,D) float64 arrays."""
    b, h, n, d = q.shape
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale  # (B,H,N,N)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # True where j > i
    scores = np.where(upper[None, None], -np.inf, scores)

    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)  # -inf -> 0
    e = np.where(np.isneginf(scores), 0.0, e)
    s = e.sum(axis=-1, keepdims=True)
    attn = e / s
    return np.matmul(attn, v)


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------

def compare_outputs(out_np, ref, tol):
    """Compare every output value against the float64 reference.

    Always checks shapes before subtracting. Returns a dict including
    shape_ok and within_tolerance; callers map the latter to numeric_failed.
    """
    out = np.asarray(out_np, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    shape_ok = bool(out.shape == ref.shape)
    if not shape_ok:
        return {
            "shape_ok": False,
            "output_shape": list(out.shape),
            "reference_shape": list(ref.shape),
            "error": "shape mismatch between output and reference",
            "within_tolerance": False,
        }

    finite_out = bool(np.isfinite(out).all())
    finite_ref = bool(np.isfinite(ref).all())

    err = np.abs(out - ref)
    denom = tol["atol"] + tol["rtol"] * np.abs(ref)
    normalized = err / denom

    if finite_out and finite_ref:
        max_abs_error = float(err.max())
        normalized_max = float(normalized.max())
        rms = float(np.sqrt(np.mean(np.square(err))))
        within_tolerance = bool((err <= denom).all())
    else:
        max_abs_error = float("nan")
        normalized_max = float("nan")
        rms = float("nan")
        within_tolerance = False

    return {
        "shape_ok": True,
        "output_shape": list(out.shape),
        "reference_shape": list(ref.shape),
        "finite_out": finite_out,
        "finite_ref": finite_ref,
        "max_abs_error": max_abs_error,
        "max_abs_error_normalized": normalized_max,
        "rms_error": rms,
        "within_tolerance": within_tolerance,
    }


# ---------------------------------------------------------------------------
# MLX helpers
# ---------------------------------------------------------------------------

def _causal_mask_mx(n, mx_dtype):
    """Additive causal mask as an mx array of dtype mx_dtype, pre-materialized.

    0 lower triangle (j <= i), -infinity upper (j > i). Building it involves
    numpy but it is created once OUTSIDE any timed region.
    """
    upper = np.triu(np.ones((n, n), dtype=np.float32), k=1)
    m = np.where(upper, -np.inf, 0.0).astype(np.float32)
    mask = mx.array(m).astype(mx_dtype)
    mx.eval(mask)
    return mask


def run_mlx_fast(q, k, v, scale, force_fused):
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=float(scale), mask="causal", force_fused=force_fused)
    mx.eval(out)
    return out


def make_mlx_expr_step(q, k, v, scale, mask, mx_dtype):
    """Build a timed callable for the unfused expression baseline.

    Arithmetic policy: QK^T and PV are computed in the input dtype; the
    softmax runs in float32 (max-subtraction, numerically stable); the
    probabilities are then cast back to the input dtype before the PV matmul.
    The causal mask is pre-built and materialized; the callable performs no
    numpy work, no input casts and no input evaluation inside the timed region.
    """
    def step():
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * float(scale)
        scores = scores + mask
        probs_f32 = mx.softmax(scores.astype(mx.float32), axis=-1)
        probs = probs_f32.astype(mx_dtype)
        out = mx.matmul(probs, v)
        mx.eval(out)
        return out
    return step


def time_mlx(step_fn, warmup, samples, inner):
    """Synchronized wall-latency timing of a step_fn returning an mx output.

    Each invocation materializes and evaluates a fresh output (mx.eval +
    mx.synchronize). Returns a flat list of per-invocation latencies (ms) of
    length samples*inner. MLX exposes no verified GPU-timestamp API here, so
    gpu_time is reported unavailable.
    """
    for _ in range(warmup):
        step_fn()
        mx.synchronize()
    latencies = []
    for _ in range(samples):
        for _ in range(inner):
            t0 = time.perf_counter()
            step_fn()
            mx.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1.0e3)
    return latencies


def run_mlx_backend(args, step_fn, ref, dtype, extra):
    """Run and time one MLX backend; returns the result record dict."""
    rec = dict(extra)
    try:
        for _ in range(args.warmup):
            step_fn()
        mx.synchronize()
        out = step_fn()
        out_np = np.array(out)
        rec["output_shape"] = list(out_np.shape)
        rec["comparison"] = compare_outputs(out_np, ref, TOLERANCES[dtype])
        rec["wall_timing"] = summarize_ms(
            time_mlx(step_fn, args.warmup, args.samples, args.inner))
        rec["gpu_time_available"] = False
        rec["gpu_time_note"] = (
            "No verified Metal GPU-timestamp API is exposed for MLX here; "
            "reported timing is warm synchronized API latency, not GPU time.")

        if not rec["comparison"]["shape_ok"]:
            rec["status"] = "numeric_failed"
        elif not rec["comparison"]["within_tolerance"]:
            rec["status"] = "numeric_failed"
        else:
            rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return rec


# ---------------------------------------------------------------------------
# MPSGraph helper invocation
# ---------------------------------------------------------------------------

def run_mpsgraph(binary, case_dir, shape, dtype, warmup, samples, inner,
                 q_bin, k_bin, v_bin):
    """Run the compiled Swift helper once and return (resp, out_array)."""
    out_bin = os.path.join(case_dir, "out.bin")
    resp_path = os.path.join(case_dir, "response.json")
    req = {
        "shape": list(shape),
        "dtype": dtype,
        "q_path": q_bin,
        "k_path": k_bin,
        "v_path": v_bin,
        "output_path": out_bin,
        "response_path": resp_path,
        "warmup": warmup,
        "samples": samples,
        "inner": inner,
    }
    req_path = os.path.join(case_dir, "request.json")
    with open(req_path, "w") as fh:
        json.dump(req, fh)

    proc = subprocess.run([binary, req_path], capture_output=True, text=True,
                          timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"swift helper exited {proc.returncode}: "
                           f"{proc.stdout}\n{proc.stderr}")

    with open(resp_path) as fh:
        resp = json.load(fh)
    if resp.get("status") != "ok":
        raise RuntimeError(f"swift helper reported status={resp.get('status')}: "
                           f"{resp.get('error')}")

    if not os.path.isfile(out_bin):
        raise RuntimeError("swift helper reported ok but wrote no output file")
    np_dtype = NP_DTYPE[dtype]
    out = np.frombuffer(open(out_bin, "rb").read(),
                        dtype=np_dtype).reshape(shape)
    return resp, out


def run_mpsgraph_backend(args, binary, case_dir, shape, dtype, ref, q_bin,
                         k_bin, v_bin):
    """Run the MPSGraph backend for one case; returns the result record."""
    rec = {}
    try:
        resp, out = run_mpsgraph(binary, case_dir, shape, dtype,
                                 args.warmup, args.samples, args.inner,
                                 q_bin, k_bin, v_bin)
        rec["output_shape"] = list(out.shape)
        rec["comparison"] = compare_outputs(out, ref, TOLERANCES[dtype])
        rec["device"] = {k: resp[k] for k in (
            "mtl_device_name", "mtl_device_registry_id", "mtl_is_low_power",
            "mtl_has_unified_memory", "mtl_max_threadgroup_memory_length",
            "mtl_max_threads_per_threadgroup") if k in resp}
        rec["compile_wall_ms"] = resp.get("compile_wall_ms")
        rec["warmup_count"] = resp.get("warmup_count")
        rec["mask"] = resp.get("mask")
        rec["scale"] = resp.get("scale")
        rec["api"] = resp.get("mpsgraph_api")
        rec["time_unit"] = resp.get("time_unit")
        rec["measurement_notes"] = resp.get("measurement_notes", [])
        rec["wall_timing"] = summarize_ms(resp.get("wall_times", []))

        gpu_avail = bool(resp.get("gpu_time_available"))
        rec["gpu_time_available"] = gpu_avail
        if gpu_avail:
            rec["gpu_timing"] = summarize_ms(resp.get("gpu_times", []))
        else:
            # Invalid/incomplete GPU timestamps: do not fabricate a summary.
            rec["gpu_timing"] = None

        if not rec["comparison"]["shape_ok"]:
            rec["status"] = "numeric_failed"
        elif not rec["comparison"]["within_tolerance"]:
            rec["status"] = "numeric_failed"
        else:
            rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return rec


def summarize_ms(samples):
    a = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
        "samples_ms": [float(x) for x in a],
    }


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def _mlx_version():
    try:
        return importlib.metadata.version("mlx")
    except Exception:  # noqa: BLE001
        return getattr(mx, "__version__", "unknown") if MLX_OK else None


def collect_meta(args):
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }
    if MLX_OK:
        meta["mlx"] = {
            "version": _mlx_version(),
            "import_ok": True,
        }
        try:
            meta["mlx"]["metal_available"] = bool(mx.metal.is_available())
        except Exception as exc:  # noqa: BLE001
            meta["mlx"]["metal_available"] = f"error: {exc}"
        # Prefer mx.device_info() (includes name/memory); fall back to the
        # metal-only variant.
        try:
            meta["mlx"]["device_info"] = mx.device_info()
        except Exception:  # noqa: BLE001
            try:
                meta["mlx"]["device_info"] = mx.metal.device_info()
            except Exception as exc:  # noqa: BLE001
                meta["mlx"]["device_info"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            meta["mlx"]["default_device"] = str(mx.default_device())
        except Exception as exc:  # noqa: BLE001
            meta["mlx"]["default_device"] = f"error: {exc}"
    else:
        meta["mlx"] = {
            "version": None,
            "import_ok": False,
            "import_error": MLX_IMPORT_ERROR,
        }
    meta["swift_toolchain"] = swift_toolchain_version()
    meta["swift_src"] = os.path.abspath(args.swift_src)
    meta["build_dir"] = os.path.abspath(args.build_dir)
    return meta


# ---------------------------------------------------------------------------
# Case handling
# ---------------------------------------------------------------------------

def sha256_hex(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_case(args, binary, case_dir, seq, dim, dtype, rng, case_seed, order):
    """Run one (seq, dim, dtype) case for the given backend order."""
    b, h, n, d = args.batch, args.heads, seq, dim
    shape = (b, h, n, d)
    scale = math.pow(float(dim), -0.5)
    np_dtype = NP_DTYPE[dtype]

    # Distinct Q, K, V rounded directly to the target dtype.
    q_d = rng.normal(0.0, 0.5, size=shape).astype(np_dtype)
    k_d = rng.normal(0.0, 0.5, size=shape).astype(np_dtype)
    v_d = rng.normal(0.0, 0.5, size=shape).astype(np_dtype)

    # Reference once per case, from the dtype-rounded inputs.
    ref = reference_attention(q_d.astype(np.float64),
                              k_d.astype(np.float64),
                              v_d.astype(np.float64), scale)

    case = {
        "seq": seq,
        "dim": dim,
        "dtype": dtype,
        "heads": h,
        "batch": b,
        "scale": scale,
        "seed": args.seed,
        "case_seed": case_seed,
        "backend_order": order,
        "inputs_sha256": {
            "q": sha256_hex(q_d),
            "k": sha256_hex(k_d),
            "v": sha256_hex(v_d),
        },
        "results": {},
    }

    # Write contiguous input binaries for the Swift helper.
    q_bin = os.path.join(case_dir, "q.bin")
    k_bin = os.path.join(case_dir, "k.bin")
    v_bin = os.path.join(case_dir, "v.bin")
    q_d.tofile(q_bin)
    k_d.tofile(k_bin)
    v_d.tofile(v_bin)

    # MLX inputs (shared across MLX backends), materialized once.
    if not args.no_mlx and MLX_OK:
        q = mx.array(q_d)
        k = mx.array(k_d)
        v = mx.array(v_d)
        mx.eval(q)
        mx.eval(k)
        mx.eval(v)
    else:
        q = k = v = None

    for backend in order:
        if backend == "mpsgraph":
            # Drain any pending MLX work so the Swift timing is not polluted by
            # overlapping GPU work from a previous backend.
            if MLX_OK:
                mx.synchronize()
            rec = run_mpsgraph_backend(args, binary, case_dir, shape, dtype,
                                       ref, q_bin, k_bin, v_bin)
        elif backend in MLX_BACKENDS:
            if not MLX_OK:
                rec = {
                    "status": "failed",
                    "error": {"type": "ImportError",
                              "message": "mlx not available: " + MLX_IMPORT_ERROR},
                }
            elif backend == "mlx_fast":
                extra = {
                    "force_fused": False,
                    "kernel": "unknown",
                    "kernel_note": (
                        "Default MLX dispatch; the actual fused/unfused kernel "
                        "is inferred from the pinned source, not guessed here. "
                        "Source dispatch inspection required."),
                }
                rec = run_mlx_backend(
                    args,
                    lambda: run_mlx_fast(q, k, v, scale, False),
                    ref, dtype, extra)
            elif backend == "mlx_fused":
                extra = {
                    "force_fused": True,
                    "kernel": "unknown",
                    "kernel_note": (
                        "force_fused=True requested; the actual selected "
                        "kernel is not queried without guessing."),
                }
                rec = run_mlx_backend(
                    args,
                    lambda: run_mlx_fast(q, k, v, scale, True),
                    ref, dtype, extra)
            else:  # mlx_expr
                mask = _causal_mask_mx(n, MX_DTYPE[dtype])
                extra = {
                    "force_fused": None,
                    "expression_baseline": True,
                    "arithmetic_policy": {
                        "qk_dtype": dtype,
                        "pv_dtype": dtype,
                        "softmax_dtype": "float32",
                        "mask": "additive_causal_prebuilt_outside_timing",
                    },
                }
                rec = run_mlx_backend(
                    args,
                    make_mlx_expr_step(q, k, v, scale, mask, MX_DTYPE[dtype]),
                    ref, dtype, extra)
        else:  # pragma: no cover - guarded by parse_args
            rec = {"status": "failed",
                   "error": {"type": "ValueError", "message": f"unknown backend {backend}"}}
        case["results"][backend] = rec

    return case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """Replace non-finite floats with None so allow_nan=False never trips."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _save_document(path, document):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_sanitize(document), fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)

    needs_swift = "mpsgraph" in args.backends
    binary = compile_swift(args.swift_src, args.build_dir) if needs_swift else None

    meta = collect_meta(args)
    meta["grid"] = {
        "seqs": args.seqs,
        "dims": args.dims,
        "dtypes": args.dtypes,
        "backends": args.backends,
        "heads": args.heads,
        "batch": args.batch,
        "warmup": args.warmup,
        "samples": args.samples,
        "inner": args.inner,
        "seed": args.seed,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    document = {
        "meta": meta,
        "timing_caveats": [
            "All timings are in milliseconds.",
            "MPSGraph GPU times are MPSCommandBuffer GPUStartTime/GPUEndTime "
            "deltas per invocation (exclude input/output CPU copies and "
            "compile/warmup). If GPU timestamps were invalid they are reported "
            "unavailable (gpu_timing=null), never mislabeled.",
            "MPSGraph wall times are synchronized API latency, timed from "
            "before command-buffer allocation through waitUntilCompleted.",
            "MLX timings are warm synchronized API latency (mx.eval + "
            "mx.synchronize) per invocation; no verified GPU-timestamp API is "
            "exposed for MLX here, so gpu_time is unavailable for MLX.",
            "Each timed invocation computes a fresh output; nothing is cached "
            "across samples. samples x inner per-invocation timings are kept "
            "in each result's wall_timing.samples_ms (and gpu_timing.samples_ms).",
            "Backends run serially per case in a randomized order; an "
            "mx.synchronize() drains MLX work before the MPSGraph helper runs.",
        ],
        "cases": [],
    }

    # Temporary binary directory for per-case inputs/outputs.
    temp_root = args.temp_dir
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix="metal_survey_")
        temp_owned = True
    else:
        os.makedirs(temp_root, exist_ok=True)
        temp_owned = False

    try:
        for seq in args.seqs:
            for dim in args.dims:
                for dtype in args.dtypes:
                    # Stable, PYTHONHASHSEED-independent per-case seed.
                    case_seed = args.seed + (
                        seq * 1000003 + dim * 104729 + len(dtype) * 131071
                    ) % (2**31)

                    # Reference peak memory cap.
                    b, h, n, d = args.batch, args.heads, seq, dim
                    ref_bytes = 6 * 8 * b * h * n * n + 4 * 8 * b * h * n * d
                    if ref_bytes > args.mem_cap_bytes:
                        case = {
                            "seq": seq, "dim": dim, "dtype": dtype,
                            "case_seed": case_seed,
                            "error": {
                                "type": "MemoryError",
                                "message": f"float64 reference peak ~{ref_bytes/1e9:.2f} GiB exceeds --mem-cap-bytes={args.mem_cap_bytes}",
                            },
                            "results": {},
                        }
                        document["cases"].append(case)
                        _save_document(args.output, document)
                        print(f"=== seq={seq} dim={dim} dtype={dtype} SKIPPED (memory cap) ===", flush=True)
                        continue

                    rng = np.random.default_rng(case_seed)
                    # Randomized backend execution order (different stream).
                    order_rng = np.random.default_rng(case_seed + 17)
                    order = list(args.backends)
                    order_rng.shuffle(order)

                    case_dir = os.path.join(temp_root, f"s{seq}_d{dim}_{dtype}")
                    os.makedirs(case_dir, exist_ok=True)

                    print(f"=== seq={seq} dim={dim} dtype={dtype} order={order} ===", flush=True)
                    case = run_case(args, binary, case_dir, seq, dim, dtype,
                                    rng, case_seed, order)
                    document["cases"].append(case)
                    for name, rec in case["results"].items():
                        status = rec.get("status")
                        if status == "ok":
                            med = rec.get("wall_timing", {}).get("median_ms")
                            print(f"    {name}: wall_median={med}ms "
                                  f"pass={rec['comparison'].get('within_tolerance')}",
                                  flush=True)
                        else:
                            print(f"    {name}: {status} "
                                  f"{rec.get('error', {}).get('type')}",
                                  flush=True)
                    _save_document(args.output, document)
    finally:
        if temp_owned and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    _save_document(args.output, document)
    print(f"Done. Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
