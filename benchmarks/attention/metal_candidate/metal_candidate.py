#!/usr/bin/env python3
"""Standalone Metal FP32 causal-prefill attention candidate benchmark harness.

Runs the correctness-first candidate Metal kernel (metal_candidate.swift) across
a grid of sequence lengths, head dims, mappings, query tiles and key tiles, and
validates every output against an independent numpy float64 stable causal
reference.

Two mappings are supported (both keep the identical flash-style streaming
online-softmax schedule):
  * "scalar": one thread per query row, query_tile 8 or 16.
  * "simdgroup": one 32-lane SIMD-group per query row, query_tile 4 or 8.

Invalid (mapping, query_tile) pairs are skipped deterministically. Defaults are
focused on the simdgroup mapping at query tile 4/8 with key tile 8/16; scalar
remains available as a smoke path.

The candidate is a flash-style streaming online-softmax kernel, explicitly NOT
a claimed vendor-speedup baseline. This harness never substitutes a baseline or
another tile; any correctness failure is recorded (and, in --verify-only, made a
hard process error) with the exact configuration and error.

Frozen numerical tolerances for the FP32 contract: atol=1e-4, rtol=1e-3.

Run from anywhere:
  PYTHONPATH=. python3 benchmarks/attention/metal_candidate/metal_candidate.py \
      --output metal_candidate.json \
      --seqs 128,512 --dims 192,256 \
      --mappings simdgroup --query-tiles 4,8 --key-tiles 8,16 \
      --verify-only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE = "float32"
TOLERANCES = {DTYPE: {"atol": 1.0e-4, "rtol": 1.0e-3}}

DEFAULT_SEQS = "128,512"
DEFAULT_DIMS = "192,256"
# Defaults are focused on the SIMD-group mapping at query tile 4/8 with key
# tile 8/16. The scalar mapping (query tile 8/16) remains available as a smoke
# path via --mappings scalar --query-tiles 8,16.
DEFAULT_MAPPINGS = "simdgroup"
DEFAULT_QUERY_TILES = "4,8"
DEFAULT_KEY_TILES = "8,16"

# Allowed query tiles per mapping (frozen kernel contract in the Swift runner).
ALLOWED_QUERY_TILES = {4, 8, 16}
QUERY_TILES_BY_MAPPING = {"scalar": {8, 16}, "simdgroup": {4, 8}}
ALLOWED_KEY_TILES = {8, 16}

# Rough cap on the float64 reference peak (bytes). Computed per head to keep the
# peak at O(N^2) rather than O(B*H*N^2).
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


def _parse_tile_list(text, what, allowed):
    parts = _parse_int_list(text, what, minimum=1)
    for v in parts:
        if v not in allowed:
            raise argparse.ArgumentTypeError(f"--{what} value {v} not in {allowed}")
    return parts


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    default_build_dir = os.path.join(root, ".lake", "metal_candidate_worker")
    p = argparse.ArgumentParser(
        description="Standalone Metal FP32 causal-prefill attention candidate "
                    "kernel benchmark harness (vs numpy float64 reference).")
    p.add_argument("--output", default="metal_candidate.json",
                   help="output JSON path (incremental writes after each config)")
    p.add_argument("--seqs", default=DEFAULT_SEQS,
                   help="comma-separated sequence lengths")
    p.add_argument("--dims", default=DEFAULT_DIMS,
                   help="comma-separated head dims (only 192 or 256 allowed)")
    p.add_argument("--mappings", default=DEFAULT_MAPPINGS,
                   help="comma-separated mappings: scalar and/or simdgroup")
    p.add_argument("--query-tiles", default=DEFAULT_QUERY_TILES,
                   help="comma-separated query tile sizes (only 4, 8, or 16); "
                        "invalid (mapping, query_tile) pairs are skipped")
    p.add_argument("--key-tiles", default=DEFAULT_KEY_TILES,
                   help="comma-separated key tile sizes (only 8 or 16)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--verify-only", action="store_true",
                   help="run correctness verification (default minimal samples) "
                        "and exit non-zero if any configuration fails")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mem-cap-bytes", type=int, default=MEMORY_CAP_BYTES,
                   help="refuse cases whose float64 reference peak exceeds this")
    p.add_argument("--swift-src", default=os.path.join(here, "metal_candidate.swift"),
                   help="path to the Metal candidate Swift runner source")
    p.add_argument("--build-dir", default=default_build_dir,
                   help="directory for the compiled Swift binary (created if "
                        "needed); default is <repo>/.lake/metal_candidate_worker")
    p.add_argument("--temp-dir", default=None,
                   help="directory for per-config binaries; a temp dir is used "
                        "if not given")
    p.add_argument("--keep-temp", action="store_true",
                   help="do not delete the temporary binary directory")
    args = p.parse_args(argv)

    try:
        args.seqs = _parse_int_list(args.seqs, "seqs")
        args.dims = _parse_int_list(args.dims, "dims")
        args.mappings = [m.strip() for m in args.mappings.split(",")
                         if m.strip() != ""]
        for m in args.mappings:
            if m not in QUERY_TILES_BY_MAPPING:
                p.error(f"--mappings value {m!r} not in "
                        f"{sorted(QUERY_TILES_BY_MAPPING)}")
        args.query_tiles = _parse_tile_list(args.query_tiles, "query-tiles",
                                            ALLOWED_QUERY_TILES)
        args.key_tiles = _parse_tile_list(args.key_tiles, "key-tiles",
                                          ALLOWED_KEY_TILES)

        # Fail clearly when the requested mappings and query tiles leave no
        # valid (mapping, query_tile) pair, i.e. zero runnable configurations.
        valid_pairs = [
            (m, q)
            for m in args.mappings
            for q in args.query_tiles
            if q in QUERY_TILES_BY_MAPPING[m]
        ]
        if not valid_pairs:
            p.error(
                "--mappings/--query-tiles select no runnable configuration: "
                "no (mapping, query_tile) pair is valid for "
                f"mappings={args.mappings} query-tiles={args.query_tiles} "
                f"(valid pairs per mapping: {QUERY_TILES_BY_MAPPING})"
            )
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))

    for name in ("warmup", "samples", "inner", "heads", "batch"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be >= 1 (got {getattr(args, name)})")
    if args.heads != 8:
        p.error("--heads must be 8 (frozen contract)")
    if args.batch != 1:
        p.error("--batch must be 1 (frozen contract)")
    if args.mem_cap_bytes < 1:
        p.error("--mem-cap-bytes must be >= 1")

    if args.verify_only and args.samples > 1:
        # Verify-only does not care about timing; shrink to a single timed run
        # so compilation + one invocation still happens and correctness is checked.
        args.samples = 1
    return args


# ---------------------------------------------------------------------------
# Swift compilation
# ---------------------------------------------------------------------------

def compile_swift(swift_src, build_dir):
    os.makedirs(build_dir, exist_ok=True)
    binary = os.path.join(build_dir, "metal_candidate")
    src = os.path.abspath(swift_src)
    if not os.path.isfile(src):
        sys.exit(f"Swift source not found: {src}")
    if os.path.isfile(binary) and os.path.getmtime(binary) >= os.path.getmtime(src):
        return binary
    cmd = ["xcrun", "swiftc", "-O", "-framework", "Metal",
           "-framework", "MetalPerformanceShaders", src, "-o", binary]
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
# Reference computation (numpy float64, stable causal softmax, per head)
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, scale):
    """float64 causal prefill attention from (B,H,N,D) float64 arrays.

    Computed head-by-head so the peak memory stays O(N^2) rather than
    O(B*H*N^2). Independently implements the stable max-subtracted softmax,
    not the kernel's streaming online-softmax.
    """
    b, h, n, d = q.shape
    out = np.zeros((b, h, n, d), dtype=np.float64)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # True where j > i
    for bi in range(b):
        for hi in range(h):
            scores = np.matmul(q[bi, hi], k[bi, hi].T) * scale  # (N,N)
            scores = np.where(upper, -np.inf, scores)
            m = scores.max(axis=-1, keepdims=True)
            e = np.exp(scores - m)
            e = np.where(np.isneginf(scores), 0.0, e)
            s = e.sum(axis=-1, keepdims=True)
            attn = e / s
            out[bi, hi] = np.matmul(attn, v[bi, hi])
    return out


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
# Runner invocation
# ---------------------------------------------------------------------------

def run_candidate(binary, config_dir, shape, mapping, query_tile, key_tile,
                  warmup, samples, inner, q_bin, k_bin, v_bin):
    """Run the compiled Swift runner once; returns (resp, out_array)."""
    tag = f"{mapping}_qt{query_tile}_kt{key_tile}"
    out_bin = os.path.join(config_dir, f"out_{tag}.bin")
    resp_path = os.path.join(config_dir, f"resp_{tag}.json")
    req = {
        "shape": list(shape),
        "dtype": DTYPE,
        "mapping": mapping,
        "q_path": q_bin,
        "k_path": k_bin,
        "v_path": v_bin,
        "output_path": out_bin,
        "response_path": resp_path,
        "query_tile": query_tile,
        "key_tile": key_tile,
        "warmup": warmup,
        "samples": samples,
        "inner": inner,
    }
    req_path = os.path.join(config_dir, f"req_{tag}.json")
    with open(req_path, "w") as fh:
        json.dump(req, fh)

    proc = subprocess.run([binary, req_path], capture_output=True, text=True,
                          timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"swift runner exited {proc.returncode}: "
                           f"{proc.stdout}\n{proc.stderr}")

    with open(resp_path) as fh:
        resp = json.load(fh)
    if resp.get("status") != "ok":
        raise RuntimeError(f"swift runner reported status={resp.get('status')}: "
                           f"{resp.get('error')}")

    if not os.path.isfile(out_bin):
        raise RuntimeError("swift runner reported ok but wrote no output file")
    out = np.frombuffer(open(out_bin, "rb").read(),
                        dtype=np.float32).reshape(shape)
    return resp, out


def summarize_ms(samples):
    a = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
        "samples_ms": [float(x) for x in a],
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def collect_meta(args):
    return {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "swift_toolchain": swift_toolchain_version(),
        "swift_src": os.path.abspath(args.swift_src),
        "build_dir": os.path.abspath(args.build_dir),
    }


def sha256_hex(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Case handling
# ---------------------------------------------------------------------------

def run_config(args, binary, config_dir, shape, mapping, query_tile, key_tile,
               ref, q_bin, k_bin, v_bin):
    """Run one (mapping, qt, kt) configuration; returns the result record."""
    rec = {
        "mapping": mapping,
        "query_tile": query_tile,
        "key_tile": key_tile,
    }
    try:
        resp, out = run_candidate(binary, config_dir, shape, mapping,
                                  query_tile, key_tile,
                                  args.warmup, args.samples, args.inner,
                                  q_bin, k_bin, v_bin)
        rec["output_shape"] = list(out.shape)
        rec["comparison"] = compare_outputs(out, ref, TOLERANCES[DTYPE])
        rec["device"] = resp.get("device")
        rec["pipeline"] = resp.get("pipeline")
        rec["launch"] = resp.get("launch")
        rec["mtl_device_name"] = resp.get("mtl_device_name")
        rec["compile_wall_ms"] = resp.get("compile_wall_ms")
        rec["warmup_count"] = resp.get("warmup_count")
        rec["scale"] = resp.get("scale")
        rec["time_unit"] = resp.get("time_unit")
        rec["measurement_notes"] = resp.get("measurement_notes", [])
        rec["wall_timing"] = summarize_ms(resp.get("wall_times", []))

        gpu_avail = bool(resp.get("gpu_time_available"))
        rec["gpu_time_available"] = gpu_avail
        if gpu_avail:
            rec["gpu_timing"] = summarize_ms(resp.get("gpu_times", []))
        else:
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


def run_case(args, binary, config_dir, seq, dim, rng, case_seed, config_order):
    """Run one (seq, dim) case for the given interleaved config order.

    config_order is a list of (mapping, query_tile, key_tile) tuples.
    """
    b, h, n, d = args.batch, args.heads, seq, dim
    shape = (b, h, n, d)
    scale = math.pow(float(dim), -0.5)

    # Distinct Q, K, V rounded directly to float32.
    q_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    k_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    v_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)

    # Independent float64 reference, once per case.
    ref = reference_attention(q_d.astype(np.float64),
                              k_d.astype(np.float64),
                              v_d.astype(np.float64), scale)

    case = {
        "seq": seq,
        "dim": dim,
        "dtype": DTYPE,
        "heads": h,
        "batch": b,
        "scale": scale,
        "seed": args.seed,
        "case_seed": case_seed,
        "config_order": [list(t) for t in config_order],
        "inputs_sha256": {
            "q": sha256_hex(q_d),
            "k": sha256_hex(k_d),
            "v": sha256_hex(v_d),
        },
        "results": {},
    }

    q_bin = os.path.join(config_dir, "q.bin")
    k_bin = os.path.join(config_dir, "k.bin")
    v_bin = os.path.join(config_dir, "v.bin")
    q_d.tofile(q_bin)
    k_d.tofile(k_bin)
    v_d.tofile(v_bin)

    for mapping, query_tile, key_tile in config_order:
        name = f"{mapping}_qt{query_tile}_kt{key_tile}"
        rec = run_config(args, binary, config_dir, shape, mapping, query_tile,
                         key_tile, ref, q_bin, k_bin, v_bin)
        case["results"][name] = rec
    return case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sanitize(obj):
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

    binary = compile_swift(args.swift_src, args.build_dir)

    meta = collect_meta(args)
    meta["grid"] = {
        "seqs": args.seqs,
        "dims": args.dims,
        "mappings": args.mappings,
        "query_tiles": args.query_tiles,
        "key_tiles": args.key_tiles,
        "heads": args.heads,
        "batch": args.batch,
        "warmup": args.warmup,
        "samples": args.samples,
        "inner": args.inner,
        "seed": args.seed,
        "verify_only": args.verify_only,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    document = {
        "meta": meta,
        "timing_caveats": [
            "All timings are in milliseconds.",
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per "
            "invocation (exclude input/output CPU copies and compile/warmup). "
            "If timestamps were invalid they are reported unavailable "
            "(gpu_timing=null), never mislabeled.",
            "Wall times are synchronized API latency, timed from before "
            "command-buffer allocation through waitUntilCompleted.",
            "Each timed invocation computes a fresh output; nothing is cached "
            "across samples. samples x inner per-invocation timings are kept in "
            "wall_timing.samples_ms (and gpu_timing.samples_ms).",
            "Configurations (mapping, query_tile, key_tile) run serially per "
            "case in a deterministic, seed-shuffled order (recorded in "
            "config_order); invalid (mapping, query_tile) pairs are skipped "
            "deterministically (recorded in skipped_configs).",
            "This candidate is a correctness-first flash-style kernel; no "
            "vendor speedup is claimed.",
        ],
        "cases": [],
    }

    temp_root = args.temp_dir
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix="metal_candidate_")
        temp_owned = True
    else:
        os.makedirs(temp_root, exist_ok=True)
        temp_owned = False

    try:
        for seq in args.seqs:
            for dim in args.dims:
                # Same per-case seed derivation as metal_survey.py / cuda_survey.py
                # for float32, so an identical --seed yields byte-identical q/k/v
                # inputs across the candidate and the vendor survey.
                case_seed = args.seed + (
                    seq * 1000003 + dim * 104729 + len(DTYPE) * 131071
                ) % (2**31)

                # float64 reference peak: per-head O(N^2) intermediates plus inputs.
                ref_bytes = 8 * 8 * seq * seq + 3 * 8 * seq * dim
                if ref_bytes > args.mem_cap_bytes:
                    case = {
                        "seq": seq, "dim": dim, "dtype": DTYPE,
                        "case_seed": case_seed,
                        "error": {
                            "type": "MemoryError",
                            "message": f"float64 reference peak ~{ref_bytes/1e9:.2f} GiB exceeds --mem-cap-bytes={args.mem_cap_bytes}",
                        },
                        "results": {},
                    }
                    document["cases"].append(case)
                    _save_document(args.output, document)
                    print(f"=== seq={seq} dim={dim} SKIPPED (memory cap) ===", flush=True)
                    continue

                rng = np.random.default_rng(case_seed)
                # Deterministic interleaving of (mapping, query_tile, key_tile)
                # configurations (different stream). Invalid (mapping,
                # query_tile) pairs are skipped deterministically.
                order_rng = np.random.default_rng(case_seed + 17)
                all_configs = [
                    (m, q, k)
                    for m in args.mappings
                    for q in args.query_tiles
                    for k in args.key_tiles
                    if q in QUERY_TILES_BY_MAPPING[m]
                ]
                order_rng.shuffle(all_configs)
                config_order = all_configs
                skipped = []
                for m in args.mappings:
                    for q in args.query_tiles:
                        if q not in QUERY_TILES_BY_MAPPING[m]:
                            skipped.append((m, q))

                config_dir = os.path.join(temp_root, f"s{seq}_d{dim}")
                os.makedirs(config_dir, exist_ok=True)

                print(f"=== seq={seq} dim={dim} config_order={config_order} "
                      f"skipped={skipped} ===", flush=True)
                case = run_case(args, binary, config_dir, seq, dim, rng,
                                case_seed, config_order)
                case["skipped_configs"] = [list(t) for t in skipped]
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

    if args.verify_only:
        failures = []
        for case in document["cases"]:
            for name, rec in case.get("results", {}).items():
                if rec.get("status") != "ok":
                    failures.append((case["seq"], case["dim"], name, rec.get("status")))
        if failures:
            for seq, dim, name, status in failures:
                print(f"VERIFY FAIL: seq={seq} dim={dim} {name}: {status}", flush=True)
            return 1
        print("VERIFY: all configurations passed correctness checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
