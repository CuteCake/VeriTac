#!/usr/bin/env python3
"""Standalone CUDA attention baseline survey.

Benchmarks torch SDPA backends (auto, cudnn, flash, efficient, math) on the
causal prefill attention operator across a grid of sequence lengths, head
dims and dtypes, with a numpy float64 stable-softmax reference for numerical
comparison. Timing uses CUDA events (repeated fresh computations per inner
batch), a synchronized wall-clock series, and CUDA graph replay.

Run from anywhere; requires a CUDA-capable PyTorch build and numpy. No other
dependencies. Everything is written to the JSON file given by --output.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

# ---------------------------------------------------------------------------
# Backend / dtype definitions
# ---------------------------------------------------------------------------

BACKEND_ORDER = ["auto", "cudnn", "flash", "efficient", "math"]

BACKEND_MAP = {
    "auto": None,  # all backends eligible (torch default selection)
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
}

NP_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
}

# Frozen numerical tolerances per dtype.
TOLERANCES = {
    "float32": {"atol": 1.0e-4, "rtol": 1.0e-3},
    "float16": {"atol": 2.0e-3, "rtol": 2.0e-2},
}

# Rough memory cap (bytes) for a single case's live tensors (Q,K,V,out) plus
# the numpy float64 reference intermediates. A safety valve against absurd
# shapes being requested by the caller.
MEMORY_CAP_BYTES = 20 * (2**30)  # 20 GiB

# Default CUDA grid.
DEFAULT_GRID = "256,1024"
DEFAULT_DIMS = "64,128,192,256"
DEFAULT_DTYPES = "float32,float16"


# ---------------------------------------------------------------------------
# Argument parsing / validation
# ---------------------------------------------------------------------------

def _parse_int_list(text: str, what: str, minimum: int = 1) -> list[int]:
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
            raise argparse.ArgumentTypeError(
                f"--{what} value {v} must be >= {minimum}")
        out.append(v)
    return out


def _parse_dtypes(text: str) -> list[str]:
    parts = [p.strip().lower() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError("--dtypes must be a non-empty comma list")
    for p in parts:
        if p not in DTYPE_MAP:
            raise argparse.ArgumentTypeError(
                f"--dtypes value {p!r} not in {sorted(DTYPE_MAP)}")
    return parts


def _parse_backends(text: str) -> list[str]:
    parts = [p.strip().lower() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError("--backends must be a non-empty comma list")
    for p in parts:
        if p not in BACKEND_ORDER:
            raise argparse.ArgumentTypeError(
                f"--backends value {p!r} not in {BACKEND_ORDER}")
    return parts


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="CUDA attention baseline survey (SDPA backends vs numpy "
                    "float64 reference).")
    p.add_argument("--output", default="cuda_survey.json",
                   help="output JSON path (incremental writes after each result)")
    p.add_argument("--seqs", default=DEFAULT_GRID,
                   help="comma-separated sequence lengths")
    p.add_argument("--dims", default=DEFAULT_DIMS,
                   help="comma-separated head dims")
    p.add_argument("--dtypes", default=DEFAULT_DTYPES,
                   help="comma-separated dtypes")
    p.add_argument("--backends", default=",".join(BACKEND_ORDER),
                   help="comma-separated backends to run")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mem-cap-bytes", type=int, default=MEMORY_CAP_BYTES,
                   help="refuse cases whose live tensors exceed this many bytes")
    args = p.parse_args(argv)

    try:
        args.seqs = _parse_int_list(args.seqs, "seqs")
        args.dims = _parse_int_list(args.dims, "dims")
        args.dtypes = _parse_dtypes(args.dtypes)
        args.backends = _parse_backends(args.backends)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    for name in ("heads", "batch", "warmup", "samples", "inner"):
        v = getattr(args, name)
        if v < 1:
            p.error(f"--{name} must be >= 1 (got {v})")
    if args.mem_cap_bytes < 1:
        p.error("--mem-cap-bytes must be >= 1")
    return args


def case_seed_of(base_seed, seq, dim, dtype):
    """Deterministic per-case seed shared with other harnesses."""
    return base_seed + (seq * 1000003 + dim * 104729 + len(dtype) * 131071) % (2**31)


# ---------------------------------------------------------------------------
# Reference computation (numpy float64, stable causal softmax)
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, scale):
    """numpy float64 causal prefill attention.

    q, k, v are float64 arrays of shape (B, H, N, D). Returns float64
    (B, H, N, D) using a numerically stable causal softmax.
    """
    b, h, n, d = q.shape
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale  # (B,H,N,N)
    # Causal: out[i] attends only to j <= i. Mask j > i to -inf.
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # True where j > i
    scores = np.where(upper[None, None], -np.inf, scores)

    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)            # -inf -> 0 automatically
    e = np.where(np.isneginf(scores), 0.0, e)
    s = e.sum(axis=-1, keepdims=True)
    attn = e / s
    return np.matmul(attn, v)


def compute_reference(q_t, k_t, v_t, scale):
    """Reference computed from the actual dtype-rounded inputs (as on GPU)."""
    q = q_t.detach().cpu().numpy().astype(np.float64)
    k = k_t.detach().cpu().numpy().astype(np.float64)
    v = v_t.detach().cpu().numpy().astype(np.float64)
    return reference_attention(q, k, v, scale)


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------

def compare_outputs(out_t, ref, tol):
    """Compare every output value against the reference.

    Returns a dict with finite check, max_abs error, normalized max error
    max(abs(err)/(atol+rtol*abs(ref))) and RMS error, plus pass/fail.
    """
    out = out_t.detach().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(out).all())
    finite_ref = bool(np.isfinite(ref).all())

    err = np.abs(out - ref)
    max_abs = float(err.max()) if finite_ref and finite else float("nan")
    denom = tol["atol"] + tol["rtol"] * np.abs(ref)
    normalized = err / denom
    normalized_max = (
        float(normalized.max()) if finite_ref and finite else float("nan"))
    rms = float(np.sqrt(np.mean(np.square(err)))) if finite_ref and finite else float("nan")

    passed = bool(
        finite and finite_ref and np.all(out <= ref + tol["atol"] + tol["rtol"] * np.abs(ref))
        and np.all(out >= ref - tol["atol"] - tol["rtol"] * np.abs(ref)))

    return {
        "finite_out": finite,
        "finite_ref": finite_ref,
        "max_abs_error": max_abs,
        "normalized_max_error": normalized_max,
        "rms_error": rms,
        "within_tolerance": passed,
    }


# ---------------------------------------------------------------------------
# Backend session: one sdpa_kernel context held open for the whole backend
# measurement (dispatch + eager timing + wall timing + CUDA graph capture), so
# the timed callable is the raw F.scaled_dot_product_attention call without
# re-entering the context on every iteration.
# ---------------------------------------------------------------------------

def _backend_context(backend):
    backends = BACKEND_MAP[backend]
    if backends is None:
        return nullcontext()
    return sdpa_kernel(backends)


_SDPA_ACCEPTS_SCALE = None


class BackendSession:
    def __init__(self, backend):
        self.backend = backend
        self._ctx = _backend_context(backend)

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)

    def raw_sdpa(self, q, k, v, scale):
        """Raw F.scaled_dot_product_attention; the sdpa_kernel context is
        already held open by the session."""
        global _SDPA_ACCEPTS_SCALE
        if _SDPA_ACCEPTS_SCALE is None:
            try:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                    scale=scale)
                _SDPA_ACCEPTS_SCALE = True
                return out
            except TypeError:
                _SDPA_ACCEPTS_SCALE = False
        if _SDPA_ACCEPTS_SCALE:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                scale=scale)
        # Older torch without the `scale` argument: default scale is D**-0.5
        # which is identical to our explicit scale for head dim D.
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

    def identify_dispatch(self, q, k, v, scale):
        """Profile CPU activities once to discover actual aten dispatch names
        (context held open, same dispatch as measured)."""
        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                self.raw_sdpa(q, k, v, scale)
            names = sorted({evt.key for evt in prof.key_averages()})
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}", "names": []}
        return {"error": None, "names": names}

    def time_eager(self, q, k, v, scale, warmup, samples, inner):
        """CUDA-event timing (repeated fresh outputs per inner batch) plus a
        separate synchronized wall-clock series for a single call."""
        for _ in range(warmup):
            self.raw_sdpa(q, k, v, scale)
        torch.cuda.synchronize()

        event_samples = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                self.raw_sdpa(q, k, v, scale)
            end.record()
            torch.cuda.synchronize()
            event_samples.append(start.elapsed_time(end) / float(inner))

        wall_samples = []
        for _ in range(samples):
            t0 = time.perf_counter()
            self.raw_sdpa(q, k, v, scale)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            wall_samples.append((t1 - t0) * 1.0e3)

        return event_samples, wall_samples

    def capture_graph(self, q, k, v, scale, inner, warmup):
        """Capture `inner` fresh SDPA calls into one CUDA graph. Warmup runs on
        a side stream first so the memory pool is warm before capture. Returns
        (graph, outputs) where outputs keep the captured outputs alive."""
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.raw_sdpa(q, k, v, scale)
        torch.cuda.synchronize()

        outputs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                outputs.append(self.raw_sdpa(q, k, v, scale))
        return graph, outputs

    @staticmethod
    def time_graph(graph, inner, warmup, samples):
        """Replay-warmup then CUDA-event time graph.replay(); divided by inner
        for ms/invocation."""
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples_ms = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end) / float(inner))
        return samples_ms


def summarize_ms(samples):
    a = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
        "samples_ms": [float(x) for x in a],
    }


def classify_error(exc):
    """Map an exception to a backend status: unsupported vs failed."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "NotImplementedError":
        return "unsupported"
    if name == "RuntimeError" and any(
            tok in msg for tok in ("not support", "unsupported", "no available",
                                   "no supported", "unimplemented")):
        return "unsupported"
    return "failed"


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def _opt_get(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception:  # noqa: BLE001
        return None


def _bool_or_none(value):
    return bool(value) if value is not None else None


def collect_meta():
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    meta["torch_cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        meta["cuda_version"] = _opt_get(torch.version, "cuda")
        meta["cudnn_version"] = _opt_get(torch.backends.cudnn, "version")()
        props = torch.cuda.get_device_properties(0)
        meta["device"] = {
            "name": getattr(props, "name", None),
            "major": getattr(props, "major", None),
            "minor": getattr(props, "minor", None),
            "sm_count": getattr(props, "multi_processor_count", None),
            "total_memory_bytes": getattr(props, "total_memory", None),
            "capability": (
                f"{getattr(props, 'major', 0)}.{getattr(props, 'minor', 0)}"
                if getattr(props, "major", None) is not None else None),
        }
        meta["device_name"] = torch.cuda.get_device_name(0)
    else:
        meta["cuda_version"] = None
        meta["cudnn_version"] = None
        meta["device"] = None

    meta["tf32_settings"] = {
        "matmul_allow_tf32": _bool_or_none(
            _opt_get(torch.backends.cuda.matmul, "allow_tf32")),
        "cudnn_allow_tf32": _bool_or_none(
            _opt_get(torch.backends.cudnn, "allow_tf32")),
    }
    meta["fp16_precision"] = {
        "matmul_allow_fp16_reduced_precision_reduction": _bool_or_none(
            _opt_get(torch.backends.cuda.matmul,
                     "allow_fp16_reduced_precision_reduction")),
        "cudnn_allow_fp16_reduced_precision_reduction": _bool_or_none(
            _opt_get(torch.backends.cudnn,
                     "allow_fp16_reduced_precision_reduction")),
    }
    return meta


# ---------------------------------------------------------------------------
# Case handling
# ---------------------------------------------------------------------------

def memory_estimate(seq, dim, heads, batch, dtype):
    b, h, n, d = batch, heads, seq, dim
    numel = b * h * n * d
    elemsize = torch.tensor([], dtype=DTYPE_MAP[dtype]).element_size()
    linear = 4 * numel * elemsize  # Q, K, V, output
    # numpy float64 reference intermediates (scores, exp, sum, attn, etc.).
    ref_intermediate = b * h * n * n * 8 * 6
    return linear + ref_intermediate


def _tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_case(args, seq, dim, dtype, order, case_seed_val, case, document):
    """Run one (seq, dim, dtype) case across the selected backends, filling
    `case` (already appended to `document["cases"]`) and saving the document
    incrementally after each backend result."""
    np_dtype = NP_DTYPE_MAP[dtype]
    scale = math.pow(float(dim), -0.5)

    case.update({
        "scale": scale,
        "case_seed": case_seed_val,
        "backend_order": order,
        "results": {},
    })

    # Deterministic inputs: input RNG (never used for backend order) draws
    # Q/K/V separately and casts directly to the numpy target dtype (float64
    # -> float16/float32) without an extra rounding, then to a CUDA tensor.
    b, h, n, d = args.batch, args.heads, seq, dim
    shape = (b, h, n, d)
    rng_in = np.random.default_rng(case_seed_val)
    q = torch.from_numpy(
        rng_in.normal(0.0, 0.5, size=shape).astype(np_dtype)).to("cuda").contiguous()
    k = torch.from_numpy(
        rng_in.normal(0.0, 0.5, size=shape).astype(np_dtype)).to("cuda").contiguous()
    v = torch.from_numpy(
        rng_in.normal(0.0, 0.5, size=shape).astype(np_dtype)).to("cuda").contiguous()
    case["input_hashes"] = {
        "q": _tensor_sha256(q),
        "k": _tensor_sha256(k),
        "v": _tensor_sha256(v),
    }

    print(f"[{seq},{dim},{dtype}] seed={case_seed_val} computing float64 reference ...",
          flush=True)
    ref = compute_reference(q, k, v, scale)

    with torch.inference_mode():
        for backend in order:
            rec = {}
            try:
                with BackendSession(backend) as session, \
                        warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    try:
                        out = session.raw_sdpa(q, k, v, scale)
                        torch.cuda.synchronize()
                        run_error = None
                    except Exception as exc:  # noqa: BLE001
                        out = None
                        run_error = exc
                    rec["warnings"] = [str(w.message) for w in wlist]

                    if run_error is not None:
                        rec["status"] = classify_error(run_error)
                        rec["error"] = {
                            "type": type(run_error).__name__,
                            "message": str(run_error),
                        }
                        case["results"][backend] = rec
                        _save_document(args.output, document)
                        print(f"    {backend}: {rec['status'].upper()} "
                              f"{type(run_error).__name__}", flush=True)
                        continue

                    rec["comparison"] = compare_outputs(
                        out, ref, TOLERANCES[dtype])
                    if not rec["comparison"]["within_tolerance"] or \
                            not rec["comparison"]["finite_out"]:
                        rec["status"] = "numeric_failed"
                    else:
                        rec["status"] = "ok"

                    rec["dispatch"] = session.identify_dispatch(q, k, v, scale)

                    try:
                        evt, wall = session.time_eager(
                            q, k, v, scale, args.warmup, args.samples, args.inner)
                        rec["cuda_event_timing"] = summarize_ms(evt)
                        rec["wall_timing"] = summarize_ms(wall)
                    except Exception as exc:  # noqa: BLE001
                        rec["timing_error"] = {
                            "type": type(exc).__name__, "message": str(exc)}

                    try:
                        graph, _outputs = session.capture_graph(
                            q, k, v, scale, args.inner, args.warmup)
                        graph_ms = BackendSession.time_graph(
                            graph, args.inner, args.warmup, args.samples)
                        torch.cuda.synchronize()
                        rec["graph_comparison"] = compare_outputs(
                            _outputs[-1], ref, TOLERANCES[dtype])
                        if not rec["graph_comparison"]["within_tolerance"] or \
                                not rec["graph_comparison"]["finite_out"]:
                            rec["graph_timing_error"] = {
                                "type": "NumericError",
                                "message": "captured CUDA graph output failed "
                                           "numerical comparison; graph_timing "
                                           "not retained as baseline"}
                            rec["status"] = "numeric_failed"
                        else:
                            rec["graph_timing"] = summarize_ms(graph_ms)
                    except Exception as exc:  # noqa: BLE001
                        rec["graph_timing_error"] = {
                            "type": type(exc).__name__, "message": str(exc)}

                    case["results"][backend] = rec
                    _save_document(args.output, document)
                    evt_med = rec.get("cuda_event_timing", {}).get("median_ms")
                    print(f"    {backend}: status={rec['status']} "
                          f"event_median={evt_med}ms "
                          f"pass={rec['comparison'].get('within_tolerance')}",
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "failed"
                rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
                case["results"][backend] = rec
                _save_document(args.output, document)
                print(f"    {backend}: FAILED {type(exc).__name__}", flush=True)

    return case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_document(args):
    meta = collect_meta()
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
    return {
        "meta": meta,
        "timing_caveats": [
            "CUDA event samples measure repeated fresh output computations "
            "(inner={}) and are divided by inner to give ms/invocation.".format(args.inner),
            "CUDA event timing includes the full SDPA op (all kernels); no "
            "input transfer, no reference computation, no profiler attached.",
            "A separate synchronized wall-clock series is reported for a "
            "single call per sample.",
            "graph_timing is CUDA graph replay of inner fresh calls (warmup "
            "replays first), divided by inner; it removes most CPU enqueue "
            "gaps and reflects GPU throughput, not a raw per-kernel duration.",
            "The sdpa_kernel context is held open for the whole backend "
            "measurement; the timed callable is the raw F.sdpa call.",
            "Dispatch/operator names are profiled once per backend OUTSIDE "
            "the timed region, under the same forced backend context.",
        ],
        "cases": [],
    }


def main(argv=None):
    args = parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available; this survey requires CUDA.")

    # Consistency: disable TF32 for float32 and record the settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # Memory cap safety.
    for seq in args.seqs:
        for dim in args.dims:
            for dtype in args.dtypes:
                est = memory_estimate(seq, dim, args.heads, args.batch, dtype)
                if est > args.mem_cap_bytes:
                    sys.exit(
                        f"Refusing case seq={seq} dim={dim} {dtype}: "
                        f"estimated live memory {est/1e9:.2f} GiB exceeds "
                        f"--mem-cap-bytes={args.mem_cap_bytes}. Reduce shapes.")

    document = _build_document(args)
    _save_document(args.output, document)

    # Reproducible per-case backend order rotation, using a separate RNG that
    # never touches the input RNG.
    for seq in args.seqs:
        for dim in args.dims:
            for dtype in args.dtypes:
                cs = case_seed_of(args.seed, seq, dim, dtype)
                rng_order = np.random.default_rng(cs + 17)
                order = list(args.backends)
                rng_order.shuffle(order)

                print(f"=== seq={seq} dim={dim} dtype={dtype} order={order} ===",
                      flush=True)
                case = {
                    "seq": seq,
                    "dim": dim,
                    "dtype": dtype,
                    "heads": args.heads,
                    "batch": args.batch,
                }
                document["cases"].append(case)
                run_case(args, seq, dim, dtype, order, cs, case, document)

    _save_document(args.output, document)
    print(f"Done. Wrote {args.output}", flush=True)
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
