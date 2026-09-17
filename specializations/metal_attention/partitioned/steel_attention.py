#!/usr/bin/env python3
"""Metal FP32 causal-prefill attention benchmark for a VENDOR-DERIVED
specialization of the installed MLX "steel" attention template.

The executed Metal kernel is a template-instantiated specialization of the
vendor MLX steel flash-attention GEMM kernel (Copyright (c) 2024-25 Apple Inc.,
MIT).  This is NOT an original generated kernel: it is a resource/schedule
specialization experiment of the existing vendor kernel, labeled honestly as
vendor-derived.  Attribution and license text are preserved in the generated
MSL source.

The harness:
  1. Flattens the installed mlx steel headers (recursively inlined, keeping the
     system Metal includes) into a single self-contained MSL source file,
     appending explicit template instantiations with host_names for every
     (qt, kt, D, WM, WN) config whose threadgroup resource fits 32 KiB.
  2. Invokes the Swift runner (steel_attention.swift) which reads that source
     via makeLibrary(source:) and builds the specialized pipeline.
  3. Validates every output against an independent numpy float64 stable causal
     reference (atol=1e-4, rtol=1e-3).  Input seed/hash contract is byte-
     identical to metal_survey.py / metal_candidate.py.

Config -> specialization mapping: qt (BQ) -> WM=qt/8, WN=1; kt (BK); D (BD).
Resource formula (vendor header): Q=BQ*(D+4)*4, KV=max((BK+4)*D,BK*(D+4))*4.

Run from anywhere:
  /tmp/veritac-survey-venv/bin/python benchmarks/attention/partitioned/steel_attention.py \
      --output steel_sweep.json --seqs 128,100 --dims 192,256 --qts 8,16,24 --kts 8,16
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
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

DTYPE = "float32"
TOLERANCES = {DTYPE: {"atol": 1.0e-4, "rtol": 1.0e-3}}

DEFAULT_SEQS = "128,100"
DEFAULT_DIMS = "192,256"
DEFAULT_QTS = "8,16,24"
DEFAULT_KTS = "8,16"

ALLOWED_QTS = {8, 16, 24}
ALLOWED_KTS = {8, 16}

# Build include root from the mlx package installation (provenance only).
import importlib.util as _ilu

_mlx_spec = _ilu.find_spec("mlx")
_mlx_root = (_mlx_spec.submodule_search_locations[0]
             if _mlx_spec and _mlx_spec.submodule_search_locations
             else os.path.dirname(_mlx_spec.origin))
MLX_INCLUDE = os.path.join(_mlx_root, "include")
MLX_STEEL_TOP = os.path.join(
    MLX_INCLUDE, "mlx", "backend", "metal", "kernels", "steel",
    "attn", "kernels", "steel_attention.h")

MEMORY_CAP_BYTES = 20 * (2**30)


def _parse_int_list(text, what, minimum=1, allowed=None):
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
        if allowed is not None and v not in allowed:
            raise argparse.ArgumentTypeError(f"--{what} value {v} not in {allowed}")
        out.append(v)
    return out


def tgmem_bytes(qt, kt, d):
    qbytes = qt * (d + 4) * 4
    kvbytes = max((kt + 4) * d, kt * (d + 4)) * 4
    return qbytes + kvbytes


def valid_configs(qts, kts, dims):
    """(qt, kt, d) configs that fit 32 KiB and satisfy BQ=WM*WN*8, TQ==1."""
    out = []
    for qt in qts:
        if qt % 8 != 0:
            continue
        wm = qt // 8
        for kt in kts:
            if kt % 8 != 0:
                continue
            for d in dims:
                if d % 8 != 0:
                    continue
                if tgmem_bytes(qt, kt, d) <= 32768:
                    out.append((qt, kt, d, wm, 1))
    return out


def flatten_includes(start_path, include_root):
    """Recursively inline all `#include "mlx/..."` headers into one string.

    System angle-bracket includes (<metal_stdlib> etc.) are left as-is.  Each
    mlx file is emitted exactly once (post-order, dependencies first), honoring
    the headers' `#pragma once` semantics.
    """
    visited = set()
    chunks = []
    inc_re = re.compile(r'^\s*#\s*include\s+"(mlx/[^"]+)"\s*$')

    def rec(path):
        apath = os.path.normpath(path)
        if apath in visited:
            return
        visited.add(apath)
        with open(apath) as fh:
            lines = fh.read().splitlines()
        for ln in lines:
            m = inc_re.match(ln)
            if m:
                dep = os.path.join(include_root, m.group(1))
                rec(dep)
        chunks.append("// ==== " + os.path.relpath(apath, include_root) + " ====")
        chunks.append("\n".join(ln for ln in lines
                                if not inc_re.match(ln) and ln.strip() != "#pragma once"))

    rec(os.path.abspath(start_path))
    return "\n\n".join(chunks) + "\n"


def instantiation_block(configs):
    lines = []
    lines.append("// Vendor-derived explicit template instantiations.")
    lines.append("// attention<T, BQ, BK, BD, WM, WN, MaskType=float, AccumType=float>")
    for qt, kt, d, wm, wn in sorted(configs):
        name = f"steel_attn_f32_bq{qt}_bk{kt}_bd{d}_wm{wm}_wn{wn}"
        lines.append(
            f"template [[host_name(\"{name}\")]] [[kernel]] void "
            f"attention<float, {qt}, {kt}, {d}, {wm}, {wn}, float, float>(\n"
            "const device float*, const device float*, const device float*, device float*,\n"
            "const constant AttnParams*, const constant AttnMaskParams*,\n"
            "const device float*, const device float*, uint, uint, uint3, uint3);\n")
    return "\n".join(lines) + "\n"


def generate_kernel_source(mlx_top, include_root, configs, out_path):
    """Build and write the flattened vendor-derived MSL source."""
    header = flatten_includes(mlx_top, include_root)
    inst = instantiation_block(configs)
    preamble = (
        "// VENDOR-DERIVED METAL KERNEL\n"
        "//\n"
        "// This file contains template instantiations of the MLX 'steel'\n"
        "// attention kernel, Copyright (c) 2024-25 Apple Inc., distributed\n"
        "// under the MIT License (see the mlx package).  The headers below are\n"
        "// copied verbatim (recursively inlined) from the installed mlx\n"
        "// package.  It is a resource/schedule specialization experiment of the\n"
        "// existing vendor kernel, NOT an original generated kernel.\n"
    )
    # The normal MLX compilation unit supplies Limits through kernels/utils.h.
    # This float-only adapter supplies its required finite_min member with the
    # same value; no bf16/complex helper implementation is needed here.
    adapter = ("\n#include <metal_stdlib>\n"
               "template <typename T> struct Limits {\n"
               "static constexpr constant T finite_min = -metal::numeric_limits<T>::max();\n"
               "};\n")
    src = preamble + adapter + header + "\n" + inst
    with open(out_path, "w") as fh:
        fh.write(src)
    return src


def source_hashes(mlx_top, include_root, configs):
    """Record SHA-256 of every inlined header plus the generated source."""
    visited = set()
    inc_re = re.compile(r'^\s*#\s*include\s+"(mlx/[^"]+)"\s*$')

    def collect(path):
        apath = os.path.normpath(path)
        if apath in visited:
            return
        visited.add(apath)
        with open(apath) as fh:
            lines = fh.read().splitlines()
        for ln in lines:
            m = inc_re.match(ln)
            if m:
                collect(os.path.join(include_root, m.group(1)))

    collect(os.path.abspath(mlx_top))
    hashes = {}
    for p in sorted(visited):
        rel = os.path.relpath(p, include_root)
        hashes[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return hashes


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    default_build_dir = os.path.join(root, ".lake", "partitioned_worker")
    p = argparse.ArgumentParser(
        description="Vendor-derived MLX steel attention small-tile specialization "
                    "benchmark (vs numpy float64 reference).")
    p.add_argument("--output", default="steel_sweep.json")
    p.add_argument("--seqs", default=DEFAULT_SEQS)
    p.add_argument("--dims", default=DEFAULT_DIMS)
    p.add_argument("--qts", default=DEFAULT_QTS,
                   help="comma-separated query tiles (8,16,24)")
    p.add_argument("--kts", default=DEFAULT_KTS,
                   help="comma-separated key tiles (8,16)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mem-cap-bytes", type=int, default=MEMORY_CAP_BYTES)
    p.add_argument("--swift-src", default=os.path.join(here, "steel_attention.swift"))
    p.add_argument("--build-dir", default=default_build_dir)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--keep-temp", action="store_true")
    args = p.parse_args(argv)

    try:
        args.seqs = _parse_int_list(args.seqs, "seqs")
        args.dims = _parse_int_list(args.dims, "dims")
        args.qts = _parse_int_list(args.qts, "qts", allowed=ALLOWED_QTS)
        args.kts = _parse_int_list(args.kts, "kts", allowed=ALLOWED_KTS)
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
        args.samples = 1
    if not os.path.isfile(args.swift_src):
        p.error(f"--swift-src not found: {args.swift_src}")
    if not os.path.isfile(MLX_STEEL_TOP):
        p.error(f"MLX steel_attention.h not found at {MLX_STEEL_TOP}")
    return args


def compile_swift(swift_src, build_dir):
    os.makedirs(build_dir, exist_ok=True)
    binary = os.path.join(build_dir, "steel_attention")
    src = os.path.abspath(swift_src)
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


def reference_attention(q, k, v, scale):
    b, h, n, d = q.shape
    out = np.zeros((b, h, n, d), dtype=np.float64)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for bi in range(b):
        for hi in range(h):
            scores = np.matmul(q[bi, hi], k[bi, hi].T) * scale
            scores = np.where(upper, -np.inf, scores)
            m = scores.max(axis=-1, keepdims=True)
            e = np.exp(scores - m)
            e = np.where(np.isneginf(scores), 0.0, e)
            s = e.sum(axis=-1, keepdims=True)
            attn = e / s
            out[bi, hi] = np.matmul(attn, v[bi, hi])
    return out


def compare_outputs(out_np, ref, tol):
    out = np.asarray(out_np, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    shape_ok = bool(out.shape == ref.shape)
    if not shape_ok:
        return {"shape_ok": False, "output_shape": list(out.shape),
                "reference_shape": list(ref.shape), "error": "shape mismatch",
                "within_tolerance": False}
    finite_out = bool(np.isfinite(out).all())
    finite_ref = bool(np.isfinite(ref).all())
    err = np.abs(out - ref)
    denom = tol["atol"] + tol["rtol"] * np.abs(ref)
    normalized = err / denom
    if finite_out and finite_ref:
        within_tolerance = bool((err <= denom).all())
        max_abs = float(err.max())
        norm_max = float(normalized.max())
        rms = float(np.sqrt(np.mean(np.square(err))))
    else:
        within_tolerance = False
        max_abs = float("nan")
        norm_max = float("nan")
        rms = float("nan")
    return {"shape_ok": True, "output_shape": list(out.shape),
            "reference_shape": list(ref.shape), "finite_out": finite_out,
            "finite_ref": finite_ref, "max_abs_error": max_abs,
            "max_abs_error_normalized": norm_max, "rms_error": rms,
            "within_tolerance": within_tolerance}


def run_candidate(binary, config_dir, shape, qt, kt, kernel_path,
                  warmup, samples, inner, q_bin, k_bin, v_bin):
    tag = f"qt{qt}_kt{kt}"
    out_bin = os.path.join(config_dir, f"out_{tag}.bin")
    resp_path = os.path.join(config_dir, f"resp_{tag}.json")
    req = {
        "shape": list(shape), "dtype": DTYPE, "qt": qt, "kt": kt,
        "kernel_path": kernel_path,
        "q_path": q_bin, "k_path": k_bin, "v_path": v_bin,
        "output_path": out_bin, "response_path": resp_path,
        "warmup": warmup, "samples": samples, "inner": inner,
    }
    with open(os.path.join(config_dir, f"req_{tag}.json"), "w") as fh:
        json.dump(req, fh)
    proc = subprocess.run([binary, os.path.join(config_dir, f"req_{tag}.json")],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"swift runner exited {proc.returncode}: "
                           f"{proc.stdout}\n{proc.stderr}")
    with open(resp_path) as fh:
        resp = json.load(fh)
    resp["runner_stderr"] = proc.stderr
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
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)), "samples_ms": [float(x) for x in a]}


def collect_meta(args):
    try:
        mlx_ver = importlib.metadata.version("mlx")
    except Exception:  # noqa: BLE001
        mlx_ver = getattr(importlib.import_module("mlx"), "__version__", "unknown")
    return {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "mlx_version": mlx_ver,
        "mlx_include_root": MLX_INCLUDE,
        "swift_toolchain": swift_toolchain_version(),
        "swift_src": os.path.abspath(args.swift_src),
        "build_dir": os.path.abspath(args.build_dir),
        "metal_validation_environment": {
            key: os.environ.get(key) for key in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION")
        },
    }


def sha256_hex(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_config(args, binary, config_dir, shape, qt, kt, kernel_path,
               ref, q_bin, k_bin, v_bin):
    rec = {"qt": qt, "kt": kt,
           "tgmem_bytes": tgmem_bytes(qt, kt, shape[3]),
           "function": f"steel_attn_f32_bq{qt}_bk{kt}_bd{shape[3]}_wm{qt//8}_wn1"}
    try:
        resp, out = run_candidate(binary, config_dir, shape, qt, kt, kernel_path,
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
        rec["runner_stderr"] = resp.get("runner_stderr", "")
        rec["wall_timing"] = summarize_ms(resp.get("wall_times", []))
        gpu_avail = bool(resp.get("gpu_time_available"))
        rec["gpu_time_available"] = gpu_avail
        rec["gpu_timing"] = summarize_ms(resp.get("gpu_times", [])) if gpu_avail else None
        if not rec["comparison"]["shape_ok"] or not rec["comparison"]["within_tolerance"]:
            rec["status"] = "numeric_failed"
        else:
            rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return rec


def run_case(args, binary, config_dir, seq, dim, rng, case_seed, config_order, kernel_path):
    b, h, n, d = args.batch, args.heads, seq, dim
    shape = (b, h, n, d)
    scale = math.pow(float(dim), -0.5)
    q_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    k_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    v_d = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    ref = reference_attention(q_d.astype(np.float64),
                              k_d.astype(np.float64),
                              v_d.astype(np.float64), scale)
    case = {
        "seq": seq, "dim": dim, "dtype": DTYPE, "heads": h, "batch": b,
        "scale": scale, "seed": args.seed, "case_seed": case_seed,
        "config_order": [list(t) for t in config_order],
        "inputs_sha256": {
            "q": sha256_hex(q_d), "k": sha256_hex(k_d), "v": sha256_hex(v_d),
        },
        "results": {},
    }
    q_bin = os.path.join(config_dir, "q.bin")
    k_bin = os.path.join(config_dir, "k.bin")
    v_bin = os.path.join(config_dir, "v.bin")
    q_d.tofile(q_bin)
    k_d.tofile(k_bin)
    v_d.tofile(v_bin)
    for qt, kt, dcfg, wm, wn in config_order:
        if dcfg != dim:
            continue
        name = f"qt{qt}_kt{kt}"
        case["results"][name] = run_config(args, binary, config_dir, shape, qt, kt,
                                           kernel_path, ref, q_bin, k_bin, v_bin)
    return case


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

    configs = valid_configs(args.qts, args.kts, args.dims)
    if not configs:
        sys.exit("no valid (qt,kt,d) configs fit 32 KiB for the requested grid")

    meta = collect_meta(args)
    meta["grid"] = {
        "seqs": args.seqs, "dims": args.dims, "qts": args.qts, "kts": args.kts,
        "heads": args.heads, "batch": args.batch, "warmup": args.warmup,
        "samples": args.samples, "inner": args.inner, "seed": args.seed,
        "verify_only": args.verify_only,
        "valid_configs": [list(c) for c in configs],
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # Generate flattened vendor-derived MSL kernel source once.
    kernel_dir = os.path.join(out_dir, "generated_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    license_text = importlib.metadata.distribution("mlx").read_text("licenses/LICENSE")
    if not license_text:
        raise RuntimeError("Installed MLX license is required alongside the generated vendor source")
    with open(os.path.join(kernel_dir, "LICENSE-MLX"), "w") as fh:
        fh.write(license_text)
    kernel_path = os.path.join(kernel_dir, f"steel_pending_{uuid.uuid4().hex}.metal")
    gen_src = generate_kernel_source(MLX_STEEL_TOP, MLX_INCLUDE, configs, kernel_path)
    source_digest = hashlib.sha256(gen_src.encode("utf-8")).hexdigest()
    archived_kernel_path = os.path.join(kernel_dir, f"steel_attention_{source_digest}.metal")
    os.replace(kernel_path, archived_kernel_path)
    kernel_path = archived_kernel_path

    document = {
        "meta": meta,
        "vendor": {
            "note": "Kernel is a template-instantiated specialization of the installed "
                    "mlx steel attention template (Apple Inc., MIT), NOT an original "
                    "generated kernel.",
            "mlx_version": meta["mlx_version"],
            "steel_attention_header": MLX_STEEL_TOP,
            "generated_kernel_path": kernel_path,
            "generated_kernel_sha256": hashlib.sha256(
                gen_src.encode("utf-8")).hexdigest(),
            "inlined_header_sha256": source_hashes(MLX_STEEL_TOP, MLX_INCLUDE, configs),
        },
        "timing_caveats": [
            "All timings are in milliseconds.",
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per "
            "invocation of the FULL single-kernel pipeline. They exclude "
            "input/output CPU copies and compile/warmup. Invalid timestamps are "
            "reported unavailable (gpu_timing=null).",
            "Wall times are synchronized API latency per invocation.",
            "Configurations (qt,kt) run serially per case in a deterministic, "
            "seed-shuffled order.",
        ],
        "cases": [],
    }

    temp_root = args.temp_dir
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix="steel_")
        temp_owned = True
    else:
        os.makedirs(temp_root, exist_ok=True)
        temp_owned = False

    try:
        for seq in args.seqs:
            for dim in args.dims:
                case_seed = args.seed + (
                    seq * 1000003 + dim * 104729 + len(DTYPE) * 131071
                ) % (2**31)
                b, h, n, d = args.batch, args.heads, seq, dim
                ref_bytes = 8 * 8 * seq * seq + 3 * 8 * seq * dim
                if ref_bytes > args.mem_cap_bytes:
                    case = {"seq": seq, "dim": dim, "dtype": DTYPE,
                            "case_seed": case_seed,
                            "error": {"type": "MemoryError",
                                      "message": f"float64 reference peak ~{ref_bytes/1e9:.2f} GiB exceeds --mem-cap-bytes={args.mem_cap_bytes}"},
                            "results": {}}
                    document["cases"].append(case)
                    _save_document(args.output, document)
                    print(f"=== seq={seq} dim={dim} SKIPPED (memory cap) ===", flush=True)
                    continue

                rng = np.random.default_rng(case_seed)
                order_rng = np.random.default_rng(case_seed + 17)
                all_configs = [c for c in configs if c[2] == dim]
                order_rng.shuffle(all_configs)
                config_order = all_configs

                config_dir = os.path.join(temp_root, f"s{seq}_d{dim}")
                os.makedirs(config_dir, exist_ok=True)

                print(f"=== seq={seq} dim={dim} config_order={config_order} ===", flush=True)
                case = run_case(args, binary, config_dir, seq, dim, rng,
                                case_seed, config_order, kernel_path)
                document["cases"].append(case)
                for name, rec in case["results"].items():
                    status = rec.get("status")
                    if status == "ok":
                        med = (rec.get("gpu_timing") or {}).get("median_ms") or \
                              rec.get("wall_timing", {}).get("median_ms")
                        print(f"    {name}: gpu_med={med}ms "
                              f"pass={rec['comparison'].get('within_tolerance')}", flush=True)
                    else:
                        print(f"    {name}: {status} "
                              f"{rec.get('error', {}).get('type')}", flush=True)
                _save_document(args.output, document)
    finally:
        if temp_owned and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    _save_document(args.output, document)
    print(f"Done. Wrote {args.output}", flush=True)

    if args.verify_only:
        failures = []
        ran_any = False
        for case in document["cases"]:
            if not case.get("results"):
                continue
            ran_any = True
            for name, rec in case["results"].items():
                if rec.get("status") != "ok":
                    failures.append((case["seq"], case["dim"], name, rec.get("status")))
        if not ran_any:
            print("VERIFY FAIL: no configurations ran", flush=True)
            return 1
        if failures:
            for seq, dim, name, status in failures:
                print(f"VERIFY FAIL: seq={seq} dim={dim} {name}: {status}", flush=True)
            return 1
        print("VERIFY: all configurations passed correctness checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
