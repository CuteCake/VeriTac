#!/usr/bin/env python3
"""Adversarial CUDA numerical validation for the candidate kernels.

Separate runner using the public build_runtime() API. Applies the same
adversarial input patterns as the Metal worker to N=257, D in {192,256}, for a
set of candidate configs (default 20, 25, 26). Compares against an independent
numpy float64 stable causal-softmax reference with the strict survey tolerance
atol 1e-4 / rtol 1e-3. Saves all raw results to JSON.

Patterns:
  zeros            Q=K=V=0
  uniform_scores   Q=K constant -> uniform scores
  constant_values  V constant -> output == V
  large_tied_scores Q=K=100 -> huge tied logits (max-softmax stress)
  peaked_scores    q/k scaled by 16 -> peaked scores
  causal_impulse   one nonzero value key; earlier causal rows must stay zero
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import run_candidate as rc

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = {"atol": 1.0e-4, "rtol": 1.0e-3}
PATTERNS = ["zeros", "uniform_scores", "constant_values", "large_tied_scores",
            "peaked_scores", "causal_impulse"]


def gen_pattern(name, rng, seq, dim):
    b, h = 1, 8
    shape = (b, h, seq, dim)
    if name == "zeros":
        q = np.zeros(shape, np.float32)
        k = np.zeros(shape, np.float32)
        v = np.zeros(shape, np.float32)
    elif name == "uniform_scores":
        q = np.full(shape, 1.0, np.float32)
        k = np.full(shape, 1.0, np.float32)
        v = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    elif name == "constant_values":
        q = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
        k = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
        v = np.full(shape, 0.7, np.float32)
    elif name == "large_tied_scores":
        q = np.full(shape, 100.0, np.float32)
        k = np.full(shape, 100.0, np.float32)
        v = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    elif name == "peaked_scores":
        q = (rng.normal(0.0, 0.5, size=shape) * 16).astype(np.float32)
        k = (rng.normal(0.0, 0.5, size=shape) * 16).astype(np.float32)
        v = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
    elif name == "causal_impulse":
        q = np.zeros(shape, np.float32)
        k = np.zeros(shape, np.float32)
        v = np.zeros(shape, np.float32)
        v[:, :, min(128, seq - 1), :] = 10.0
    else:
        raise ValueError(name)
    return q, k, v


def reference_attention(q, k, v, scale):
    b, h, n, d = q.shape
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(upper[None, None], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    e = np.where(np.isneginf(scores), 0.0, e)
    s = e.sum(axis=-1, keepdims=True)
    attn = e / s
    return np.matmul(attn, v)


def compare(out_t, ref):
    out = out_t.detach().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(out).all())
    finite_ref = bool(np.isfinite(ref).all())
    err = np.abs(out - ref)
    max_abs = float(err.max()) if (finite and finite_ref) else float("nan")
    denom = TOL["atol"] + TOL["rtol"] * np.abs(ref)
    norm = err / denom
    norm_max = float(norm.max()) if (finite and finite_ref) else float("nan")
    rms = float(np.sqrt(np.mean(np.square(err)))) if (finite and finite_ref) else float("nan")
    passed = bool(
        finite and finite_ref
        and np.all(out <= ref + TOL["atol"] + TOL["rtol"] * np.abs(ref))
        and np.all(out >= ref - TOL["atol"] - TOL["rtol"] * np.abs(ref)))
    return {"finite_out": finite, "finite_ref": finite_ref,
            "max_abs_error": max_abs, "normalized_max_error": norm_max,
            "rms_error": rms, "within_tolerance": passed}


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cuda-adversarial.json"))
    ap.add_argument("--configs", default="20,25,26")
    ap.add_argument("--dims", default="192,256")
    ap.add_argument("--seq", type=int, default=257)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    configs = [int(x) for x in args.configs.split(",") if x.strip()]
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    torch.manual_seed(0)
    print("[adv] building runtime ...", flush=True)
    run, modules = rc.build_runtime()
    print("[adv] modules:", {k: m.__file__ for k, m in modules.items()}, flush=True)

    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": {
                "name": torch.cuda.get_device_name(0),
                "capability": "%d.%d" % torch.cuda.get_device_capability(0),
                "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            },
            "tolerance": TOL,
            "grid": {"configs": configs, "dims": dims, "seq": args.seq,
                     "seed": args.seed, "patterns": PATTERNS},
            "module_files": {k: m.__file__ for k, m in modules.items()},
        },
        "cases": [],
    }

    failures = 0
    for cid in configs:
        for dim in dims:
            scale = math.pow(float(dim), -0.5)
            cs = rc.case_seed_of(args.seed, args.seq, dim, "float32")
            rng = np.random.default_rng(cs)
            for pat in PATTERNS:
                q, k, v = gen_pattern(pat, rng, args.seq, dim)
                qt = torch.from_numpy(q).cuda().contiguous()
                kt = torch.from_numpy(k).cuda().contiguous()
                vt = torch.from_numpy(v).cuda().contiguous()
                ref = reference_attention(q.astype(np.float64),
                                          k.astype(np.float64),
                                          v.astype(np.float64), scale)
                out_phys = run.alloc_out(qt)
                prep = run.prepare(cid, qt, kt, vt, out_phys, scale, 1)
                info = prep["info"]
                rec = {
                    "config": cid, "config_name": rc.CONFIG_NAMES.get(cid),
                    "module": info.get("module"), "base_config_id": info.get("base_config_id"),
                    "seq": args.seq, "dim": dim, "pattern": pat,
                    "case_seed": cs, "supported": info.get("supported", False),
                    "validate": prep["validate"],
                    "input_hashes": {
                        "q": rc.tensor_sha256(qt), "k": rc.tensor_sha256(kt),
                        "v": rc.tensor_sha256(vt)},
                }
                if info.get("supported", False) and prep["validate"].get("ok"):
                    try:
                        run.launch(cid, qt, kt, vt, out_phys, scale)
                        torch.cuda.synchronize()
                        rec["comparison"] = compare(run.logical(out_phys), ref)
                        ok = rec["comparison"].get("within_tolerance", False)
                        if not ok:
                            failures += 1
                        print(f"  cfg{cid} N{args.seq} D{dim} {pat}: "
                              f"pass={ok} nmax={rec['comparison']['normalized_max_error']}"
                              f" maxabs={rec['comparison']['max_abs_error']:.3e}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = str(e)
                        failures += 1
                        print(f"  cfg{cid} N{args.seq} D{dim} {pat}: ERROR {e}", flush=True)
                else:
                    rec["error"] = "unsupported"
                    failures += 1
                    print(f"  cfg{cid} N{args.seq} D{dim} {pat}: UNSUPPORTED", flush=True)
                results["cases"].append(rec)
                _save(args.out, results)
    _save(args.out, results)
    print("[adv] final JSON:", args.out, flush=True)
    print(f"[adv] failures={failures}", flush=True)
    if failures > 0 or not results["cases"]:
        sys.exit(1)


def _save(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sanitize(results), f, allow_nan=False, default=str)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
