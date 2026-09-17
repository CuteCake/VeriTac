#!/usr/bin/env python3
"""CUDA attention candidate driver.

Hardware-aware re-instantiation of the installed PyTorch mem_eff
AttentionKernel<float, cutlass::arch::Sm80, ...> for the causal prefill FP32
operator, plus mechanically generated derivatives (launch-order reversal, FP32
scratch-lifetime alias + async drains). Builds the extension modules, runs
numerical tests, then a target sweep with CUDA-event, synchronized wall, and
CUDA-graph replay timing (matching the survey harness protocol). Records
config/resources/source/version/input hashes and full samples to JSON
incrementally.

Public API: build_runtime() -> (Runner, modules_dict). Runner.infos is the
complete immutable config catalogue keyed by selectable (effective) ids; each
entry carries base_config_id for derivative modules. modules_dict maps module
names to the actual module objects (each with __file__ for binary-hash
binding). main() and run_sanitize.py both use this API.

Output ABI: the installed kernel writes output with physical layout
(B, N, H, D) (o_strideM = H*D). We allocate a (B,N,H,D) buffer for all timed
launches and view it as logical (B,H,N,D) via permute only for comparison.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "attention_kernel_forward_standalone.cu")
REV_SOURCE = os.path.join(HERE, "attention_reversed.cu")
ALIAS_SOURCE = os.path.join(HERE, "attention_alias.cu")
REVDERIV_SOURCE = os.path.join(HERE, "attention_revderiv.cu")

CUTLASS_INCLUDE = os.environ.get(
    "VERITAC_CUTLASS_INCLUDE",
    "/home/cake/.cache/veritac-attention-run/cuda_candidate/cutlass/include",
)

TOL = {"atol": 1.0e-4, "rtol": 1.0e-3}
CONFIG_NAMES = {
    0: "Q32K256-min1", 1: "Q32K128-min3", 2: "Q32K64-min6", 3: "Q64K64-min3",
    4: "Q32K128-min1", 5: "Q64K64-min1", 6: "Q64K64-min2", 7: "Q32K64-min1",
    10: "Q64K128-min1", 11: "Q64K128-min2", 12: "Q128K64-min1", 13: "Q128K64-min2",
    8: "Q32K192-min2", 9: "Q32K192-min1",
    14: "Q32K128-min3-REV", 15: "Q64K64-min3-REV",
    16: "Q32K128-min1-REV", 17: "Q64K64-min1-REV",
    20: "Q64K128-min1-ALIAS", 21: "Q64K128-min2-ALIAS",
    22: "Q32K128-min1-ALIAS", 23: "Q64K128-min1-DRAIN", 24: "Q32K128-min1-DRAIN",
    25: "Q32K128-min1-REVDRAIN", 26: "Q64K128-min1-REVALIAS",
}
# selectable (effective) id -> (module name, base id within that module).
# Effective ids are stable/immutable for the catalogue; base_config_id records
# the id as used inside a derivative module (retained separately).
MODULE_CFG = {}
for _c in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]:
    MODULE_CFG[_c] = ("normal", _c)
MODULE_CFG.update({
    14: ("reverse", 1), 15: ("reverse", 3), 16: ("reverse", 4), 17: ("reverse", 5),
    20: ("alias", 20), 21: ("alias", 21), 22: ("alias", 22),
    23: ("alias", 23), 24: ("alias", 24),
    25: ("revderiv", 25), 26: ("revderiv", 26),
})
# Complete selectable catalogue (K192 optional ids 8/9 excluded by default).
ALL_CONFIGS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17,
               20, 21, 22, 23, 24, 25, 26]


def tensor_sha256(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def case_seed_of(base_seed, seq, dim, dtype):
    return base_seed + (seq * 1000003 + dim * 104729 + len(dtype) * 131071) % (2**31)


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


def compare_outputs(out_t, ref):
    out = out_t.detach().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(out).all())
    finite_ref = bool(np.isfinite(ref).all())
    err = np.abs(out - ref)
    max_abs = float(err.max()) if (finite and finite_ref) else float("nan")
    denom = TOL["atol"] + TOL["rtol"] * np.abs(ref)
    normalized = err / denom
    normalized_max = (
        float(normalized.max()) if (finite and finite_ref) else float("nan"))
    rms = float(np.sqrt(np.mean(np.square(err)))) if (finite and finite_ref) else float("nan")
    passed = bool(
        finite and finite_ref
        and np.all(out <= ref + TOL["atol"] + TOL["rtol"] * np.abs(ref))
        and np.all(out >= ref - TOL["atol"] - TOL["rtol"] * np.abs(ref)))
    return {
        "finite_out": finite,
        "finite_ref": finite_ref,
        "max_abs_error": max_abs,
        "normalized_max_error": normalized_max,
        "rms_error": rms,
        "within_tolerance": passed,
    }


def _cflags(k192=False):
    flags = ["-O3", "-Xptxas", "-v", "-lineinfo"]
    if k192:
        flags.append("-DVERITAC_ENABLE_K192")
    return flags


def build_extension(k192=False):
    if not os.path.exists(CUTLASS_INCLUDE):
        raise RuntimeError("CUTLASS include dir not found: " + CUTLASS_INCLUDE)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    return cpp_extension.load(
        name="veritac_attn_candidate",
        sources=[SOURCE],
        extra_include_paths=[CUTLASS_INCLUDE, HERE],
        extra_cuda_cflags=_cflags(k192),
        build_directory=os.path.join(HERE, "build"),
        verbose=False,
    )


def build_rev_extension():
    if not os.path.exists(CUTLASS_INCLUDE):
        raise RuntimeError("CUTLASS include dir not found: " + CUTLASS_INCLUDE)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    return cpp_extension.load(
        name="veritac_attn_candidate_rev",
        sources=[REV_SOURCE],
        extra_include_paths=[CUTLASS_INCLUDE, HERE],
        extra_cuda_cflags=_cflags(),
        build_directory=os.path.join(HERE, "build"),
        verbose=False,
    )


def build_alias_extension():
    if not os.path.exists(CUTLASS_INCLUDE):
        raise RuntimeError("CUTLASS include dir not found: " + CUTLASS_INCLUDE)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    return cpp_extension.load(
        name="veritac_attn_candidate_alias",
        sources=[ALIAS_SOURCE],
        extra_include_paths=[CUTLASS_INCLUDE, HERE],
        extra_cuda_cflags=_cflags(),
        build_directory=os.path.join(HERE, "build"),
        verbose=False,
    )


def build_revderiv_extension():
    if not os.path.exists(CUTLASS_INCLUDE):
        raise RuntimeError("CUTLASS include dir not found: " + CUTLASS_INCLUDE)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    return cpp_extension.load(
        name="veritac_attn_candidate_revderiv",
        sources=[REVDERIV_SOURCE],
        extra_include_paths=[CUTLASS_INCLUDE, HERE],
        extra_cuda_cflags=_cflags(),
        build_directory=os.path.join(HERE, "build"),
        verbose=False,
    )


def build_runtime(k192=False):
    """Build the default module set (normal, reverse, alias, revderiv) and
    return (Runner, modules_dict). modules_dict maps module names to the actual
    module objects (each has __file__ for binary-hash binding)."""
    modules = {
        "normal": build_extension(k192=k192),
        "reverse": build_rev_extension(),
        "alias": build_alias_extension(),
        "revderiv": build_revderiv_extension(),
    }
    return Runner(modules), modules


def make_inputs(case_seed, seq, dim, heads=8, batch=1):
    rng = np.random.default_rng(case_seed)
    shape = (batch, heads, seq, dim)
    q = rng.normal(0.0, 0.5, size=shape).astype(np.float32).copy()
    k = rng.normal(0.0, 0.5, size=shape).astype(np.float32).copy()
    v = rng.normal(0.0, 0.5, size=shape).astype(np.float32).copy()
    return q, k, v


class Runner:
    """Dispatches by selectable (effective) config id across modules, and
    exposes a complete immutable catalogue in self.infos keyed by effective id
    (each entry has base_config_id + module + normalized 'id')."""

    def __init__(self, modules):
        self.modules = modules
        self.mod = modules["normal"]
        self.infos = {}
        self._prepared = {}
        for cid in ALL_CONFIGS:
            try:
                self.infos[cid] = self._query(cid)
            except Exception as e:  # noqa: BLE001
                self.infos[cid] = {"id": cid, "error": str(e), "supported": False,
                                   "module": MODULE_CFG[cid][0],
                                   "base_config_id": MODULE_CFG[cid][1]}

    def _module(self, cid):
        mname, base = MODULE_CFG[cid]
        return self.modules[mname], base

    def _query(self, cid):
        mname, base = MODULE_CFG[cid]
        mod = self.modules[mname]
        if mname == "reverse":
            info = mod.config_info_rev(base)
        elif mname == "alias":
            info = mod.config_info_alias(base)
        elif mname == "revderiv":
            info = mod.config_info_revderiv(base)
        else:
            info = mod.config_info(base)
        info["id"] = cid
        info["base_config_id"] = base
        info["module"] = mname
        return info

    def alloc_out(self, q):
        return torch.empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
                           dtype=q.dtype, device=q.device)

    def logical(self, out_phys):
        return out_phys.permute(0, 2, 1, 3)

    def prepare(self, cid, q, k, v, out_phys, scale, causal):
        key = (cid, q.shape[2], q.shape[1], q.shape[3])
        if key in self._prepared:
            return self._prepared[key]
        mname, base = MODULE_CFG[cid]
        mod = self.modules[mname]
        ptr_args = (int(base), int(q.data_ptr()), int(k.data_ptr()),
                    int(v.data_ptr()), int(out_phys.data_ptr()),
                    int(q.shape[2]), int(q.shape[1]), int(q.shape[3]),
                    int(causal), float(scale))
        if mname == "reverse":
            info = mod.config_info_rev(base)
            res = mod.validate_rev(*ptr_args)
        elif mname == "alias":
            info = mod.config_info_alias(base)
            res = mod.validate_alias(*ptr_args)
        elif mname == "revderiv":
            info = mod.config_info_revderiv(base)
            res = mod.validate_revderiv(*ptr_args)
        else:
            info = mod.config_info(base)
            res = mod.validate(*ptr_args)
        info["id"] = cid
        info["base_config_id"] = base
        info["module"] = mname
        self._prepared[key] = {"info": info, "validate": res}
        return self._prepared[key]

    def launch(self, cid, q, k, v, out_phys, scale, causal=1):
        s = torch.cuda.current_stream().cuda_stream
        mname, base = MODULE_CFG[cid]
        mod = self.modules[mname]
        args = (int(base), int(q.data_ptr()), int(k.data_ptr()),
                int(v.data_ptr()), int(out_phys.data_ptr()),
                int(q.shape[2]), int(q.shape[1]), int(q.shape[3]),
                int(causal), float(scale), int(s))
        if mname == "reverse":
            mod.launch_attn_rev(*args)
        elif mname == "alias":
            mod.launch_attn_alias(*args)
        elif mname == "revderiv":
            mod.launch_attn_revderiv(*args)
        else:
            mod.launch_attn(*args)

    def run_once(self, cid, q, k, v, scale, causal=1):
        out_phys = self.alloc_out(q)
        self.launch(cid, q, k, v, out_phys, scale, causal)
        return out_phys


def event_timing(run, cid, q, k, v, out_phys, scale, warmup, samples, inner,
                 causal=1):
    run.launch(cid, q, k, v, out_phys, scale, causal)
    torch.cuda.synchronize()
    for _ in range(warmup):
        run.launch(cid, q, k, v, out_phys, scale, causal)
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            run.launch(cid, q, k, v, out_phys, scale, causal)
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)
    return stat(samples_ms)


def wall_timing(run, cid, q, k, v, out_phys, scale, warmup, samples, causal=1):
    run.launch(cid, q, k, v, out_phys, scale, causal)
    torch.cuda.synchronize()
    for _ in range(warmup):
        run.launch(cid, q, k, v, out_phys, scale, causal)
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(samples):
        t0 = time.perf_counter()
        run.launch(cid, q, k, v, out_phys, scale, causal)
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1e3)
    return stat(samples_ms)


def graph_timing(run, cid, q, k, v, out_phys, scale, inner, warmup, samples,
                 causal=1):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run.launch(cid, q, k, v, out_phys, scale, causal)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # Match the vendor graph's distinct outputs per captured call. Reusing one
    # output can confer an artificial cache advantage; retain the caller's
    # buffer for the final call so its graph result is numerically checked.
    graph_outputs = [run.alloc_out(q) for _ in range(inner - 1)] + [out_phys]
    with torch.cuda.graph(g):
        for graph_out in graph_outputs:
            run.launch(cid, q, k, v, graph_out, scale, causal)
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)
    g.replay()
    torch.cuda.synchronize()
    return stat(samples_ms)


def stat(samples_ms):
    arr = np.array(samples_ms)
    return {
        "median_ms": float(np.median(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "samples_ms": [float(x) for x in arr],
    }


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


def save(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sanitize(results), f, allow_nan=False, default=str)
    os.replace(tmp, path)


def parse_csv(s, default):
    s = s.strip()
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cuda-candidate.json"))
    ap.add_argument("--mode", choices=["build", "numeric", "sweep", "all"], default="all")
    ap.add_argument("--configs", default=",".join(str(c) for c in ALL_CONFIGS))
    ap.add_argument("--seqs", default="1024,2048")
    ap.add_argument("--dims", default="192,256")
    ap.add_argument("--numeric-seqs", default="17,100,257,1536")
    ap.add_argument("--numeric-dims", default="192,256")
    ap.add_argument("--k192", action="store_true",
                    help="build with VERITAC_ENABLE_K192 and include D192-only configs 8/9")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--optin-mb", type=int, default=0,
                    help="unused; prep reads live device attrs")
    args = ap.parse_args()

    configs = [int(x) for x in args.configs.split(",") if x.strip()]
    if args.k192:
        configs = [c for c in configs if c not in (8, 9)] + [8, 9]
    seqs = parse_csv(args.seqs, [1024, 2048])
    dims = parse_csv(args.dims, [192, 256])
    numeric_seqs = parse_csv(args.numeric_seqs, [17, 100, 257, 1536])
    numeric_dims = parse_csv(args.numeric_dims, [192, 256])

    torch.manual_seed(0)
    print(f"[driver] building runtime (mode={args.mode}, k192={args.k192}) ...", flush=True)
    run, modules = build_runtime(k192=args.k192)
    print("[driver] modules:", {k: m.__file__ for k, m in modules.items()}, flush=True)
    print("[driver] config info (effective ids):", flush=True)
    for cid in configs:
        print("   ", cid, CONFIG_NAMES.get(cid, "?"),
              json.dumps(run.infos.get(cid, {}), default=str), flush=True)

    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": {
                "name": torch.cuda.get_device_name(0),
                "capability": "%d.%d" % torch.cuda.get_device_capability(0),
                "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_git_version": torch.version.git_version,
            "module_files": {k: m.__file__ for k, m in modules.items()},
            "provenance": {
                "kernel": "PyTorch installed mem_eff AttentionKernel<float,Sm80,true,Q,K,MD,false,false>",
                "arithmetic": "unmodified vendor OpMultiplyAddFastF32 (CUTLASS 3xTF32 emulation); "
                              "NOT single TF32, NOT FP16, NOT IEEE scalar FP32",
                "attribution": "Meta Platforms BSD-3-Clause PyTorch mem_eff_attention; "
                               "NVIDIA CUTLASS BSD-3-Clause",
                "cutlass_pin": "e05f953a5b3d38adc240df2ff928e0421c2abba3",
                "pytorch_pin": "08187d9e0fba026dc8217405802ab5381dc88d90",
                "derivatives": {
                    "reverse": "ids14-17: query_start reversed (gridDim.x-1-blockIdx.x)*Q "
                               "in PyTorchMemEffAttentionRev",
                    "alias": "ids20-22: union mm1/epilogue + cp_async drains in "
                             "PyTorchMemEffAttentionAlias; ids23-24: drain-only control "
                             "PyTorchMemEffAttentionDrain; ids25-26: +reversal "
                             "PyTorchMemEffAttentionRevDrain/RevAlias; hashes in "
                             "docs/cuda_attention_specialization.md",
                },
            },
            "output_layout": "physical (B,N,H,D) with o_strideM=H*D; logical view (B,H,N,D) for comparison",
            "tolerance": TOL,
            "grid": {
                "configs": configs,
                "config_names": {c: CONFIG_NAMES.get(c) for c in configs},
                "config_modules": {c: MODULE_CFG[c][0] for c in configs},
                "config_base_ids": {c: MODULE_CFG[c][1] for c in configs},
                "seqs": seqs,
                "dims": dims,
                "numeric_seqs": numeric_seqs,
                "numeric_dims": numeric_dims,
                "seed": args.seed,
                "warmup": args.warmup,
                "samples": args.samples,
                "inner": args.inner,
            },
        },
        "configs": {c: run.infos.get(c) for c in configs},
        "cases": [],
    }

    failures = 0
    tested_valid = 0

    if args.mode in ("numeric", "all"):
        print("[numeric] small-N correctness tests ...", flush=True)
        for cid in configs:
            for seq in numeric_seqs:
                for dim in numeric_dims:
                    scale = math.pow(float(dim), -0.5)
                    cs = case_seed_of(args.seed, seq, dim, "float32")
                    q, k, v = make_inputs(cs, seq, dim)
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
                        "config": cid, "config_name": CONFIG_NAMES.get(cid),
                        "seq": seq, "dim": dim, "scale": scale, "case_seed": cs,
                        "module": info.get("module"), "base_config_id": info.get("base_config_id"),
                        "supported": info.get("supported", False),
                        "validate": prep["validate"],
                        "input_hashes": {
                            "q": tensor_sha256(qt), "k": tensor_sha256(kt),
                            "v": tensor_sha256(vt)},
                    }
                    if info.get("supported", False) and prep["validate"].get("ok"):
                        try:
                            run.launch(cid, qt, kt, vt, out_phys, scale)
                            torch.cuda.synchronize()
                            rec["comparison"] = compare_outputs(run.logical(out_phys), ref)
                            ok = rec["comparison"].get("within_tolerance", False)
                            tested_valid += 1
                            if not ok:
                                failures += 1
                            print(f"  cfg{cid}({CONFIG_NAMES.get(cid)}) N{seq} D{dim}: "
                                  f"pass={ok} nmax={rec['comparison']['normalized_max_error']:.3e}",
                                  flush=True)
                        except Exception as e:  # noqa: BLE001
                            rec["error"] = str(e)
                            failures += 1
                            print(f"  cfg{cid} N{seq} D{dim}: ERROR {e}", flush=True)
                    else:
                        rec["error"] = "unsupported or failed validation"
                        print(f"  cfg{cid} N{seq} D{dim}: UNSUPPORTED "
                              f"({rec.get('validate')})", flush=True)
                    results["cases"].append(rec)
                    save(args.out, results)
        print("[numeric] done.", flush=True)

    if args.mode in ("sweep", "all"):
        print("[sweep] target sweep ...", flush=True)
        for cid in configs:
            for seq in seqs:
                for dim in dims:
                    scale = math.pow(float(dim), -0.5)
                    cs = case_seed_of(args.seed, seq, dim, "float32")
                    q, k, v = make_inputs(cs, seq, dim)
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
                        "config": cid, "config_name": CONFIG_NAMES.get(cid),
                        "seq": seq, "dim": dim, "scale": scale, "case_seed": cs,
                        "module": info.get("module"), "base_config_id": info.get("base_config_id"),
                        "supported": info.get("supported", False),
                        "validate": prep["validate"],
                        "input_hashes": {
                            "q": tensor_sha256(qt), "k": tensor_sha256(kt),
                            "v": tensor_sha256(vt)},
                    }
                    if info.get("supported", False) and prep["validate"].get("ok"):
                        try:
                            run.launch(cid, qt, kt, vt, out_phys, scale)
                            torch.cuda.synchronize()
                            tested_valid += 1
                            rec["comparison"] = compare_outputs(run.logical(out_phys), ref)
                            rec["cuda_event_timing"] = event_timing(
                                run, cid, qt, kt, vt, out_phys, scale,
                                args.warmup, args.samples, args.inner)
                            rec["wall_timing"] = wall_timing(
                                run, cid, qt, kt, vt, out_phys, scale,
                                args.warmup, args.samples)
                            rec["graph_timing"] = graph_timing(
                                run, cid, qt, kt, vt, out_phys, scale,
                                args.inner, args.warmup, args.samples)
                            rec["graph_comparison"] = compare_outputs(
                                run.logical(out_phys), ref)
                            ok = rec["comparison"].get("within_tolerance", False)
                            gok = rec["graph_comparison"].get("within_tolerance", False)
                            if not ok or not gok:
                                failures += 1
                            print(f"  cfg{cid}({CONFIG_NAMES.get(cid)}) N{seq} D{dim}: "
                                  f"pass={ok} graph_pass={gok} "
                                  f"evt={rec['cuda_event_timing']['median_ms']:.4f}ms "
                                  f"wall={rec['wall_timing']['median_ms']:.4f}ms "
                                  f"graph={rec['graph_timing']['median_ms']:.4f}ms",
                                  flush=True)
                        except Exception as e:  # noqa: BLE001
                            rec["error"] = str(e)
                            failures += 1
                            print(f"  cfg{cid} N{seq} D{dim}: ERROR {e}", flush=True)
                    else:
                        rec["error"] = "unsupported or failed validation"
                        print(f"  cfg{cid} N{seq} D{dim}: UNSUPPORTED "
                              f"({rec.get('validate')})", flush=True)
                    results["cases"].append(rec)
                    save(args.out, results)
        print("[sweep] done.", flush=True)

    save(args.out, results)
    print("[driver] final JSON:", args.out, flush=True)
    print(f"[driver] tested_valid={tested_valid} failures={failures}", flush=True)
    if args.mode in ("numeric", "sweep", "all") and (tested_valid == 0 or failures > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
