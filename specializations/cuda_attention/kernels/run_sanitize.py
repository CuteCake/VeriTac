#!/usr/bin/env python3
"""Run one candidate config under compute-sanitizer (memcheck/racecheck).

Small, single-launch (no CUDA graph) run intended to be wrapped by
`compute-sanitizer --tool memcheck|racecheck`. Uses the same public
build_runtime() API as the driver. Exits non-zero if the kernel reports an
error or numerics fail.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

import run_candidate as rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=4)
    ap.add_argument("--seq", type=int, default=100)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--k192", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    run, modules = rc.build_runtime(k192=args.k192)

    scale = math.pow(float(args.dim), -0.5)
    cs = rc.case_seed_of(1234, args.seq, args.dim, "float32")
    q, k, v = rc.make_inputs(cs, args.seq, args.dim)
    qt = torch.from_numpy(q).cuda().contiguous()
    kt = torch.from_numpy(k).cuda().contiguous()
    vt = torch.from_numpy(v).cuda().contiguous()
    ref = rc.reference_attention(q.astype(np.float64), k.astype(np.float64),
                                 v.astype(np.float64), scale)

    info = run.infos.get(args.config, {})
    print("config_info", info, flush=True)
    if not info.get("supported", False):
        print("config unsupported, skipping launch", flush=True)
        sys.exit(0)

    out_phys = run.alloc_out(qt)
    prep = run.prepare(args.config, qt, kt, vt, out_phys, scale, 1)
    print("validate", prep["validate"], flush=True)
    run.launch(args.config, qt, kt, vt, out_phys, scale)
    torch.cuda.synchronize()
    comp = rc.compare_outputs(run.logical(out_phys), ref)
    print("comparison", comp, flush=True)
    if not comp.get("within_tolerance", False):
        print("NUMERIC FAIL", flush=True)
        sys.exit(2)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
