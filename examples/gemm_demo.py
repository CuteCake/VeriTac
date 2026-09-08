"""
VeriTac GEMM demo — end-to-end CPU GEMM optimization.

Pipeline:
  1. Build a matmul loop nest (Lean conventions: flat 1000-stride layout).
  2. The search agent picks a verified schedule (guard-checked by the Lean CLI:
     tile/fuse/reorder/parallel/vectorize all pass Lean's soundness gates).
  3. Lower to C, emitting OpenMP pragmas *only* where a write-dependence check
     (mirroring the proven `indexLocalP_flat_leading` criterion) says the loop
     is independent; reduction loops get a comment instead.
  4. Compile with an OpenMP-capable compiler, run, check results against numpy,
     and report timings.

Requires: `lake build veritac` (Lean CLI), numpy, an OpenMP-capable compiler
(gcc-15/14/13 or clang with libomp; macOS clang without OpenMP falls back to
a serial compile where pragmas are inert no-ops — still correct).

Usage:
    .venv/bin/python examples/gemm_demo.py [dim]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Search.interface import make_matmul_stmt, call_lean
from Search.agent import enumerate_schedules
from CodeGen.runner import benchmark_matmul


def top_schedule(max_len: int = 3, top_k: int = 3):
    """Ask the search agent for the best schedule on a 256^3 matmul."""
    print("== Step 1: search for a schedule (Lean-verified) ==")
    schedules = enumerate_schedules(max_length=max_len)
    best = schedules[0]
    tactics, score = best
    print(f"top schedule (score {score:.0f}):")
    for t in tactics:
        print(f"   {t['kind']} {t['vars']}{t['int_params'] or ''}")
    return tactics


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    dim = args.dim
    stmt = make_matmul_stmt(dim, dim, dim)

    print("== Step 2: baseline (no tactics) ==")
    b0 = benchmark_matmul(stmt, dim, dim, dim, num_runs=args.runs)
    print(f"  baseline      C time: {b0['c_time_ms']:8.3f} ms   correct: {b0['correct']}")

    tactics = top_schedule()

    print("== Step 3: apply via Lean CLI ==")
    r = call_lean(stmt, tactics)
    if "stmt" not in r:
        raise SystemExit(f"schedule rejected: {r}")
    print(f"  applied {r['applied']} tactics")

    print("== Step 4: lower to C + benchmark ==")
    b1 = benchmark_matmul(r["stmt"], dim, dim, dim, num_runs=args.runs)
    print(f"  optimized    C time: {b1['c_time_ms']:8.3f} ms   correct: {b1['correct']}")
    if b1["c_time_ms"] and b0["c_time_ms"]:
        print(f"  speedup (C vs C): {b0['c_time_ms'] / b1['c_time_ms']:.2f}x")
    print("  numpy reference (OpenBLAS, multithreaded): "
          f"{b1['numpy_time_ms']:.3f} ms/eval")


if __name__ == "__main__":
    main()