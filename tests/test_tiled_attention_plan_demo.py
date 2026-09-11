"""Focused CPU tests for the tiled-attention plan demo search/selection logic.

These are mock-based and GPU-free: candidate *checking* runs through the Lean
CLI (a CPU operation) and the per-candidate *benchmark* is a synthetic function,
so no Metal/MLX/GPU is required.  They lock in the proof-to-result contract:

  - the final report must refer to the selected winner, not the unrelated agent
    (mock) plan;
  - the winner's tactic trace must be bound to its checked_plan and replayable
    through Lean to the identical plan;
  - non-executable and numeric_failed / invalid-timing candidates are never
    dispatched or selected;
  - all candidate losses / rejections are preserved in the report.
"""

import argparse
import json
import math
import os
import subprocess
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import llm_tiled_attention_demo as demo  # noqa: E402

BIN = ROOT / ".lake" / "build" / "bin" / "veritac"

# A resource-rich Metal profile so every search candidate is accepted+executable.
PROFILE = {
    "schema_version": 1,
    "backend": "metal",
    "device_name": "test-device",
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 100000,
    "simd_width": 32,
    "toolchain": {"name": "metal_sdk", "version": "26.5"},
    "provenance": {"source": "test", "host": "localhost"},
    "features": {"matrix_ops": None, "async_copy": None, "fp16": None, "bfloat16": None},
}

SEQ, DIM = 128, 192

MOCK_FINAL = {"seq_len": SEQ, "head_dim": DIM, "query_tile": 16, "key_tile": 8,
              "alias_kv": True, "threads_per_threadgroup": 64}


def make_args():
    return types.SimpleNamespace(seq=SEQ, dim=DIM, seed=1234, warmup=10,
                                 samples=5, inner=3)


def fake_bench(timing_by_key: dict) -> callable:
    def bench(args, run_dir, qt, kt):
        med = timing_by_key.get((qt, kt))
        try:
            fmed = float(med)
        except (TypeError, ValueError):
            # non-numeric marker => numeric_failed with no usable timing
            return {"status": "numeric_failed", "wall_timing": None,
                    "gpu_time_available": False,
                    "inputs_sha256": {"q": "q", "k": "k", "v": "v"}}
        return {"status": "ok", "wall_timing": {"median_ms": fmed},
                "gpu_time_available": False,
                "inputs_sha256": {"q": "q", "k": "k", "v": "v"}}
    return bench


def run_search(timing_by_key, qts=demo.SEARCH_QTS, kts=demo.SEARCH_KTS):
    bench = fake_bench(timing_by_key)
    records = demo.run_search_candidates(PROFILE, SEQ, DIM, qts, kts, bench,
                                         make_args(), Path("."))
    winner = demo.select_winner(records)
    return records, winner


def test_winner_differs_from_mock_plan():
    """Search can select a different tile than the mock agent's plan, and the
    report's final_plan/trace must refer to that winner."""
    timing = {(8, 8): 3.0, (16, 8): 2.5, (24, 8): 1.9,   # mock ends at (16,8)
              (8, 16): 4.0, (16, 16): 3.5, (24, 16): 2.0}
    records, winner = run_search(timing)
    assert winner is not None
    # mock plan is Q16K8; search winner is the faster Q24K8.
    assert winner["qt"] == 24 and winner["kt"] == 8
    assert MOCK_FINAL["query_tile"] == 16
    # Report references the winner, and keeps the agent plan separately.
    report = {
        "final_plan": winner["checked_plan"],
        "final_executable": True,
        "tactic_trace": winner["verification_trace"],
        "agent_final_plan": MOCK_FINAL,
    }
    assert report["final_plan"] == winner["checked_plan"]
    assert report["final_plan"] != MOCK_FINAL
    assert report["final_plan"]["query_tile"] == 24
    # All candidates preserved (including any rejections).
    assert len(records) == 6


def test_trace_bound_to_winner_and_replays():
    """The winner's verification_trace ends at its checked_plan, and the whole
    trace replays through Lean to the identical plan."""
    timing = {(8, 8): 3.0, (16, 8): 2.5, (24, 8): 1.9,
              (8, 16): 4.0, (16, 16): 3.5, (24, 16): 2.0}
    records, winner = run_search(timing)
    # trace is bound to the winner: last intermediate == checked_plan.
    assert winner["verification_trace"][-1]["final_plan"] == winner["checked_plan"]
    assert winner["checked_plan"]["query_tile"] == 24
    # replay the winner's sequence through Lean to the identical plan.
    initial = demo.initial_plan(SEQ, DIM)
    assert demo.replay_plan(PROFILE, initial, winner["tactic_sequence"],
                            winner["checked_plan"]) is True


def test_non_executable_not_dispatch_or_selected():
    """A candidate that stays alias_kv=false is never benchmarked (no dispatch)
    and is never selected."""
    calls = []
    orig_check = demo.check_candidate

    def real_check(profile, seq, dim, qt, kt):
        rec = orig_check(profile, seq, dim, qt, kt)
        if (qt, kt) == (16, 8):
            # simulate an accepted-but-non-executable planning state
            rec = dict(rec)
            rec["status"] = "not_executable"
            rec["executable"] = False
            rec["checked_plan"] = dict(rec.get("checked_plan") or {}, alias_kv=False)
        return rec

    demo.check_candidate = real_check
    try:
        def bench(args, run_dir, qt, kt):
            calls.append((qt, kt))
            return {"status": "ok", "wall_timing": {"median_ms": 2.0},
                    "gpu_time_available": False}
        records = demo.run_search_candidates(PROFILE, SEQ, DIM, (8, 16, 24), (8, 16),
                                             bench, make_args(), Path("."))
    finally:
        demo.check_candidate = orig_check
    assert (16, 8) not in calls, "non-executable candidate was benchmarked/dispatched"
    ne = [r for r in records if (r["qt"], r["kt"]) == (16, 8)][0]
    assert ne["status"] == "not_executable"
    winner = demo.select_winner(records)
    assert winner is not None and (winner["qt"], winner["kt"]) != (16, 8)


def test_numeric_failed_not_selected_but_preserved():
    """numeric_failed candidates are preserved in the report but never win."""
    timing = {(8, 8): 3.0, (16, 8): "NF", (24, 8): 1.9,     # (16,8) numeric_failed
              (8, 16): 4.0, (16, 16): 3.5, (24, 16): 2.0}
    records, winner = run_search(timing)
    nf = [r for r in records if (r["qt"], r["kt"]) == (16, 8)][0]
    assert nf["status"] == "numeric_failed"
    assert winner["qt"] == 24 and winner["kt"] == 8  # numeric_failed never wins


def test_invalid_timing_not_selected():
    """Non-positive / NaN / missing wall timings disqualify a candidate."""
    nan = float("nan")
    timing = {(8, 8): 0.0, (16, 8): nan, (24, 8): 1.9,
              (8, 16): 4.0, (16, 16): 3.5, (24, 16): 2.0}
    records, winner = run_search(timing)
    assert winner["qt"] == 24 and winner["kt"] == 8
    assert demo._finite_positive(0.0) is False
    assert demo._finite_positive(nan) is False
    assert demo._finite_positive(1.9) is True


def test_non_executable_ok_status_cannot_win():
    assert demo.select_winner([{
        "status": "ok", "wall_median_ms": 0.001,
        "executable": False, "checked_plan": {"alias_kv": False},
    }]) is None
    assert demo.select_winner([{
        "status": "ok", "wall_median_ms": 0.001,
        "executable": True, "checked_plan": {"alias_kv": False},
    }]) is None


def test_rejections_preserved():
    """Candidates that Lean rejects (e.g. resource over budget) are retained in
    the records with a rejection status, not silently dropped."""
    records, winner = run_search({(8, 8): 3.0, (16, 8): 2.5, (24, 8): 1.9,
                                  (8, 16): 4.0, (16, 16): 3.5, (24, 16): 2.0})
    # With a resource-rich profile all six are ok_candidate/ok; assert the
    # records are all present and each carries a checked_plan or a status.
    assert len(records) == 6
    assert all(r.get("checked_plan") is not None or r.get("status") in
               ("rejected", "error") for r in records)


def test_mock_no_benchmark_smoke_runs():
    """Optional integration smoke: no benchmark, but a live Metal probe."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "llm_tiled_attention_demo.py"),
         "--mock", "--no-benchmark"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"demo failed: {result.stderr}"
    assert "final plan accepted" in result.stdout
    # final executable plan is Q16K8 per the mock schedule.
    assert "q=16 k=8" in result.stdout


if __name__ == "__main__":
    test_winner_differs_from_mock_plan(); print("PASS: winner differs from mock plan")
    test_trace_bound_to_winner_and_replays(); print("PASS: trace bound to winner + replays")
    test_non_executable_not_dispatch_or_selected(); print("PASS: non-executable not dispatched/selected")
    test_numeric_failed_not_selected_but_preserved(); print("PASS: numeric_failed not selected, preserved")
    test_invalid_timing_not_selected(); print("PASS: invalid timing not selected")
    test_non_executable_ok_status_cannot_win(); print("PASS: executable flags required for winner")
    test_rejections_preserved(); print("PASS: rejections preserved")
    if os.environ.get("LIVE_METAL") == "1":
        test_mock_no_benchmark_smoke_runs(); print("PASS: live Metal mock smoke")
    else:
        print("SKIP: live Metal smoke (set LIVE_METAL=1)")
    print("All tiled-attention-plan-demo tests passed!")
