"""End-to-end tests for the `check_attention_plan` CLI mode.

These invoke the built `veritac` binary and lock in the Lean-checked
`PlanLegal` boundary for the tiled FP32 Metal attention resource plan: SIMD-32
thread accounting, measured threadgroup-memory byte layouts, K/V aliasing
executability, and per-row causal partition legality.  They also guard that the
legacy `check_attention_launch` / `check_attention_tactics` and stmt modes are
unchanged.
"""

import json
import subprocess
import time
from pathlib import Path

BIN = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"

TARGET = {
    "schema_version": 1,
    "backend": "metal",
    "device_name": "test-device",
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 32768,
    "simd_width": 32,
    "toolchain": {"name": "metal_sdk", "version": "26.5"},
    "provenance": {"source": "test", "host": "localhost"},
    "features": {"matrix_ops": None, "async_copy": None, "fp16": None, "bfloat16": None},
}


def plan(**overrides) -> dict:
    base = {
        "seq_len": 4,
        "head_dim": 256,
        "query_tile": 8,
        "key_tile": 8,
        "alias_kv": True,
    }
    base.update(overrides)
    return base


def run_cli(payload: dict) -> dict:
    result = subprocess.run(
        [str(BIN), "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def check_plan(target: dict, initial: dict, tactics: list) -> dict:
    return run_cli({
        "mode": "check_attention_plan",
        "schema_version": 1,
        "target": target,
        "initial_plan": initial,
        "tactics": tactics,
    })


def target_with(**overrides) -> dict:
    tgt = dict(TARGET)
    tgt.update(overrides)
    return tgt


# Measured FP32 layouts (see Plan.lean):
#   Q = query_tile*(head_dim+4)*4 ; K = (key_tile+4)*head_dim*4 ;
#   V = key_tile*(head_dim+4)*4 ; shared = Q + (alias ? max(K,V) : K+V)
def q_b(q, hd): return q * (hd + 4) * 4
def k_b(k, hd): return (k + 4) * hd * 4
def v_b(k, hd): return k * (hd + 4) * 4
def shared_b(q, k, hd, alias):
    return q_b(q, hd) + (max(k_b(k, hd), v_b(k, hd)) if alias else k_b(k, hd) + v_b(k, hd))


def test_default_accepted():
    out = check_plan(TARGET, plan(), [])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["applied"] == 0
    assert out["executable"] is True
    assert out["threads_per_threadgroup"] == 32      # (8/8)*32
    assert out["shared_memory_bytes"] == shared_b(8, 8, 256, True) == 20608
    assert out["final_plan"]["seq_len"] == 4
    assert out["final_plan"]["head_dim"] == 256
    assert out["final_plan"]["query_tile"] == 8
    assert out["final_plan"]["key_tile"] == 8
    assert out["final_plan"]["alias_kv"] is True
    assert out["diagnostic"] == "accepted"
    assert out["schema_version"] == 1


def test_memory_exact_boundary_accepted():
    """shared for qt8/kt8/hd256/alias is 20608; an exact budget is accepted."""
    tgt = target_with(max_threadgroup_memory_bytes=20608)
    out = check_plan(tgt, plan(), [])
    assert out["accepted"] is True, f"exact budget should fit: {out}"
    assert out["shared_memory_bytes"] == 20608


def test_memory_one_less_rejected():
    """one byte under the exact budget is rejected."""
    tgt = target_with(max_threadgroup_memory_bytes=20607)
    out = check_plan(tgt, plan(), [])
    assert out["accepted"] is False
    assert "shared" in out["diagnostic"]


def test_key_tile_16_memory_boundary():
    """qt8/kt16/hd256/alias: shared = Q + max(K,V) where K=(20)*256*4=20480,
    so shared = 8320 + 20480 = 28800. Exact budget accepted."""
    out = check_plan(TARGET, plan(key_tile=16), [])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["shared_memory_bytes"] == shared_b(8, 16, 256, True) == 28800


def test_invalid_query_tile_rejected():
    """query tile 12 is not SIMD-divisible (not in 8/16/24/32)."""
    out = check_plan(TARGET, plan(query_tile=12), [])
    assert out["accepted"] is False
    assert "query_tile" in out["diagnostic"]


def test_invalid_simd_width_rejected():
    tgt = target_with(simd_width=64)
    out = check_plan(tgt, plan(), [])
    assert out["accepted"] is False
    assert "simd_width" in out["diagnostic"]


def test_bad_head_dim_rejected():
    out = check_plan(TARGET, plan(head_dim=128), [])
    assert out["accepted"] is False
    assert "head_dim" in out["diagnostic"]


def test_bad_key_tile_rejected():
    out = check_plan(TARGET, plan(key_tile=64), [])
    assert out["accepted"] is False
    assert "key_tile" in out["diagnostic"]


def test_alias_false_accepted_not_executable():
    """an alias_kv=false plan is resource-legal (stages Q+K+V) and is accepted
    by the checker, but it is NOT executable by the vendor shader."""
    out = check_plan(TARGET, plan(alias_kv=False), [])
    assert out["accepted"] is True, f"alias=false should be resource-legal: {out}"
    assert out["executable"] is False
    # Q8K8 D256 alias=false uses Q + K + V = 8320 + 12288 + 8320 = 28928.
    assert out["shared_memory_bytes"] == shared_b(8, 8, 256, False) == 28928


def test_alias_false_bytes_saved_on_reuse():
    """reuse_kv_storage makes the plan executable AND reduces measured bytes:
    Q8K8 D256 goes from 28928 (Q+K+V) down to 20608 (Q+max(K,V))."""
    out = check_plan(TARGET, plan(alias_kv=False), [
        {"kind": "reuse_kv_storage"},
    ])
    assert out["accepted"] is True
    assert out["applied"] == 1
    assert out["executable"] is True
    assert out["final_plan"]["alias_kv"] is True
    assert out["shared_memory_bytes"] == shared_b(8, 8, 256, True) == 20608


def test_alias_false_overbudget_initial_rejected():
    """an alias=false Q8K8 D256 plan needs 28928 bytes; a budget below that
    rejects it as an over-budget initial plan (initial still resource-legal
    requirement)."""
    tgt = target_with(max_threadgroup_memory_bytes=28927)
    out = check_plan(tgt, plan(alias_kv=False), [])
    assert out["accepted"] is False
    assert out["executable"] is False
    assert "shared" in out["diagnostic"]


def test_alias_false_fits_32k_both_dims():
    """alias=false Q8K8 fits 32 KiB for both D192 and D256 (planning state)."""
    for dim in (192, 256):
        out = check_plan(TARGET, plan(alias_kv=False, head_dim=dim), [])
        assert out["accepted"] is True, f"alias=false D{dim} should fit: {out}"


def test_reuse_then_tile16_d192_executable():
    """reuse -> set_query_tile 16 produces the executable Q16K8 D192 final plan
    used by the demo benchmark."""
    out = check_plan(TARGET, plan(alias_kv=False, head_dim=192), [
        {"kind": "reuse_kv_storage"},
        {"kind": "set_query_tile", "value": 16},
    ])
    assert out["accepted"] is True
    assert out["applied"] == 2
    assert out["executable"] is True
    fp = out["final_plan"]
    assert fp["query_tile"] == 16
    assert fp["key_tile"] == 8
    assert fp["alias_kv"] is True
    assert out["threads_per_threadgroup"] == 64   # (16/8)*32
    assert out["shared_memory_bytes"] == shared_b(16, 8, 192, True) == 21760


def test_zero_seq_len_rejected():
    out = check_plan(TARGET, plan(seq_len=0), [])
    assert out["accepted"] is False
    assert "seq_len" in out["diagnostic"]


def test_malformed_plan_rejected():
    out = run_cli({
        "mode": "check_attention_plan",
        "schema_version": 1,
        "target": TARGET,
        "initial_plan": {"head_dim": 256},
        "tactics": [],
    })
    assert "error" in out
    assert "plan" in out["error"]


def test_malformed_tactic_rejected():
    out = check_plan(TARGET, plan(), [{"kind": "bogus", "value": 8}])
    assert "error" in out
    assert "Tactic" in out["error"]


def test_missing_tactic_value_rejected():
    out = check_plan(TARGET, plan(), [{"kind": "set_query_tile"}])
    assert "error" in out
    assert "Tactic" in out["error"]


def test_successful_tactic_sequence():
    out = check_plan(TARGET, plan(), [
        {"kind": "set_query_tile", "value": 16},
        {"kind": "reuse_kv_storage"},
    ])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["applied"] == 2
    assert out["final_plan"]["query_tile"] == 16
    assert out["final_plan"]["alias_kv"] is True
    assert out["threads_per_threadgroup"] == 64   # (16/8)*32
    assert out["shared_memory_bytes"] == shared_b(16, 8, 256, True) == 28928


def test_reuse_kv_is_accepted_tactic():
    """reuse_kv_storage on an already-executable plan is an accepted tactic that
    keeps `alias_kv = true` (the vendor shader always aliases K/V)."""
    out = check_plan(TARGET, plan(), [
        {"kind": "reuse_kv_storage"},
    ])
    assert out["accepted"] is True, f"expected accepted after reuse: {out}"
    assert out["applied"] == 1
    assert out["final_plan"]["alias_kv"] is True
    assert out["executable"] is True


def test_early_failure_stops():
    """first illegal tactic stops: applied counts only accepted prefix."""
    out = check_plan(TARGET, plan(), [
        {"kind": "set_query_tile", "value": 16},
        {"kind": "set_key_tile", "value": 64},
        {"kind": "set_query_tile", "value": 8},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 1
    assert out["final_plan"]["query_tile"] == 16
    assert out["final_plan"]["key_tile"] == 8
    assert "key_tile" in out["diagnostic"]


def test_initial_illegal():
    """an illegal initial plan keeps it and applies nothing."""
    out = check_plan(TARGET, plan(head_dim=128), [
        {"kind": "set_query_tile", "value": 16},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 0
    assert out["final_plan"]["head_dim"] == 128
    assert "head_dim" in out["diagnostic"]


def test_bad_schema_rejected():
    out = run_cli({
        "mode": "check_attention_plan",
        "schema_version": 2,
        "target": TARGET,
        "initial_plan": plan(),
        "tactics": [],
    })
    assert "error" in out
    assert "schema_version" in out["error"]


def test_missing_initial_plan_rejected():
    out = run_cli({
        "mode": "check_attention_plan",
        "schema_version": 1,
        "target": TARGET,
        "tactics": [],
    })
    assert "error" in out
    assert "initial_plan" in out["error"]


def test_smoke_n1024_runtime():
    """a 1024-row plan must check quickly (per-row partition legality is a
    decidable finite enumeration) and be accepted."""
    start = time.monotonic()
    out = check_plan(TARGET, plan(seq_len=1024), [])
    elapsed = time.monotonic() - start
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["final_plan"]["seq_len"] == 1024
    assert elapsed < 5.0, f"smoke check too slow: {elapsed:.2f}s"


def test_legacy_launch_unchanged():
    out = run_cli({
        "mode": "check_attention_launch",
        "schema_version": 1,
        "target": TARGET,
        "launch": {
            "mapping": "simdgroup", "head_dim": 256, "query_tile": 4,
            "key_tile": 8, "threads_per_threadgroup": 128, "dtype_bytes": 4,
            "stage_k": True, "stage_v": True, "extra_shared_bytes": 0,
        },
    })
    assert out["accepted"] is True, f"legacy launch mode changed: {out}"


def test_legacy_tactics_unchanged():
    out = run_cli({
        "mode": "check_attention_tactics",
        "schema_version": 1,
        "target": TARGET,
        "initial_launch": {
            "mapping": "scalar", "head_dim": 256, "query_tile": 8,
            "key_tile": 8, "threads_per_threadgroup": 8, "dtype_bytes": 4,
            "stage_k": True, "stage_v": True, "extra_shared_bytes": 0,
        },
        "tactics": [{"kind": "set_mapping", "mapping": "simdgroup"},
                    {"kind": "set_query_tile", "value": 4}],
    })
    assert out["accepted"] is True, f"legacy tactics mode changed: {out}"


def test_legacy_stmt_unchanged():
    stmt = {"tag": "skip"}
    out = run_cli({"stmt": stmt})
    assert "error" not in out
    assert out["stmt"]["tag"] == "skip"


if __name__ == "__main__":
    test_default_accepted(); print("PASS: default accepted (shared 20608, threads 32)")
    test_memory_exact_boundary_accepted(); print("PASS: exact memory boundary accepted")
    test_memory_one_less_rejected(); print("PASS: one byte under budget rejected")
    test_key_tile_16_memory_boundary(); print("PASS: kt16 memory boundary")
    test_invalid_query_tile_rejected(); print("PASS: invalid query tile rejected")
    test_invalid_simd_width_rejected(); print("PASS: invalid simd width rejected")
    test_bad_head_dim_rejected(); print("PASS: bad head dim rejected")
    test_bad_key_tile_rejected(); print("PASS: bad key tile rejected")
    test_zero_seq_len_rejected(); print("PASS: zero seq_len rejected")
    test_malformed_plan_rejected(); print("PASS: malformed plan rejected")
    test_malformed_tactic_rejected(); print("PASS: malformed tactic rejected")
    test_missing_tactic_value_rejected(); print("PASS: missing tactic value rejected")
    test_successful_tactic_sequence(); print("PASS: successful tactic sequence")
    test_reuse_kv_is_accepted_tactic(); print("PASS: reuse_kv_storage accepted tactic")
    test_alias_false_accepted_not_executable(); print("PASS: alias_kv=false accepted, not executable")
    test_alias_false_bytes_saved_on_reuse(); print("PASS: reuse reduces bytes (28928 -> 20608)")
    test_alias_false_overbudget_initial_rejected(); print("PASS: alias=false overbudget initial rejected")
    test_alias_false_fits_32k_both_dims(); print("PASS: alias=false fits 32k for D192/D256")
    test_reuse_then_tile16_d192_executable(); print("PASS: reuse->setQT16 D192 executable")
    test_early_failure_stops(); print("PASS: early failure stops")
    test_initial_illegal(); print("PASS: initial illegal")
    test_bad_schema_rejected(); print("PASS: bad schema rejected")
    test_missing_initial_plan_rejected(); print("PASS: missing initial_plan rejected")
    test_smoke_n1024_runtime(); print("PASS: N1024 smoke runtime")
    test_legacy_launch_unchanged(); print("PASS: legacy launch unchanged")
    test_legacy_tactics_unchanged(); print("PASS: legacy tactics unchanged")
    test_legacy_stmt_unchanged(); print("PASS: legacy stmt unchanged")
    print("All attention-plan tests passed!")
