"""End-to-end tests for the `check_cuda_attention_plan` CLI mode.

These invoke the built `veritac` binary with a synthetic CUDA config catalogue
built from known real runtime observations (id1 Q32K128 supported; id2 Q32K64
supported; id0 Q32K256 unsupported).  They lock in the Lean-checked
`CudaCheckLegal` boundary: backend cuda, warp 32, threads = Q*K/32 within kernel
and device limits, dynamic+static shared memory within the device budget,
head_dim 192/256 <= max_k, supported, plus unknown/duplicate id rejection and
per-row partition legality.  Old modes are asserted unchanged.
"""

import json
import subprocess
import time
from pathlib import Path

BIN = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"

TARGET = {
    "schema_version": 1,
    "backend": "cuda",
    "device_name": "test-cuda",
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 101376,
    "simd_width": 32,
    "toolchain": {"name": "cuda_sdk", "version": "12.4"},
    "provenance": {"source": "test", "host": "localhost"},
    "features": {"matrix_ops": None, "async_copy": None, "fp16": None, "bfloat16": None},
}


def config(**kw) -> dict:
    base = {
        "id": 1,
        "queries_per_block": 32,
        "keys_per_block": 128,
        "max_k": 256,
        "num_threads": 128,
        "smem_bytes": 76288,
        "static_shared_bytes": 0,
        "kernel_max_threads": 128,
        "supported": True,
    }
    base.update(kw)
    return base


CATALOG = [
    config(id=1),                                        # Q32 K128, 128 thr, 76288 smem
    config(id=2, keys_per_block=64, num_threads=64, smem_bytes=37632),  # Q32 K64
    config(id=0, keys_per_block=256, num_threads=256, smem_bytes=133632,
           kernel_max_threads=256, supported=False),     # unsupported
]


def plan(**kw) -> dict:
    base = {"seq_len": 128, "head_dim": 192, "config_id": 1}
    base.update(kw)
    return base


def run_cli(payload: dict) -> dict:
    result = subprocess.run(
        [str(BIN), "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def check_cuda(target: dict, catalog: list, initial: dict, tactics: list) -> dict:
    return run_cli({
        "mode": "check_cuda_attention_plan",
        "schema_version": 1,
        "target": target,
        "catalog": catalog,
        "initial_plan": initial,
        "tactics": tactics,
    })


def target_with(**overrides) -> dict:
    tgt = dict(TARGET)
    tgt.update(overrides)
    return tgt


def test_legal_config_accepted():
    out = check_cuda(TARGET, CATALOG, plan(), [])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["applied"] == 0
    assert out["shared_memory_bytes"] == 76288
    assert out["threads_per_threadgroup"] == 128
    fp = out["final_plan"]
    assert fp["config_id"] == 1
    assert fp["queries_per_block"] == 32 and fp["keys_per_block"] == 128
    assert fp["seq_len"] == 128 and fp["head_dim"] == 192
    assert out["diagnostic"] == "accepted"
    assert out["schema_version"] == 1


def test_memory_exact_boundary():
    """id1 uses exactly 76288 bytes; an exact budget is accepted, one less is not."""
    ok = check_cuda(target_with(max_threadgroup_memory_bytes=76288), CATALOG, plan(), [])
    assert ok["accepted"] is True, f"exact budget should fit: {ok}"
    bad = check_cuda(target_with(max_threadgroup_memory_bytes=76287), CATALOG, plan(), [])
    assert bad["accepted"] is False
    assert "shared" in bad["diagnostic"]


def test_wrong_threads_rejected():
    """num_threads must equal queries_per_block*keys_per_block/32 (128 here)."""
    catalog = CATALOG + [config(id=5, num_threads=100)]
    out = check_cuda(TARGET, catalog, plan(config_id=5), [])
    assert out["accepted"] is False
    assert "threads must equal" in out["diagnostic"]


def test_oversized_rejected():
    """supported config whose dynamic+static smem exceeds the device budget."""
    catalog = CATALOG + [config(id=6, smem_bytes=200000, static_shared_bytes=1000)]
    out = check_cuda(TARGET, catalog, plan(config_id=6), [])
    assert out["accepted"] is False
    assert "shared memory exceeds" in out["diagnostic"]


def test_non_cuda_rejected():
    tgt = target_with(backend="metal")
    out = check_cuda(tgt, CATALOG, plan(), [])
    assert out["accepted"] is False
    assert "not cuda" in out["diagnostic"]


def test_warp_not32_rejected():
    tgt = target_with(simd_width=64)
    out = check_cuda(tgt, CATALOG, plan(), [])
    assert out["accepted"] is False
    assert "simd_width" in out["diagnostic"]


def test_bad_head_dim_rejected():
    out = check_cuda(TARGET, CATALOG, plan(head_dim=128), [])
    assert out["accepted"] is False
    assert "head_dim" in out["diagnostic"]


def test_maxk_rejects_d256():
    """a config with max_k 192 rejects head_dim 256."""
    catalog = CATALOG + [config(id=7, max_k=192)]
    out = check_cuda(TARGET, catalog, plan(head_dim=256, config_id=7), [])
    assert out["accepted"] is False
    assert "max_k" in out["diagnostic"]


def test_unknown_id_rejected():
    out = check_cuda(TARGET, CATALOG, plan(config_id=9), [])
    assert out["accepted"] is False
    assert "unknown config id" in out["diagnostic"]


def test_duplicate_ids_rejected():
    catalog = CATALOG + [config(id=1)]
    out = check_cuda(TARGET, catalog, plan(), [])
    assert out["accepted"] is False
    assert "duplicate" in out["diagnostic"]


def test_successful_select():
    out = check_cuda(TARGET, CATALOG, plan(), [{"kind": "select_config", "value": 2}])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["applied"] == 1
    assert out["final_plan"]["config_id"] == 2
    assert out["final_plan"]["keys_per_block"] == 64
    assert out["threads_per_threadgroup"] == 64
    assert out["shared_memory_bytes"] == 37632


def test_early_failure_keeps_last_accepted():
    """unknown id stops the sequence; applied counts only accepted tactics."""
    out = check_cuda(TARGET, CATALOG, plan(), [
        {"kind": "select_config", "value": 2},
        {"kind": "select_config", "value": 9},
        {"kind": "select_config", "value": 1},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 1
    assert out["final_plan"]["config_id"] == 2
    assert "unknown config id" in out["diagnostic"]


def test_initial_illegal():
    out = check_cuda(TARGET, CATALOG, plan(config_id=9), [{"kind": "select_config", "value": 2}])
    assert out["accepted"] is False
    assert out["applied"] == 0
    assert out["final_plan"]["config_id"] == 9


def test_unsupported_config_rejected():
    out = check_cuda(TARGET, CATALOG, plan(config_id=0), [])
    assert out["accepted"] is False
    assert "unsupported" in out["diagnostic"]


def test_missing_fields_rejected():
    base = {"mode": "check_cuda_attention_plan", "schema_version": 1,
            "target": TARGET, "catalog": CATALOG, "initial_plan": plan(), "tactics": []}
    out = run_cli({k: v for k, v in base.items() if k != "catalog"})
    assert "error" in out and "catalog" in out["error"]
    out = run_cli({k: v for k, v in base.items() if k != "initial_plan"})
    assert "error" in out and "initial_plan" in out["error"]
    out = run_cli({**base, "initial_plan": {"head_dim": 192, "config_id": 1}})
    assert "error" in out and "plan" in out["error"]


def test_malformed_tactic_rejected():
    out = check_cuda(TARGET, CATALOG, plan(), [{"kind": "bogus", "value": 1}])
    assert "error" in out
    assert "Tactic" in out["error"]


def test_bad_schema_rejected():
    out = run_cli({
        "mode": "check_cuda_attention_plan", "schema_version": 2,
        "target": TARGET, "catalog": CATALOG, "initial_plan": plan(), "tactics": [],
    })
    assert "error" in out and "schema_version" in out["error"]


def test_n1024_runtime():
    start = time.monotonic()
    out = check_cuda(TARGET, CATALOG, plan(seq_len=1024), [])
    elapsed = time.monotonic() - start
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert elapsed < 5.0, f"N1024 check too slow: {elapsed:.2f}s"


def test_old_modes_preserved():
    # legacy check_attention_launch
    out = run_cli({
        "mode": "check_attention_launch", "schema_version": 1,
        "target": target_with(backend="metal"),
        "launch": {"mapping": "scalar", "head_dim": 256, "query_tile": 8,
                   "key_tile": 8, "threads_per_threadgroup": 8, "dtype_bytes": 4,
                   "stage_k": True, "stage_v": True, "extra_shared_bytes": 0},
    })
    assert out["accepted"] is True, "legacy launch mode changed"
    # legacy stmt mode
    out = run_cli({"stmt": {"tag": "skip"}})
    assert out["stmt"]["tag"] == "skip", "legacy stmt mode changed"


if __name__ == "__main__":
    test_legal_config_accepted(); print("PASS: legal config accepted (76288, 128 thr)")
    test_memory_exact_boundary(); print("PASS: memory exact boundary")
    test_wrong_threads_rejected(); print("PASS: wrong threads rejected")
    test_oversized_rejected(); print("PASS: oversized rejected")
    test_non_cuda_rejected(); print("PASS: non-cuda rejected")
    test_warp_not32_rejected(); print("PASS: warp != 32 rejected")
    test_bad_head_dim_rejected(); print("PASS: bad head dim rejected")
    test_maxk_rejects_d256(); print("PASS: max_k 192 rejects D256")
    test_unknown_id_rejected(); print("PASS: unknown id rejected")
    test_duplicate_ids_rejected(); print("PASS: duplicate ids rejected")
    test_successful_select(); print("PASS: successful select")
    test_early_failure_keeps_last_accepted(); print("PASS: early failure keeps last accepted")
    test_initial_illegal(); print("PASS: initial illegal")
    test_unsupported_config_rejected(); print("PASS: unsupported config rejected")
    test_missing_fields_rejected(); print("PASS: missing fields rejected")
    test_malformed_tactic_rejected(); print("PASS: malformed tactic rejected")
    test_bad_schema_rejected(); print("PASS: bad schema rejected")
    test_n1024_runtime(); print("PASS: N1024 runtime")
    test_old_modes_preserved(); print("PASS: old modes preserved")
    print("All CUDA attention-plan tests passed!")
