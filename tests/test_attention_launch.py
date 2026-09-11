"""End-to-end tests for the `check_attention_launch` CLI mode.

These invoke the built `veritac` binary (like the other CLI tests) and lock in
the machine-checked `LaunchLegal` acceptance boundary: backend, head dimension,
tile shapes, thread limit, dtype, staging, and shared-memory byte accounting.
"""

import json
import subprocess
from pathlib import Path

BIN = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"

# A normalized Metal target with a threadgroup-memory limit of 32768 bytes, so
# that K16 (32768) fits exactly but K16 + 4 bytes overflows.
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


def launch(**overrides) -> dict:
    base = {
        "head_dim": 256,
        "query_tile": 8,
        "key_tile": 8,
        "threads_per_threadgroup": 8,
        "dtype_bytes": 4,
        "stage_k": True,
        "stage_v": True,
        "extra_shared_bytes": 0,
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


def check_attention(target: dict, l: dict) -> dict:
    return run_cli({
        "mode": "check_attention_launch",
        "schema_version": 1,
        "target": target,
        "launch": l,
    })


def check_tactics(target: dict, initial: dict, tactics: list) -> dict:
    return run_cli({
        "mode": "check_attention_tactics",
        "schema_version": 1,
        "target": target,
        "initial_launch": initial,
        "tactics": tactics,
    })


def target_with(**overrides) -> dict:
    tgt = dict(TARGET)
    tgt.update(overrides)
    return tgt


def test_d256_k8_accepted():
    out = check_attention(TARGET, launch())
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["shared_memory_bytes"] == 16384
    assert out["diagnostic"] == "accepted"
    assert out["schema_version"] == 1


def test_d256_k16_accepted():
    out = check_attention(TARGET, launch(key_tile=16))
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["shared_memory_bytes"] == 32768


def test_overflow_rejected():
    """K16 needs exactly 32768 bytes; +4 must overflow the 32768 limit."""
    out = check_attention(TARGET, launch(key_tile=16, extra_shared_bytes=4))
    assert out["accepted"] is False
    assert out["shared_memory_bytes"] == 32772


def test_bad_query_tile_rejected():
    out = check_attention(TARGET, launch(query_tile=32, threads_per_threadgroup=32))
    assert out["accepted"] is False
    assert "query_tile" in out["diagnostic"]


def test_bad_key_tile_rejected():
    out = check_attention(TARGET, launch(key_tile=32))
    assert out["accepted"] is False
    assert "key_tile" in out["diagnostic"]


def test_bad_dtype_rejected():
    out = check_attention(TARGET, launch(dtype_bytes=2))
    assert out["accepted"] is False
    assert "dtype_bytes" in out["diagnostic"]


def test_thread_limit_rejected():
    """threads must equal query_tile, so 16 threads with an 8 query tile is
    invalid; and 16 threads over a tiny limit is also invalid."""
    out = check_attention(TARGET, launch(query_tile=8, threads_per_threadgroup=16))
    assert out["accepted"] is False

    small = dict(TARGET)
    small["max_threads_per_threadgroup"] = 8
    out = check_attention(small, launch(threads_per_threadgroup=16, query_tile=16))
    assert out["accepted"] is False
    assert "threads exceed target limit" in out["diagnostic"]


def test_missing_staging_rejected():
    for stage_k, stage_v in ((False, True), (True, False)):
        out = check_attention(TARGET, launch(stage_k=stage_k, stage_v=stage_v))
        assert out["accepted"] is False
        assert "staging" in out["diagnostic"]


def test_backend_rejected():
    tgt = dict(TARGET)
    tgt["backend"] = "cuda"
    out = check_attention(tgt, launch())
    assert out["accepted"] is False
    assert "backend" in out["diagnostic"]


def test_bad_schema_rejected():
    out = run_cli({
        "mode": "check_attention_launch",
        "schema_version": 2,
        "target": TARGET,
        "launch": launch(),
    })
    assert "error" in out
    assert "schema_version" in out["error"]


def test_missing_schema_rejected():
    out = run_cli({
        "mode": "check_attention_launch",
        "target": TARGET,
        "launch": launch(),
    })
    assert "error" in out
    assert "schema_version" in out["error"]


def test_malformed_target_rejected():
    out = run_cli({
        "mode": "check_attention_launch",
        "schema_version": 1,
        "target": {"backend": "metal"},
        "launch": launch(),
    })
    assert "error" in out
    assert "Target" in out["error"]


def test_malformed_launch_rejected():
    out = run_cli({
        "mode": "check_attention_launch",
        "schema_version": 1,
        "target": TARGET,
        "launch": {"head_dim": 256},
    })
    assert "error" in out
    assert "Launch" in out["error"]


def test_legacy_request_works():
    stmt = {
        "tag": "loop", "var": "i0",
        "lo": {"tag": "lit", "val": 0}, "hi": {"tag": "lit", "val": 256},
        "ann": "none",
        "body": {"tag": "bufWrite", "buf": "C",
                 "indices": [{"tag": "var", "name": "i0"}],
                 "val": {"tag": "bufRead", "buf": "A",
                         "indices": [{"tag": "var", "name": "i0"}]}},
    }
    out = run_cli({
        "stmt": stmt,
        "tactics": [{"kind": "tile", "vars": ["i0"], "int_params": [32]}],
    })
    assert "error" not in out, f"legacy request should work: {out}"
    assert out["applied"] == 1
    assert out["stmt"]["var"] == "i0_outer"


def test_mapping_scalar_default_and_explicit():
    """scalar is the default mapping, and an explicit scalar mapping matches."""
    out = check_attention(TARGET, launch())
    assert out["accepted"] is True
    out = check_attention(TARGET, launch(mapping="scalar"))
    assert out["accepted"] is True


def test_mapping_simd_q4_q8():
    """simdgroup layout accepts query tiles 4 and 8 with 32 threads/row."""
    for q in (4, 8):
        out = check_attention(TARGET, launch(
            mapping="simdgroup", query_tile=q, threads_per_threadgroup=q * 32))
        assert out["accepted"] is True, f"simdgroup q{q} should be accepted: {out}"
        assert out["shared_memory_bytes"] == 16384


def test_wrong_simd_width_rejected():
    """simdgroup layout requires simd_width 32."""
    tgt = target_with(simd_width=64)
    out = check_attention(tgt, launch(
        mapping="simdgroup", query_tile=4, threads_per_threadgroup=4 * 64))
    assert out["accepted"] is False
    assert "simd_width" in out["diagnostic"]


def test_simd_thread_count_mismatch_rejected():
    out = check_attention(TARGET, launch(
        mapping="simdgroup", query_tile=4, threads_per_threadgroup=8))
    assert out["accepted"] is False
    assert "threads" in out["diagnostic"]


def test_memory_limit_overflow_rejected_tactics():
    """K16 needs 32768 bytes; on a 16384-byte target the set_key_tile 16 tactic
    is rejected, so only the earlier set_query_tile 8 is applied."""
    tgt = target_with(max_threadgroup_memory_bytes=16384)
    out = check_tactics(tgt, launch(), [
        {"kind": "set_query_tile", "value": 8},
        {"kind": "set_key_tile", "value": 16},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 1
    assert out["final_launch"]["query_tile"] == 8
    assert out["final_launch"]["key_tile"] == 8
    assert "shared" in out["diagnostic"]


def test_tactics_empty_sequence_accepted():
    """an empty tactic sequence is accepted via the core checker when the
    initial launch is legal, applying 0 tactics."""
    out = check_tactics(TARGET, launch(), [])
    assert out["accepted"] is True
    assert out["applied"] == 0
    assert out["final_launch"]["key_tile"] == 8
    assert out["shared_memory_bytes"] == 16384
    assert out["diagnostic"] == "accepted"


def test_tactics_initial_illegal():
    """an illegal initial launch gives applied 0 and keeps the initial launch."""
    out = check_tactics(TARGET, launch(dtype_bytes=2), [
        {"kind": "set_key_tile", "value": 8},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 0
    assert out["final_launch"]["dtype_bytes"] == 2
    assert "dtype_bytes" in out["diagnostic"]


def test_tactics_rejection_stops_before_later_tactics():
    """on the first illegal tactic, applied counts only accepted tactics, the
    final launch is the last accepted one, and the diagnostic describes the
    rejected proposed state."""
    out = check_tactics(TARGET, launch(), [
        {"kind": "set_mapping", "mapping": "simdgroup"},
        {"kind": "set_query_tile", "value": 4},
        {"kind": "set_key_tile", "value": 32},
        {"kind": "set_key_tile", "value": 8},
    ])
    assert out["accepted"] is False
    assert out["applied"] == 2
    assert out["final_launch"]["mapping"] == "simdgroup"
    assert out["final_launch"]["query_tile"] == 4
    assert out["final_launch"]["key_tile"] == 8
    assert out["shared_memory_bytes"] == 16384
    assert "key_tile" in out["diagnostic"]


def test_tactics_successful_sequence():
    """scalar q8/k16 -> set_mapping simdgroup -> set_query_tile 4 ->
    set_key_tile 8 is accepted with 3 applied tactics."""
    out = check_tactics(TARGET, launch(key_tile=16), [
        {"kind": "set_mapping", "mapping": "simdgroup"},
        {"kind": "set_query_tile", "value": 4},
        {"kind": "set_key_tile", "value": 8},
    ])
    assert out["accepted"] is True, f"expected accepted: {out}"
    assert out["applied"] == 3
    final = out["final_launch"]
    assert final["mapping"] == "simdgroup"
    assert final["query_tile"] == 4
    assert final["key_tile"] == 8
    assert final["threads_per_threadgroup"] == 128
    assert out["shared_memory_bytes"] == 16384
    assert out["diagnostic"] == "accepted"


def test_tactics_malformed_tactic_rejected():
    out = check_tactics(TARGET, launch(), [
        {"kind": "bogus", "value": 8},
    ])
    assert "error" in out
    assert "Tactic" in out["error"]


def test_tactics_malformed_tactic_missing_value_rejected():
    out = check_tactics(TARGET, launch(), [
        {"kind": "set_query_tile"},
    ])
    assert "error" in out
    assert "Tactic" in out["error"]


def test_unknown_mode_rejected():
    out = run_cli({
        "mode": "check_attention_frobnicate",
        "schema_version": 1,
        "target": TARGET,
        "launch": launch(),
    })
    assert "error" in out
    assert "Unknown mode" in out["error"]


def test_tactics_bad_top_schema_rejected():
    out = run_cli({
        "mode": "check_attention_tactics",
        "schema_version": 2,
        "target": TARGET,
        "initial_launch": launch(),
        "tactics": [],
    })
    assert "error" in out
    assert "schema_version" in out["error"]


def test_tactics_missing_initial_rejected():
    out = run_cli({
        "mode": "check_attention_tactics",
        "schema_version": 1,
        "target": TARGET,
        "tactics": [],
    })
    assert "error" in out
    assert "initial_launch" in out["error"]


def test_bad_target_schema_rejected():
    """target objects must declare their own schema_version 1."""
    tgt = target_with(schema_version=2)
    out = check_attention(tgt, launch())
    assert "error" in out
    assert "Target" in out["error"]


def test_target_missing_schema_rejected():
    tgt = target_with()
    del tgt["schema_version"]
    out = check_attention(tgt, launch())
    assert "error" in out
    assert "Target" in out["error"]


def test_tactics_legacy_mode_absent():
    """with no mode field, the legacy stmt/tactics protocol still runs."""
    stmt = {
        "tag": "skip",
    }
    out = run_cli({"stmt": stmt})
    assert "error" not in out
    assert out["stmt"]["tag"] == "skip"


if __name__ == "__main__":
    test_d256_k8_accepted(); print("PASS: D256 K8 accepted (16384)")
    test_d256_k16_accepted(); print("PASS: D256 K16 accepted (32768)")
    test_overflow_rejected(); print("PASS: +4 bytes rejected")
    test_bad_query_tile_rejected(); print("PASS: bad query tile rejected")
    test_bad_key_tile_rejected(); print("PASS: bad key tile rejected")
    test_bad_dtype_rejected(); print("PASS: bad dtype rejected")
    test_thread_limit_rejected(); print("PASS: thread limit rejected")
    test_missing_staging_rejected(); print("PASS: missing staging rejected")
    test_backend_rejected(); print("PASS: non-metal backend rejected")
    test_bad_schema_rejected(); print("PASS: bad schema rejected")
    test_missing_schema_rejected(); print("PASS: missing schema rejected")
    test_malformed_target_rejected(); print("PASS: malformed target rejected")
    test_malformed_launch_rejected(); print("PASS: malformed launch rejected")
    test_mapping_scalar_default_and_explicit(); print("PASS: scalar default/explicit")
    test_mapping_simd_q4_q8(); print("PASS: SIMD q4/q8 accepted")
    test_wrong_simd_width_rejected(); print("PASS: wrong SIMD width rejected")
    test_simd_thread_count_mismatch_rejected(); print("PASS: SIMD thread mismatch rejected")
    test_memory_limit_overflow_rejected_tactics(); print("PASS: memory-limit sequence rejected")
    test_tactics_empty_sequence_accepted(); print("PASS: empty tactics sequence accepted")
    test_tactics_initial_illegal(); print("PASS: tactics initial illegal")
    test_tactics_rejection_stops_before_later_tactics(); print("PASS: tactics rejection stops early")
    test_tactics_successful_sequence(); print("PASS: tactics successful sequence")
    test_tactics_malformed_tactic_rejected(); print("PASS: malformed tactic rejected")
    test_tactics_malformed_tactic_missing_value_rejected(); print("PASS: malformed tactic missing value")
    test_unknown_mode_rejected(); print("PASS: unknown mode rejected")
    test_tactics_bad_top_schema_rejected(); print("PASS: tactics bad top schema")
    test_tactics_missing_initial_rejected(); print("PASS: tactics missing initial")
    test_bad_target_schema_rejected(); print("PASS: bad target schema")
    test_target_missing_schema_rejected(); print("PASS: target missing schema")
    test_legacy_request_works(); print("PASS: legacy request works")
    test_tactics_legacy_mode_absent(); print("PASS: legacy with mode absent")
    print("All attention-launch tests passed!")
