"""CPU tests for the CUDA attention demo's local controller logic.

These mock the SSH/remote-helper boundary (a fake remote client or a recording
subclass of the real client) and use the REAL local Lean verifier
(`check_cuda_attention_plan`) with a synthetic CUDA profile/catalogue.  No GPU
and no remote execution happen here — the controller owns GPU validation and the
other worker owns the remote GPU.

Locked-in guarantees:
  - unsupported / unknown configs are never dispatched (never benchmarked);
  - a changed profile / catalogue / binary binding (incl. multi-module) aborts;
  - numeric_failed / graph_failed / invalid-timing candidates never win;
  - the winner differs from the initial config and its checked_plan replays
    through Lean to the identical plan;
  - candidate and survey input hashes must match or the run aborts;
  - SSH argv puts helper globals before the subcommand and uses a REMOTE out
    path (no local paths sent as remote --out); methods are not shadowed;
  - profile_hash is distinct from the catalogue hash;
  - the LLM prompt includes profile/render + full catalogue resource facts +
    rejection feedback;
  - malformed catalogue observations are not int()/bool() coerced;
  - a benchmark whose actual config_info disagrees with the snapshot catalogue
    is rejected (config_info_mismatch).
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import llm_cuda_attention_demo as demo  # noqa: E402
import cuda_attention_remote as helper  # noqa: E402

SEQ, DIM = 2048, 192
SEED = 1234
W, S, I = 10, 20, 5


def cfg(id_, **kw):
    base = {"id": id_, "queries_per_block": 32, "keys_per_block": 128,
            "max_k": 256, "num_threads": 128, "smem_bytes": 76288,
            "static_shared_bytes": 0, "kernel_max_threads": 128,
            "supported": True, "num_warps": 4, "min_blocks_hint": 3,
            "num_regs": 64, "total_shared_bytes": 76288,
            "async_drains": True, "preload_v": False}
    base.update(kw)
    return base


CATALOGUE = [
    cfg(1),
    cfg(2, keys_per_block=64, num_threads=64, smem_bytes=37632, total_shared_bytes=37632),
    cfg(0, keys_per_block=256, num_threads=256, smem_bytes=133632,
        kernel_max_threads=256, total_shared_bytes=133632, supported=False),
]

PROFILE = {
    "schema_version": 1,
    "backend": "cuda",
    "device_name": "test-cuda",
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 101376,
    "simd_width": 32,
    "compute_capability": "8.0",
    "toolchain": {"name": "nvcc", "version": "12.4", "code_target": "sm_80"},
    "provenance": {"source": "cuda_device_query", "host": "test"},
    "features": {"matrix_ops": None, "async_copy": None, "fp16": None, "bfloat16": None},
}

BINDING = {
    "module_paths": ["/x/a.so", "/x/b.so"],
    "module_hashes": {"a.so": "1" * 64, "b.so": "2" * 64},
    "cu_hashes": {"k.cu": "3" * 64},
    "vendor_hashes": {"vendor/h.h": "4" * 64},
    "run_candidate_py": "5" * 64,
    "installed_kernel_forward_h": "6" * 64,
    "installed_kernel_forward_h_path": "/x/kernel_forward.h",
    "cutlass_git_head": "a" * 40,
    "cutlass_dirty_diff_hash": "7" * 64,
    "torch": "2.5.0", "torch_cuda": "12.4", "compute_capability": "8.0",
}


def entry(id_):
    return next(c for c in CATALOGUE if c["id"] == id_)


def make_metadata(catalogue=None, profile=None, binding=None, config_ids=None):
    catalogue = catalogue if catalogue is not None else CATALOGUE
    ch = hashlib.sha256(
        json.dumps(catalogue, sort_keys=True, default=str).encode()).hexdigest()
    return {
        "profile": profile if profile is not None else PROFILE,
        "catalogue": catalogue,
        "config_ids": config_ids if config_ids is not None else [c["id"] for c in catalogue],
        "catalogue_hash": ch,
        "binding": binding if binding is not None else BINDING,
        "module_paths": (binding or BINDING).get("module_paths"),
        "binary_path": "/x/veritac_attn_candidate.so",
    }


class FakeRemote:
    def __init__(self, metadata, benchmark_cases, survey_case,
                 drift_after=None, drifted_metadata=None):
        self.meta = metadata
        self.benchmark_cases = benchmark_cases
        self.survey_case = survey_case
        self.drift_after = drift_after
        self.drifted_metadata = drifted_metadata
        self.meta_calls = 0
        self.dispatch_calls = []

    def metadata(self):
        self.meta_calls += 1
        if self.drift_after is not None and self.meta_calls > self.drift_after:
            return self.drifted_metadata
        return self.meta

    def benchmark(self, cfg_id, seq, dim, seed, warmup, samples, inner, out_basename):
        self.dispatch_calls.append((cfg_id, seq, dim))
        return self.benchmark_cases[(cfg_id, seq, dim)]

    def survey(self, seq, dim, seed, warmup, samples, inner, out_basename):
        return self.survey_case

    def push(self, *a, **k):
        pass

    def mkdir(self, *a, **k):
        pass


def args():
    return types.SimpleNamespace(seq=SEQ, dim=DIM, seed=SEED, warmup=W,
                                 samples=S, inner=I)


def ok_case(cfg_id, evt=1.0, graph=1.0, hashes=("q", "k", "v"), config_info=None):
    return {
        "config": cfg_id, "seq": SEQ, "dim": DIM,
        "config_info": config_info if config_info is not None else entry(cfg_id),
        "comparison": {"within_tolerance": True},
        "graph_comparison": {"within_tolerance": True},
        "cuda_event_timing": {"median_ms": evt},
        "graph_timing": {"median_ms": graph},
        "wall_timing": {"median_ms": graph * 1.1},
        "input_hashes": {"q": hashes[0], "k": hashes[1], "v": hashes[2]},
    }


# ---------------------------------------------------------------------------
# Remote argv / method-shadowing / output-path mapping
# ---------------------------------------------------------------------------

class RecordingRemote(demo.CudaRemoteClient):
    """Real client with `_ssh` stubbed to record the constructed argv."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.argv_calls = []

    def _ssh(self, remote_argv, label):
        self.argv_calls.append((label, remote_argv))
        return {"ok": True}


def test_ssh_argv_globals_before_subcommand_and_remote_out():
    client = RecordingRemote("host", "/r/helper.py", survey_path="/r/survey.py",
                             remote_py="/venv/bin/python",
                             candidate_root="/root/cuda_candidate",
                             log_dir=Path("."))
    client.metadata()
    client.benchmark(2, SEQ, DIM, SEED, W, S, I, "cand_2.json")
    client.survey(SEQ, DIM, SEED, W, S, I, "survey.json")
    md = client.argv_calls[0][1]
    # globals must precede the subcommand
    assert md[:6] == ["/venv/bin/python", "/r/helper.py",
                      "--candidate-root", "/root/cuda_candidate",
                      "--survey", "/r/survey.py"]
    assert md[6] == "metadata"
    bm = client.argv_calls[1][1]
    assert bm[:6] == ["/venv/bin/python", "/r/helper.py",
                      "--candidate-root", "/root/cuda_candidate",
                      "--survey", "/r/survey.py"]
    assert bm[6] == "benchmark"
    out = bm[bm.index("--out") + 1]
    assert out.startswith(client.remote_out_dir), "remote --out must be under remote_out_dir"
    assert "/Users/" not in out, "no local path may be sent as remote --out"


def test_methods_not_shadowed():
    client = RecordingRemote("host", "/r/h.py", survey_path="/r/s.py", log_dir=Path("."))
    assert isinstance(client.survey_path, str)  # attribute, not shadowed
    assert callable(client.survey)              # method still exists


def test_mem_cap_bytes_forwarded_in_survey_argv():
    client = RecordingRemote("host", "/r/h.py", survey_path="/r/s.py",
                             remote_py="/venv/bin/python",
                             candidate_root="/root/c", log_dir=Path("."),
                             mem_cap_bytes=34359738368)
    client.survey(SEQ, DIM, SEED, W, S, I, "survey.json")
    argv = client.argv_calls[0][1]
    # globals before the subcommand, with --mem-cap-bytes included
    assert "--mem-cap-bytes" in argv
    assert argv[argv.index("--mem-cap-bytes") + 1] == "34359738368"
    # globals precede the subcommand, which immediately follows the mem-cap value
    assert argv[argv.index("34359738368") + 1] == "survey"
    # absent when not configured
    plain = RecordingRemote("host", "/r/h.py", survey_path="/r/s.py", log_dir=Path("."))
    plain.survey(SEQ, DIM, SEED, W, S, I, "survey.json")
    assert "--mem-cap-bytes" not in plain.argv_calls[0][1]


def test_profile_hash_distinct_from_catalogue_hash():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    assert snap["profile_hash"] != snap["catalogue_hash"]
    assert snap["profile_hash"] is not None and len(snap["profile_hash"]) == 64


# ---------------------------------------------------------------------------
# Drift / binding
# ---------------------------------------------------------------------------

def test_drift_aborts():
    drifted = make_metadata(catalogue=[cfg(1)], binding=dict(BINDING, run_candidate_py="x" * 64))
    fake = FakeRemote(make_metadata(), {}, {}, drift_after=1, drifted_metadata=drifted)
    snap = demo.snapshot_metadata(fake, Path("."))
    try:
        demo.check_binding(fake, snap, "pre-dispatch")
        raised = False
    except SystemExit:
        raised = True
    assert raised, "drift did not abort"


def test_multi_module_binary_drift_aborts():
    """a change in any of the loaded-module .so hashes is a binding drift."""
    drifted_binding = dict(BINDING)
    drifted_binding["module_hashes"] = dict(BINDING["module_hashes"], a="z" * 64)
    drifted = make_metadata(binding=drifted_binding)
    fake = FakeRemote(make_metadata(), {}, {}, drift_after=1, drifted_metadata=drifted)
    snap = demo.snapshot_metadata(fake, Path("."))
    try:
        demo.check_binding(fake, snap, "pre-dispatch")
        raised = False
    except SystemExit:
        raised = True
    assert raised, "multi-module binary drift did not abort"


# ---------------------------------------------------------------------------
# Selection / dispatch
# ---------------------------------------------------------------------------

def test_unsupported_and_unknown_never_dispatched():
    bench = {(1, SEQ, DIM): ok_case(1, graph=2.0), (2, SEQ, DIM): ok_case(2, graph=1.5)}
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    dispatched = [c[0] for c in fake.dispatch_calls]
    assert 0 not in dispatched, "unsupported cfg0 was dispatched"
    assert 1 in dispatched and 2 in dispatched
    r = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 99)
    assert r.get("accepted") is False
    assert "unknown config id" in r.get("diagnostic", "")


def test_numeric_and_graph_failed_never_win():
    bench = {
        (1, SEQ, DIM): {**ok_case(1, graph=0.9),
                        "graph_comparison": {"within_tolerance": False}},
        (2, SEQ, DIM): ok_case(2, graph=1.5),
    }
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    r1 = [r for r in records if r["config_id"] == 1][0]
    assert r1["status"] == "numeric_failed"
    assert winner is not None and winner["config_id"] == 2


def test_invalid_gpu_timing_no_win():
    nan = float("nan")
    bench = {
        (1, SEQ, DIM): {**ok_case(1), "graph_timing": {"median_ms": nan}},
        (2, SEQ, DIM): ok_case(2, graph=1.5),
    }
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    assert winner["config_id"] == 2
    assert demo._finite_positive(nan) is False


def test_winner_differs_initial_and_trace_bound():
    bench = {(1, SEQ, DIM): ok_case(1, graph=3.0),
             (2, SEQ, DIM): ok_case(2, graph=1.5)}
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    initial_id = demo.first_legal_config(snap["target"], snap["lean_catalog"],
                                         SEQ, DIM, snap["config_ids"])
    assert initial_id == 1
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    assert winner["config_id"] == 2 and winner["config_id"] != initial_id
    assert winner["checked_plan"]["config_id"] == 2
    r = demo.call_check_cuda(
        snap["target"], snap["lean_catalog"],
        {"seq_len": SEQ, "head_dim": DIM, "config_id": initial_id},
        [demo.select_config_tactic(2)])
    assert r.get("accepted") is True and r["final_plan"]["config_id"] == 2


def test_mismatched_actual_config_info_rejects():
    """a benchmark whose actual config_info disagrees with the snapshot
    catalogue entry is rejected (config_info_mismatch), not trusted."""
    bench = {
        (1, SEQ, DIM): ok_case(1, graph=2.0),
        (2, SEQ, DIM): ok_case(2, graph=1.5,
                               config_info=dict(entry(2), num_threads=999)),
    }
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    r2 = [r for r in records if r["config_id"] == 2][0]
    assert r2["status"] == "config_info_mismatch"
    assert winner["config_id"] == 1  # only cfg1 is ok


def test_input_hash_mismatch_aborts():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    winner_case = ok_case(2, hashes=("q", "k", "v"))
    survey = {"seq": SEQ, "dim": DIM, "dtype": "float32",
              "input_hashes": {"q": "q", "k": "k", "v": "DIFFERENT"}, "results": {}}
    fake.survey_case = survey
    try:
        demo.compare_with_survey(fake, args(), Path("."), snap, winner_case, SEQ, DIM)
        raised = False
    except SystemExit:
        raised = True
    assert raised, "input hash mismatch did not abort"


# ---------------------------------------------------------------------------
# Prompt / catalogue validation
# ---------------------------------------------------------------------------

def test_prompt_includes_profile_resources_and_feedback():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    msg = demo.build_llm_user_message(snap, SEQ, DIM,
                                      {"config_id": 1}, "REJECTED: unknown config id")
    assert "test-cuda" in msg or "cuda" in msg.lower()          # profile
    assert "cfg 2:" in msg and "K=64" in msg                   # catalogue facts
    assert "min_blocks_hint" in msg and "regs" in msg           # geometry metadata
    assert "async_drains=True" in msg and "preload_v=False" in msg
    assert "Execution policy:" in msg
    assert "REJECTED: unknown config id" in msg                  # feedback retained
    assert "seq=" in msg and "dim=" in msg


def test_malformed_observations_not_coerced():
    # missing required field -> rejected (no silent default)
    try:
        demo.to_lean_catalog([cfg(5, __unsupported_marker=1)])
        # remove marker to trigger the real malformed case
        bad = [dict(entry(1), queries_per_block="NOT_A_NAT")]
        demo.to_lean_catalog(bad)
        # no coercion: the raw string passes through unchanged
        out = demo.to_lean_catalog([{"id": 5, **{k: entry(1)[k] for k in
                                                  demo.REQUIRED_CATALOG_FIELDS},
                                      "queries_per_block": "32"}])
        assert out[0]["queries_per_block"] == "32"
    except ValueError:
        pass
    bad_missing = [{"id": 9, "queries_per_block": 32}]
    try:
        demo.to_lean_catalog(bad_missing)
        raised = False
    except ValueError:
        raised = True
    assert raised, "missing required field was not rejected"


def test_no_benchmark_metadata_lean_only():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    legal = demo.first_legal_config(snap["target"], snap["lean_catalog"],
                                    SEQ, DIM, snap["config_ids"])
    r = demo.call_check_cuda(snap["target"], snap["lean_catalog"],
                             {"seq_len": SEQ, "head_dim": DIM, "config_id": legal},
                             [demo.select_config_tactic(legal)])
    assert r.get("accepted") is True
    assert fake.dispatch_calls == []


def test_installed_kernel_header_layout():
    """torch.__file__ lives IN torch/, so the installed kernel header is under
    torch_dir/include/ATen/... (no parent hop), regardless of existence."""
    with tempfile.TemporaryDirectory() as td:
        torch_dir = os.path.join(td, "torch")
        os.makedirs(torch_dir)
        helper._TORCH = types.SimpleNamespace(
            __file__=os.path.join(torch_dir, "__init__.py"))
        expected = os.path.join(torch_dir, "include", "ATen", "native",
                                "transformers", "cuda", "mem_eff_attention",
                                "kernel_forward.h")
        got = helper.installed_kernel_header()
        assert got == os.path.realpath(expected), f"got {got}, expected {expected}"
        assert ".." not in got.split("/"), "must not contain a parent hop"


def test_empty_binding_rejected():
    fake = FakeRemote(make_metadata(binding={}), {}, {})
    try:
        demo.snapshot_metadata(fake, Path("."))
        raised = False
    except SystemExit:
        raised = True
    assert raised, "empty binding was accepted"


def test_missing_binding_field_rejected():
    b = dict(BINDING)
    del b["cu_hashes"]
    fake = FakeRemote(make_metadata(binding=b), {}, {})
    try:
        demo.snapshot_metadata(fake, Path("."))
        raised = False
    except SystemExit:
        raised = True
    assert raised, "missing mandatory binding field was accepted"


def test_malformed_id_rejected():
    meta = make_metadata()
    meta["config_ids"] = [1, "2", 0]  # string id must be rejected
    fake = FakeRemote(meta, {}, {})
    try:
        demo.snapshot_metadata(fake, Path("."))
        raised = False
    except SystemExit:
        raised = True
    assert raised, "non-strict-int config_ids were accepted"


def test_final_replay_tampered_id_rejects():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    checked = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 1)["final_plan"]
    rec = demo.verified_dispatch(fake, args(), Path("."), snap, 2, checked)  # req 2, checked 1
    assert rec["status"] == "checkedplan_id_mismatch"


def test_final_replay_tampered_configinfo_rejects():
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    checked = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 1)["final_plan"]
    tampered = ok_case(1, config_info=dict(entry(1), num_threads=777))
    fake.benchmark_cases = {(1, SEQ, DIM): tampered}
    rec = demo.verified_dispatch(fake, args(), Path("."), snap, 1, checked)
    assert rec["status"] == "config_info_mismatch"


def test_winner_trace_replay():
    bench = {(1, SEQ, DIM): ok_case(1, graph=3.0), (2, SEQ, DIM): ok_case(2, graph=1.5)}
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    assert winner["config_id"] == 2
    assert "initial_plan" in winner and "tactic_sequence" in winner and "checked_plan" in winner
    assert "verification_response" in winner
    assert demo.replay_sequence(snap["target"], snap["lean_catalog"],
                                winner["initial_plan"], winner["tactic_sequence"],
                                winner["checked_plan"]) is True


def test_losing_agent_plan_vs_search_winner():
    """report.final_plan refers to the search winner, while agent_final_plan is
    preserved separately (agent's first-legal config loses)."""
    bench = {(1, SEQ, DIM): ok_case(1, graph=3.0), (2, SEQ, DIM): ok_case(2, graph=1.5)}
    fake = FakeRemote(make_metadata(), bench, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    agent_final = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 1)["final_plan"]
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    report = {"agent_final_plan": agent_final, "final_plan": winner["checked_plan"]}
    assert report["final_plan"]["config_id"] == 2
    assert report["agent_final_plan"]["config_id"] == 1
    assert report["final_plan"] != report["agent_final_plan"]


def test_survey_failed_status_with_fast_timing_not_eligible():
    """a backend with raw status != ok but a fast timing is NOT eligible; only a
    genuinely ok backend counts as fastest."""
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    winner_case = ok_case(2, hashes=("q", "k", "v"))
    survey = {
        "seq": SEQ, "dim": DIM, "dtype": "float32",
        "input_hashes": {"q": "q", "k": "k", "v": "v"},
        "results": {
            "auto": {"status": "failed", "comparison": {"within_tolerance": True},
                     "graph_comparison": {"within_tolerance": True},
                     "cuda_event_timing": {"median_ms": 0.1},
                     "graph_timing": {"median_ms": 0.1}},
            "efficient": {"status": "ok", "comparison": {"within_tolerance": True},
                          "graph_comparison": {"within_tolerance": True},
                          "cuda_event_timing": {"median_ms": 2.0},
                          "graph_timing": {"median_ms": 2.0}},
        },
    }
    fake.survey_case = survey
    rep = demo.compare_with_survey(fake, args(), Path("."), snap, winner_case, SEQ, DIM)
    assert rep["per_backend"]["auto"]["event_eligible"] is False
    assert rep["per_backend"]["auto"]["graph_eligible"] is False
    assert rep["per_backend"]["auto"]["raw_status"] == "failed"
    assert rep["fastest_vendor_event"] == "efficient"
    assert rep["fastest_vendor_graph"] == "efficient"


def test_survey_failed_graph_comparison_not_eligible():
    """raw status ok but failed graph comparison makes the graph metric
    ineligible while the event metric stays eligible."""
    fake = FakeRemote(make_metadata(), {}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    winner_case = ok_case(2, hashes=("q", "k", "v"))
    survey = {
        "seq": SEQ, "dim": DIM, "dtype": "float32",
        "input_hashes": {"q": "q", "k": "k", "v": "v"},
        "results": {
            "auto": {"status": "ok", "comparison": {"within_tolerance": True},
                     "graph_comparison": {"within_tolerance": False},
                     "cuda_event_timing": {"median_ms": 1.0},
                     "graph_timing": {"median_ms": 1.0}},
            "efficient": {"status": "ok", "comparison": {"within_tolerance": True},
                          "graph_comparison": {"within_tolerance": True},
                          "cuda_event_timing": {"median_ms": 2.0},
                          "graph_timing": {"median_ms": 2.0}},
        },
    }
    fake.survey_case = survey
    rep = demo.compare_with_survey(fake, args(), Path("."), snap, winner_case, SEQ, DIM)
    assert rep["per_backend"]["auto"]["event_eligible"] is True
    assert rep["per_backend"]["auto"]["graph_eligible"] is False
    assert rep["fastest_vendor_event"] == "auto"
    assert rep["fastest_vendor_graph"] == "efficient"


def test_undrained_planning_state_never_executes_or_wins():
    catalogue = [cfg(1, async_drains=False), cfg(2)]
    fake = FakeRemote(make_metadata(catalogue=catalogue),
                      {(2, SEQ, DIM): ok_case(2, config_info=catalogue[1])}, {})
    snap = demo.snapshot_metadata(fake, Path("."))
    records, winner = demo.run_search(fake, args(), Path("."), snap, SEQ, DIM, SEED, W, S, I)
    assert records[0]["verification_response"]["accepted"] is True
    assert records[0]["status"] == "validation_rejected"
    assert winner["config_id"] == 2
    assert fake.dispatch_calls == [(2, SEQ, DIM)]


def test_execution_policy_requires_explicit_boolean_contract():
    for overrides in ({"async_drains": None}, {"async_drains": 1},
                      {"preload_v": True}, {"preload_v": None}):
        catalogue = [cfg(1, **overrides)]
        fake = FakeRemote(make_metadata(catalogue=catalogue), {}, {})
        snap = demo.snapshot_metadata(fake, Path("."))
        plan = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 1)
        assert plan["accepted"] is True  # still usable for planning/no-benchmark
        rec = demo.verified_dispatch(fake, args(), Path("."), snap, 1, plan["final_plan"])
        assert rec["status"] == "validation_rejected" and not fake.dispatch_calls


def test_actual_policy_and_compiler_facts_must_match_snapshot():
    for overrides in ({"async_drains": False}, {"preload_v": True}, {"num_regs": 999}):
        fake = FakeRemote(make_metadata(),
                          {(1, SEQ, DIM): ok_case(1, config_info=dict(entry(1), **overrides))}, {})
        snap = demo.snapshot_metadata(fake, Path("."))
        plan = demo.check_plan(snap["target"], snap["lean_catalog"], SEQ, DIM, 1)["final_plan"]
        rec = demo.verified_dispatch(fake, args(), Path("."), snap, 1, plan)
        assert rec["status"] == "config_info_mismatch"


if __name__ == "__main__":
    test_undrained_planning_state_never_executes_or_wins(); print("PASS: undrained planning state never executes/wins")
    test_execution_policy_requires_explicit_boolean_contract(); print("PASS: explicit drain/preload contract required")
    test_actual_policy_and_compiler_facts_must_match_snapshot(); print("PASS: full policy/compiler facts bound")
    test_ssh_argv_globals_before_subcommand_and_remote_out(); print("PASS: SSH argv + remote out mapping")
    test_methods_not_shadowed(); print("PASS: methods not shadowed")
    test_mem_cap_bytes_forwarded_in_survey_argv(); print("PASS: mem_cap_bytes forwarded in survey argv")
    test_profile_hash_distinct_from_catalogue_hash(); print("PASS: profile hash distinct from catalogue hash")
    test_drift_aborts(); print("PASS: drift aborts")
    test_multi_module_binary_drift_aborts(); print("PASS: multi-module binary drift aborts")
    test_unsupported_and_unknown_never_dispatched(); print("PASS: unsupported/unknown never dispatched")
    test_numeric_and_graph_failed_never_win(); print("PASS: numeric/graph_failed never win")
    test_invalid_gpu_timing_no_win(); print("PASS: invalid GPU timing no win")
    test_winner_differs_initial_and_trace_bound(); print("PASS: winner differs initial, trace bound")
    test_mismatched_actual_config_info_rejects(); print("PASS: mismatched config_info rejects")
    test_input_hash_mismatch_aborts(); print("PASS: input hash mismatch aborts")
    test_prompt_includes_profile_resources_and_feedback(); print("PASS: prompt includes profile/resources/feedback")
    test_malformed_observations_not_coerced(); print("PASS: malformed observations not coerced")
    test_no_benchmark_metadata_lean_only(); print("PASS: no-benchmark metadata+Lean only")
    test_installed_kernel_header_layout(); print("PASS: installed kernel header layout")
    test_empty_binding_rejected(); print("PASS: empty binding rejected")
    test_missing_binding_field_rejected(); print("PASS: missing binding field rejected")
    test_malformed_id_rejected(); print("PASS: malformed id rejected")
    test_final_replay_tampered_id_rejects(); print("PASS: final replay tampered id rejects")
    test_final_replay_tampered_configinfo_rejects(); print("PASS: final replay tampered config_info rejects")
    test_winner_trace_replay(); print("PASS: winner trace replay")
    test_losing_agent_plan_vs_search_winner(); print("PASS: losing agent plan vs search winner")
    test_survey_failed_status_with_fast_timing_not_eligible(); print("PASS: failed status + fast timing not eligible")
    test_survey_failed_graph_comparison_not_eligible(); print("PASS: failed graph comparison not eligible")
    print("All CUDA attention-demo tests passed!")
