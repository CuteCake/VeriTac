"""
VeriTac LLM-driven CUDA attention demo (local controller).

The controller runs on a Mac and drives a REMOTE CUDA host over SSH.  It pushes
a small remote helper (plus a copy of Hardware/profile.py) into a unique
controller_runs/<UUID> directory on the remote host, probes the actual CUDA
hardware, fetches the compiled `config_info` catalogue, and feeds an IMMUTABLE
normalized target + catalogue into the LOCAL Lean verifier
(`check_cuda_attention_plan`).  It then proposes `select_config` tactics (mock or
LLM), Lean-checks every one, and benchmarks only the authoritative ACCEPTED
config ids.

Everything the remote helper reports (profile, catalogue, binary/source binding
hashes) is an OBSERVED TRUST BOUNDARY, not a Lean proof: Lean only checks
resource/partition legality given those observations, and binding a selected
config to a specific executable hash is the controller's responsibility.

The candidate is the vendor-derived PyTorch/CUTLASS schedule specialization and
its native `OpMultiplyAddFastF32` (3-component TF32 emulation) — NOT IEEE scalar
FP32 and NOT a new precision reduction. Confirmed shapes and protocols are
documented in docs/attention_vendor_results.md.

Usage:
    python3 examples/llm_cuda_attention_demo.py --mock --no-benchmark   # metadata+Lean only
    python3 examples/llm_cuda_attention_demo.py --mock --search         # recommended: sweep
    OPENAI_API_KEY=sk-... python3 examples/llm_cuda_attention_demo.py  # LLM select_config

Requires `lake build veritac` (local Lean CLI).  Outputs under
`.lake/cuda_attention_demo/run_<uuid>`.
"""
import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import urllib.error
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import llm_attention_demo as lad  # noqa: E402  (reuse LLM client + Hardware helpers)

OUT = ROOT / ".lake" / "cuda_attention_demo"
VERITAC_BIN = ROOT / ".lake" / "build" / "bin" / "veritac"

HELPER_PY = ROOT / "examples" / "cuda_attention_remote.py"
PROFILE_PY = ROOT / "Hardware" / "profile.py"

DEFAULT_HOST = "spark-379a.tail64c925.ts.net"
REMOTE_ROOT = "/home/cake/.cache/veritac-attention-run"
CANDIDATE_ROOT = REMOTE_ROOT + "/cuda_candidate"
REMOTE_PY = "/home/cake/.cache/veritac-attention-venv/bin/python"
SURVEY_PY = REMOTE_ROOT + "/cuda_survey.py"

DEFAULT_MODEL = lad.DEFAULT_MODEL
DEFAULT_BASE = lad.DEFAULT_BASE
hw = lad.hw

CUDA_LABEL = ("vendor-derived PyTorch/CUTLASS schedule specialization and native "
              "OpMultiplyAddFastF32 (3-component TF32 emulation), not IEEE scalar "
              "FP32 and not new precision reduction")

SEARCH_WARMUP = 10
SEARCH_SAMPLES = 20
SEARCH_INNER = 5
BACKENDS = ["auto", "efficient", "cudnn", "flash", "math"]

REQUIRED_CATALOG_FIELDS = [
    "queries_per_block", "keys_per_block", "max_k", "num_threads",
    "smem_bytes", "static_shared_bytes", "kernel_max_threads", "supported",
]
HEX = set("0123456789abcdefABCDEF")


# ---------------------------------------------------------------------------
# Remote client (SSH).  Tests substitute a fake client with canned responses.
# ---------------------------------------------------------------------------

class CudaRemoteClient:
    def __init__(self, host, helper_remote, survey_path=SURVEY_PY, remote_py=REMOTE_PY,
                 candidate_root=CANDIDATE_ROOT, log_dir=None, mem_cap_bytes=None):
        self.host = host
        self.helper_remote = helper_remote
        self.survey_path = survey_path
        self.remote_py = remote_py
        self.candidate_root = candidate_root
        self.log_dir = log_dir or (OUT / "remote_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.remote_uuid = uuid.uuid4().hex
        self.remote_out_dir = f"{candidate_root}/controller_runs/{self.remote_uuid}"
        self.mem_cap_bytes = mem_cap_bytes

    def _ssh(self, remote_argv, label):
        remote_str = shlex.join(remote_argv)
        proc = subprocess.run(["ssh", self.host, remote_str],
                              capture_output=True, text=True, timeout=3600)
        with open(self.log_dir / f"{label}-{uuid.uuid4().hex}.log", "w") as f:
            f.write(proc.stdout + "\n---STDERR---\n" + proc.stderr)
        if proc.returncode != 0:
            return {"error": f"ssh {label} failed: {proc.stderr[-1000:]}"}
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError as e:
            return {"error": f"invalid JSON from remote ({label}): {e}"}

    def _globals_before(self, sub):
        argv = [self.remote_py, self.helper_remote,
                "--candidate-root", self.candidate_root,
                "--survey", self.survey_path]
        if self.mem_cap_bytes is not None:
            argv += ["--mem-cap-bytes", str(self.mem_cap_bytes)]
        return argv + sub

    def push(self, local, remote):
        with open(local, "rb") as f:
            proc = subprocess.run(["ssh", self.host, "cat > " + shlex.quote(remote)],
                                  stdin=f, capture_output=True)
        if proc.returncode != 0:
            sys.exit(f"ABORT: could not push {local} to {remote}: {proc.stderr}")

    def mkdir(self, remote_dir):
        subprocess.run(["ssh", self.host, "mkdir -p " + shlex.quote(remote_dir)],
                       capture_output=True, check=True)

    def metadata(self):
        return self._ssh(self._globals_before(["metadata"]), "metadata")

    def benchmark(self, config_id, seq, dim, seed, warmup, samples, inner, out_basename):
        remote_out = f"{self.remote_out_dir}/{out_basename}"
        return self._ssh(self._globals_before([
            "benchmark", "--configs", str(config_id), "--seqs", str(seq),
            "--dims", str(dim), "--seed", str(seed), "--warmup", str(warmup),
            "--samples", str(samples), "--inner", str(inner), "--out", remote_out]),
            f"bench_{config_id}_{seq}_{dim}")

    def survey(self, seq, dim, seed, warmup, samples, inner, out_basename):
        remote_out = f"{self.remote_out_dir}/{out_basename}"
        return self._ssh(self._globals_before([
            "survey", "--seqs", str(seq), "--dims", str(dim), "--seed", str(seed),
            "--warmup", str(warmup), "--samples", str(samples), "--inner", str(inner),
            "--out", remote_out]), f"survey_{seq}_{dim}")


# ---------------------------------------------------------------------------
# Lean CLI
# ---------------------------------------------------------------------------

def call_check_cuda(target, catalog, plan, tactics):
    payload = {
        "mode": "check_cuda_attention_plan", "schema_version": 1,
        "target": target, "catalog": catalog,
        "initial_plan": plan, "tactics": tactics,
    }
    proc = subprocess.run([str(VERITAC_BIN), "--json", json.dumps(payload)],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()}
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON from CLI: {e}"}


def to_lean_catalog(catalogue):
    """Map raw config_info observations to the Lean catalogue schema WITHOUT
    coercing types — malformed observations pass through as-is (Lean's Nat/Bool
    parser rejects them)."""
    out = []
    for c in catalogue:
        missing = [f for f in REQUIRED_CATALOG_FIELDS + ["id"] if f not in c]
        if missing:
            raise ValueError(f"malformed catalogue entry {c.get('id')}: missing {missing}")
        out.append({
            "id": c["id"],
            "queries_per_block": c["queries_per_block"],
            "keys_per_block": c["keys_per_block"],
            "max_k": c["max_k"],
            "num_threads": c["num_threads"],
            "smem_bytes": c["smem_bytes"],
            "static_shared_bytes": c["static_shared_bytes"],
            "kernel_max_threads": c["kernel_max_threads"],
            "supported": c["supported"],
        })
    return out


def select_config_tactic(cfg_id):
    return {"kind": "select_config", "value": int(cfg_id)}


def check_plan(target, catalog, seq, dim, cfg_id):
    return call_check_cuda(target, catalog,
                           {"seq_len": seq, "head_dim": dim, "config_id": cfg_id}, [])


def first_legal_config(target, catalog, seq, dim, ids):
    for cid in ids:
        r = check_plan(target, catalog, seq, dim, cid)
        if r.get("accepted") is True:
            return int(cid)
    return None


def replay_sequence(target, catalog, initial_plan, sequence, expected_final_plan):
    """Replay a stored tactic sequence through Lean from its initial_plan and
    require the exact final_plan.  Returns True/False."""
    r = call_check_cuda(target, catalog, initial_plan, sequence)
    if r.get("accepted") is not True:
        return False
    return r.get("final_plan") == expected_final_plan


# ---------------------------------------------------------------------------
# LLM helpers (reused from the attention demo)
# ---------------------------------------------------------------------------

ensure_completions_url = lad.ensure_completions_url
load_env_file = lad.load_env_file
get_api_key = lad.get_api_key
llm_propose = lad.llm_propose
parse_llm_reply = lad.parse_llm_reply


def _finite_positive(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in HEX for c in s)


def _strict_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Metadata snapshot + drift + validation
# ---------------------------------------------------------------------------

def validate_binding(binding):
    """Require the exact mandatory binding fields, nonempty module/cu hashes,
    64-hex hash values, and module-path/hash correspondence.  Raises on any
    violation (no silent acceptance of an empty/malformed binding)."""
    if not isinstance(binding, dict):
        raise ValueError("binding metadata is not an object")
    for f in ("module_paths", "module_hashes", "cu_hashes", "vendor_hashes",
              "run_candidate_py", "installed_kernel_forward_h",
              "cutlass_git_head", "cutlass_dirty_diff_hash",
              "torch", "torch_cuda", "compute_capability"):
        if f not in binding:
            raise ValueError(f"binding missing mandatory field {f}")
    mp = binding["module_paths"]
    mh = binding["module_hashes"]
    cu = binding["cu_hashes"]
    vh = binding["vendor_hashes"]
    if not isinstance(mp, list) or not mp:
        raise ValueError("binding module_paths must be a nonempty list")
    if not isinstance(mh, dict) or not mh:
        raise ValueError("binding module_hashes must be a nonempty dict")
    if {os.path.basename(p) for p in mp} != set(mh.keys()):
        raise ValueError("binding module_paths do not correspond to module_hashes")
    for k, v in mh.items():
        if not _is_hex(v, 64):
            raise ValueError(f"binding module hash {k} is not 64-hex")
    if not isinstance(cu, dict) or not cu:
        raise ValueError("binding cu_hashes must be a nonempty dict")
    if not isinstance(vh, dict) or not vh:
        raise ValueError("binding vendor_hashes must be a nonempty dict")
    for key, d in (("cu_hashes", cu), ("vendor_hashes", vh)):
        for k, v in d.items():
            if not _is_hex(v, 64):
                raise ValueError(f"binding {key}[{k}] is not 64-hex")
    for f in ("run_candidate_py", "installed_kernel_forward_h",
              "cutlass_dirty_diff_hash"):
        if not _is_hex(binding[f], 64):
            raise ValueError(f"binding {f} is not 64-hex")
    if not isinstance(binding["cutlass_git_head"], str) or not binding["cutlass_git_head"]:
        raise ValueError("binding cutlass_git_head is empty")
    for f in ("torch", "torch_cuda", "compute_capability"):
        if not isinstance(binding[f], str) or not binding[f]:
            raise ValueError(f"binding {f} is empty")


def snapshot_metadata(remote, run_dir):
    meta = remote.metadata()
    if "error" in meta:
        sys.exit(f"ABORT: metadata fetch failed: {meta['error']}")
    profile = meta["profile"]
    hw.validate_schema(profile)
    profile_hash = hw.profile_hash(profile)
    catalogue = meta["catalogue"]
    for c in catalogue:
        if not _strict_int(c.get("id")):
            sys.exit(f"ABORT: catalogue entry has non-strict-int id: {c.get('id')!r}")
    ids = [c["id"] for c in catalogue]
    if len(ids) != len(set(ids)):
        sys.exit("ABORT: catalogue has duplicate config ids")
    config_ids = meta.get("config_ids", [])
    if not all(_strict_int(i) for i in config_ids):
        sys.exit("ABORT: config_ids are not strict integers")
    if len(config_ids) != len(ids) or set(config_ids) != set(ids):
        sys.exit("ABORT: config_ids do not match catalogue ids")
    try:
        lean_catalog = to_lean_catalog(catalogue)
    except ValueError as e:
        sys.exit(f"ABORT: malformed catalogue observation: {e}")
    local_hash = hashlib.sha256(
        json.dumps(catalogue, sort_keys=True, default=str).encode()).hexdigest()
    if local_hash != meta.get("catalogue_hash"):
        sys.exit("ABORT: catalogue hash mismatch (local recompute differs from remote)")
    binding = meta.get("binding")
    try:
        validate_binding(binding)
    except ValueError as e:
        sys.exit(f"ABORT: {e}")
    return {
        "profile": profile,
        "profile_hash": profile_hash,
        "catalogue": catalogue,
        "catalogue_hash": meta.get("catalogue_hash"),
        "binding": binding,
        "binary_path": meta.get("binary_path"),
        "module_paths": meta.get("module_paths"),
        "lean_catalog": lean_catalog,
        "target": profile,
        "config_ids": config_ids,
    }


def check_binding(remote, snap, label):
    meta = remote.metadata()
    if "error" in meta:
        sys.exit(f"ABORT: {label} metadata re-fetch failed: {meta['error']}")
    if meta.get("profile") != snap["profile"]:
        sys.exit(f"ABORT: {label} profile changed (drift detected)")
    if meta.get("catalogue") != snap["catalogue"]:
        sys.exit(f"ABORT: {label} catalogue changed (drift detected)")
    if meta.get("binding") != snap["binding"]:
        sys.exit(f"ABORT: {label} binary/source binding changed (drift detected)")
    print(f"  {label}: binding stable (catalogue hash {str(snap['catalogue_hash'])[:12]}…)")


# ---------------------------------------------------------------------------
# Verified dispatch (used by EVERY benchmark, including the final replay)
# ---------------------------------------------------------------------------

def candidate_ok(case):
    comp = case.get("comparison") or {}
    gcomp = case.get("graph_comparison") or {}
    if not comp.get("within_tolerance") or not gcomp.get("within_tolerance"):
        return False, "not_within_tolerance"
    evt = (case.get("cuda_event_timing") or {}).get("median_ms")
    gr = (case.get("graph_timing") or {}).get("median_ms")
    if not _finite_positive(evt) or not _finite_positive(gr):
        return False, "invalid_gpu_timing"
    return True, None


def executable_policy(snap_entry):
    """Controller validation policy (based on observed cp.async racecheck warning
    failures), separate from the Lean resource checker: an entry is executable
    for this FP32 demo only when `async_drains is True` (explicit compiled drain
    contract) and `preload_v is False`.  Undrained normal/reverse variants are
    Lean-legal planning states but are never dispatched."""
    if snap_entry is None:
        return False, "config not in snapshot catalogue"
    drains = snap_entry.get("async_drains")
    preload = snap_entry.get("preload_v")
    if not (drains is True and preload is False):
        return False, (f"not executable for FP32 demo: async_drains={drains!r}, "
                       f"preload_v={preload!r} (undrained/preload variant)")
    return True, None


def verified_dispatch(remote, args, run_dir, snap, cfg_id, checked_plan):
    """The single VERIFIED dispatch used for every benchmark, including the
    final replay.  Requires: checked_plan config_id == requested, checked shape,
    the frozen catalogue entry passes the controller executable-validation
    policy (async_drains=True, preload_v=False), pre/post binding stable, actual
    case shape/id matches, every raw config_info field equals the frozen
    snapshot entry, and candidate_ok."""
    if int(checked_plan["config_id"]) != int(cfg_id):
        return {"config_id": int(cfg_id), "checked_plan": checked_plan,
                "status": "checkedplan_id_mismatch"}
    if (checked_plan.get("seq_len"), checked_plan.get("head_dim")) != (args.seq, args.dim):
        return {"config_id": int(cfg_id), "checked_plan": checked_plan,
                "status": "checkedplan_shape_mismatch"}
    snap_entry = next((c for c in snap["catalogue"] if c.get("id") == cfg_id), None)
    ok_exec, reason = executable_policy(snap_entry)
    if not ok_exec:
        return {"config_id": int(cfg_id), "checked_plan": checked_plan,
                "status": "validation_rejected", "reason": reason}
    check_binding(remote, snap, "pre-dispatch")
    out_basename = f"cand_{cfg_id}_{args.seq}_{args.dim}_{uuid.uuid4().hex}.json"
    case = remote.benchmark(int(cfg_id), args.seq, args.dim, args.seed,
                            args.warmup, args.samples, args.inner, out_basename)
    check_binding(remote, snap, "post-dispatch")
    rec = {"config_id": int(cfg_id), "checked_plan": checked_plan, "case": case}
    if "error" in case:
        rec["status"] = "error"
        return rec
    rec["matched"] = bool(case.get("config") == cfg_id
                          and case.get("seq") == args.seq
                          and case.get("dim") == args.dim)
    if not rec["matched"]:
        rec["status"] = "mismatch"
        return rec
    ci = case.get("config_info")
    # Bind every compiler and policy fact, including the drain/preload contract.
    if ci is None or snap_entry is None or ci != snap_entry:
        rec["status"] = "config_info_mismatch"
        return rec
    ok, why = candidate_ok(case)
    rec["status"] = ("ok" if ok else
                     ("numeric_failed" if why == "not_within_tolerance" else "invalid_timing"))
    rec["graph_median_ms"] = (case.get("graph_timing") or {}).get("median_ms")
    rec["event_median_ms"] = (case.get("cuda_event_timing") or {}).get("median_ms")
    rec["input_hashes"] = case.get("input_hashes")
    return rec


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search(remote, args, run_dir, snap, seq, dim, seed, warmup, samples, inner):
    records = []
    initial_id = first_legal_config(snap["target"], snap["lean_catalog"],
                                    seq, dim, snap["config_ids"])
    if initial_id is None:
        sys.exit("ABORT: no legal config in catalogue")
    initial_plan = {"seq_len": seq, "head_dim": dim, "config_id": initial_id}
    for cid in snap["config_ids"]:
        tactic = select_config_tactic(cid)
        r = call_check_cuda(snap["target"], snap["lean_catalog"], initial_plan, [tactic])
        record = {"config_id": cid, "initial_plan": initial_plan,
                  "tactic_sequence": [tactic], "verification_response": r}
        if "error" in r:
            record["status"] = "error"
            record["error"] = r["error"]
            records.append(record)
            continue
        if r.get("accepted") is not True:
            record["status"] = "rejected"
            record["diagnostic"] = r.get("diagnostic", "rejected")
            records.append(record)
            continue
        final_plan = r["final_plan"]
        record["checked_plan"] = final_plan
        if final_plan.get("supported") is not True:
            record["status"] = "unsupported"
            records.append(record)
            continue
        rec = verified_dispatch(remote, args, run_dir, snap, cid, final_plan)
        rec.update({"initial_plan": initial_plan, "tactic_sequence": [tactic],
                    "verification_response": r})
        records.append(rec)
        print(f"    cfg{cid}: status={rec.get('status')} "
              f"graph={rec.get('graph_median_ms')} ms")
    good = [r for r in records if r.get("status") == "ok"
            and _finite_positive(r.get("graph_median_ms"))]
    if not good:
        return records, None
    winner = min(good, key=lambda r: r["graph_median_ms"])
    return records, winner


# ---------------------------------------------------------------------------
# Final replay + survey comparison
# ---------------------------------------------------------------------------

def median_ms(case, key):
    return (case.get(key) or {}).get("median_ms")


def compare_with_survey(remote, args, run_dir, snap, winner_case, seq, dim):
    """Fresh survey comparison.  A vendor backend is eligible for a metric only
    if its ORIGINAL raw status == 'ok' AND its eager comparison is within
    tolerance (event) / graph comparison is within tolerance (graph), AND the
    relevant timing is finite positive.  Failures are never reclassified as
    wins; raw status is preserved in the report."""
    surv = remote.survey(seq, dim, args.seed, args.warmup, args.samples,
                         args.inner, f"survey_{seq}_{dim}.json")
    if "error" in surv:
        sys.exit(f"ABORT: survey failed: {surv['error']}")
    for field in ("q", "k", "v"):
        ch = (winner_case.get("input_hashes") or {}).get(field)
        sh = (surv.get("input_hashes") or {}).get(field)
        if not ch or ch != sh:
            sys.exit(f"ABORT: input hash mismatch for {field} "
                     f"between candidate and survey ({ch} vs {sh}).")

    cand_cfg = winner_case.get("config")
    cand_evt = median_ms(winner_case, "cuda_event_timing")
    cand_gr = median_ms(winner_case, "graph_timing")

    per_backend = {}
    for b in BACKENDS:
        rec = (surv.get("results") or {}).get(b) or {}
        raw_status = rec.get("status")
        comp = rec.get("comparison") or {}
        gcomp = rec.get("graph_comparison") or {}
        evt = median_ms(rec, "cuda_event_timing")
        gr = median_ms(rec, "graph_timing")
        evt_eligible = (raw_status == "ok" and comp.get("within_tolerance") is True
                        and _finite_positive(evt))
        gr_eligible = (raw_status == "ok" and comp.get("within_tolerance") is True
                       and gcomp.get("within_tolerance") is True
                       and _finite_positive(gr))
        per_backend[b] = {
            "raw_status": raw_status,
            "event_eligible": evt_eligible,
            "graph_eligible": gr_eligible,
            "event_median_ms": evt,
            "graph_median_ms": gr,
            "comparison": comp,
            "graph_comparison": gcomp,
            "raw": rec,
        }
        print(f"  vendor {b}: raw_status={raw_status} event_eligible={evt_eligible} "
              f"graph_eligible={gr_eligible} event={evt} graph={gr} ms")

    evt_ok = {b: d["event_median_ms"] for b, d in per_backend.items()
              if d["event_eligible"]}
    gr_ok = {b: d["graph_median_ms"] for b, d in per_backend.items()
             if d["graph_eligible"]}

    report = {
        "candidate_config_id": cand_cfg,
        "candidate_event_median_ms": cand_evt,
        "candidate_graph_median_ms": cand_gr,
        "per_backend": per_backend,
        "input_hashes_verified": True,
        "event_vs_event": {b: (cand_evt, per_backend[b]["event_median_ms"])
                           for b in BACKENDS},
        "graph_vs_graph": {b: (cand_gr, per_backend[b]["graph_median_ms"])
                           for b in BACKENDS},
        "raw_candidate": winner_case,
        "raw_survey": surv,
    }
    if evt_ok:
        fe = min(evt_ok, key=evt_ok.get)
        report["fastest_vendor_event"] = fe
        report["fastest_vendor_event_median_ms"] = evt_ok[fe]
        if cand_evt is not None:
            report["candidate_over_fastest_event_ratio"] = cand_evt / evt_ok[fe]
        print(f"  fastest vendor (event): {fe} ({evt_ok[fe]} ms)")
    if gr_ok:
        fg = min(gr_ok, key=gr_ok.get)
        report["fastest_vendor_graph"] = fg
        report["fastest_vendor_graph_median_ms"] = gr_ok[fg]
        if cand_gr is not None:
            report["candidate_over_fastest_graph_ratio"] = cand_gr / gr_ok[fg]
        print(f"  fastest vendor (graph): {fg} ({gr_ok[fg]} ms)")
    return report


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

def catalogue_facts(snap):
    lines = []
    for c in snap["catalogue"]:
        lines.append(
            f"  cfg {c.get('id')}: Q={c.get('queries_per_block')} "
            f"K={c.get('keys_per_block')} max_k={c.get('max_k')} "
            f"threads={c.get('num_threads')} smem={c.get('smem_bytes')} "
            f"static={c.get('static_shared_bytes')} "
            f"kernel_max_threads={c.get('kernel_max_threads')} "
            f"min_blocks_hint={c.get('min_blocks_hint')} "
            f"regs={c.get('num_regs')} supported={c.get('supported')} "
            f"module={c.get('module')} alias_scratch={c.get('alias_scratch')} "
            f"reversed={c.get('reversed')} async_drains={c.get('async_drains')} "
            f"preload_v={c.get('preload_v')}")
    return "\n".join(lines)


def build_llm_user_message(snap, seq, dim, current_plan, last_feedback):
    return (
        hw.render_prompt(snap["profile"]) + "\n\n"
        "Compiled catalogue (observed resource facts):\n"
        + catalogue_facts(snap) + "\n\n"
        "Execution policy: choose a supported configuration with async_drains=True "
        "and preload_v=False; undrained variants are planning/diagnostic only.\n"
        f"shape: seq={seq} dim={dim} B1 H8\n"
        f"current accepted plan (JSON):\n{json.dumps(current_plan, indent=2)}\n\n"
        f"Last Lean feedback: {last_feedback}\n\n"
        "Propose the next single select_config (or done):")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="fixed schedule, no API key")
    ap.add_argument("--search", action="store_true",
                    help="enumerate catalogue configs, Lean-check from a legal "
                         "initial, benchmark correct ones, pick fastest, then fresh replay")
    ap.add_argument("--config-id", type=int, default=None,
                    help="explicit config id choice (default mock preset is a "
                         "confirmed-clean candidate, id 25 short / 26 long; "
                         "LLM starts from the first Lean-legal planning state)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--warmup", type=int, default=SEARCH_WARMUP)
    ap.add_argument("--samples", type=int, default=SEARCH_SAMPLES)
    ap.add_argument("--inner", type=int, default=SEARCH_INNER)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mem-cap-bytes", type=int, default=20 * (2 ** 30),
                    help="cuda_survey reference memory cap in bytes (default 20 GiB; "
                         "use e.g. 32 GiB = 34359738368 for N8192 D256 on a 128 GB GB10)")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    if args.seq < 1 or args.dim not in (192, 256):
        ap.error("--seq >= 1 and --dim in {192, 256}")
    if args.warmup < 0 or args.samples < 1 or args.inner < 1:
        ap.error("--warmup >= 0, --samples >= 1, --inner >= 1")
    if args.mem_cap_bytes < 1:
        ap.error("--mem-cap-bytes must be positive")

    print("╭─ VeriTac · CUDA attention demo (local controller) ─" + "─" * 22 + "╮")
    print("│  every select_config tactic is verified by the local Lean verifier.")
    print("╰" + "─" * 60 + "╯")
    print("")
    print("target host:", args.host)
    print(f"shape: seq={args.seq} dim={args.dim} B1 H8")
    print(f"mode: {'mock' if args.mock else 'LLM agent'}"
          f"{' + search' if args.search else ''}")
    print("")

    env = load_env_file()
    args.model = (args.model or os.environ.get("OPENAI_MODEL")
                  or env.get("OPENAI_MODEL") or DEFAULT_MODEL)
    base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                or env.get("OPENAI_BASE_URL") or DEFAULT_BASE)
    base_url = ensure_completions_url(base_url)
    key = None
    if not args.mock:
        key, _ = get_api_key(env)
        if not key:
            print("No API key — running in --mock mode instead.")
            args.mock = True

    run_dir = OUT / ("run_" + uuid.uuid4().hex)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"outputs: {run_dir}")

    helper_remote = f"{CANDIDATE_ROOT}/controller_runs/{uuid.uuid4().hex}/cuda_attention_remote.py"
    remote = CudaRemoteClient(args.host, helper_remote, log_dir=run_dir / "remote_logs",
                              mem_cap_bytes=args.mem_cap_bytes)
    remote.mkdir(os.path.dirname(helper_remote))
    remote.push(str(HELPER_PY), helper_remote)
    remote.push(str(PROFILE_PY), os.path.join(os.path.dirname(helper_remote), "profile.py"))

    snap = snapshot_metadata(remote, run_dir)
    print(f"hardware: {snap['profile'].get('device_name')} "
          f"backend={snap['profile'].get('backend')} "
          f"warp={snap['profile'].get('simd_width')} "
          f"threads={snap['profile'].get('max_threads_per_threadgroup')} "
          f"shared_limit={snap['profile'].get('max_threadgroup_memory_bytes')}B")
    print(f"profile hash: {snap['profile_hash'][:12]}… "
          f"catalogue: {len(snap['config_ids'])} configs "
          f"(hash {str(snap['catalogue_hash'])[:12]}…)")

    legal_first = first_legal_config(snap["target"], snap["lean_catalog"],
                                     args.seq, args.dim, snap["config_ids"])
    if legal_first is None:
        sys.exit("ABORT: no legal config in the compiled catalogue.")

    report = {
        "mode": "check_cuda_attention_plan",
        "seed": args.seed, "seq": args.seq, "dim": args.dim,
        "search": args.search,
        "hardware": {"device_name": snap["profile"].get("device_name"),
                     "profile_hash": snap["profile_hash"]},
        "label": CUDA_LABEL,
        "snapshot": {
            "profile": snap["profile"],
            "profile_hash": snap["profile_hash"],
            "catalogue": snap["catalogue"],
            "catalogue_hash": snap["catalogue_hash"],
            "binding": snap["binding"],
            "module_paths": snap["module_paths"],
        },
        "agent_initial_plan": {"seq_len": args.seq, "head_dim": args.dim,
                               "config_id": legal_first},
        "agent_tactic_sequence": [],
        "agent_verification_response": None,
        "agent_final_plan": None,
        "final_plan": None,
        "candidates": [],
        "benchmark": None,
    }

    # ---- agent (mock or LLM): start from the legal initial ----
    initial_plan = dict(report["agent_initial_plan"])
    # Demonstrate the measured specialization in mock mode; the catalogue and
    # Lean still decide whether the preset is available and legal on this host.
    preset = 25 if args.seq <= 1024 else 26
    default_choice = preset if args.mock and preset in snap["config_ids"] else legal_first
    chosen = args.config_id if args.config_id is not None else default_choice
    tactic = select_config_tactic(chosen)
    r = call_check_cuda(snap["target"], snap["lean_catalog"], initial_plan, [tactic])
    if "error" in r:
        sys.exit(f"ABORT: Lean check failed: {r['error']}")
    if r.get("accepted") is not True:
        sys.exit(f"ABORT: config {chosen} not accepted: {r.get('diagnostic')}")
    agent_final_plan = r["final_plan"]
    agent_sequence = [tactic]
    agent_response = r
    report["agent_tactic_sequence"] = list(agent_sequence)
    report["agent_verification_response"] = agent_response

    if not args.mock:
        # LLM: one select_config per turn, with profile/resources + feedback.
        messages = [{"role": "system", "content": (
            "You are the CUDA attention config-selection agent for VeriTac. "
            "Every select_config is machine-checked by Lean against the compiled "
            "catalogue; only accepted configs are benchmarked. Propose ONE tactic "
            'per turn: {"kind":"select_config","value":<config_id>} or '
            '{"kind":"done"}. Resource facts come from the compiled catalogue '
            "only; never invent memory/thread numbers.")}]
        last_feedback = "start"
        current_plan = agent_final_plan
        for _turn in range(args.max_steps):
            user = build_llm_user_message(snap, args.seq, args.dim,
                                          current_plan, last_feedback)
            messages = messages[:1] + [{"role": "user", "content": user}]
            reply = None
            prop = None
            for _a in range(3):
                try:
                    reply = llm_propose(key, base_url, args.model, messages)
                except urllib.error.HTTPError as e:
                    sys.exit(f"ABORT: LLM API error HTTP {e.code}: {e.read().decode()[:300]}")
                except Exception as e:  # noqa: BLE001
                    sys.exit(f"ABORT: LLM API error: {type(e).__name__}: {e}")
                prop = parse_llm_reply(reply)
                if prop is not None:
                    break
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Reply with ONLY one tactic JSON."})
            if prop is None:
                break
            if prop.get("kind") == "done":
                break
            if prop.get("kind") != "select_config" or "value" not in prop:
                last_feedback = "REJECTED: unknown tactic; select_config only."
                messages.append({"role": "assistant", "content": json.dumps(prop)})
                messages.append({"role": "user", "content": last_feedback})
                continue
            res = call_check_cuda(snap["target"], snap["lean_catalog"], current_plan, [prop])
            if res.get("accepted") is not True:
                last_feedback = f"REJECTED: {res.get('diagnostic')}"
                messages.append({"role": "assistant", "content": json.dumps(prop)})
                messages.append({"role": "user", "content": last_feedback})
                continue
            current_plan = res["final_plan"]
            agent_final_plan = current_plan
            agent_sequence.append(prop)
            agent_response = res
            report["agent_tactic_sequence"] = list(agent_sequence)
            report["agent_verification_response"] = agent_response
            last_feedback = f"OK (accepted {json.dumps(prop)})"
    report["agent_final_plan"] = agent_final_plan
    print(f"  agent final plan: config_id={agent_final_plan['config_id']}")

    # Replay the retained agent sequence through Lean (always).
    if not replay_sequence(snap["target"], snap["lean_catalog"], initial_plan,
                           agent_sequence, agent_final_plan):
        sys.exit("ABORT: agent sequence does not replay to its final_plan")

    if args.no_benchmark:
        print("\n--no-benchmark: metadata + Lean only; no remote benchmark.")
        report["final_plan"] = agent_final_plan
        _write_report(run_dir, report)
        return

    # ---- benchmark stage ----
    if args.search:
        print("\n── search ─" + "─" * 52)
        records, winner = run_search(remote, args, run_dir, snap,
                                     args.seq, args.dim, args.seed,
                                     args.warmup, args.samples, args.inner)
        report["candidates"] = records
        if winner is None:
            _write_report(run_dir, report)
            sys.exit("ABORT: no correct (ok) executable candidate to benchmark.")
        dispatch = winner
        print(f"  [search] winner cfg{dispatch['checked_plan']['config_id']} "
              f"graph={dispatch['graph_median_ms']} ms")
        report["winner"] = {k: dispatch.get(k) for k in
                            ("config_id", "status", "graph_median_ms",
                             "event_median_ms", "checked_plan",
                             "initial_plan", "tactic_sequence",
                             "verification_response")}
        report["selection_timing_ms"] = dispatch["graph_median_ms"]
    else:
        if agent_final_plan.get("supported") is not True:
            sys.exit("ABORT: agent final plan config is unsupported; never dispatch.")
        dispatch = {"config_id": int(agent_final_plan["config_id"]),
                    "checked_plan": agent_final_plan,
                    "initial_plan": initial_plan,
                    "tactic_sequence": agent_sequence,
                    "verification_response": agent_response}

    # Replay the dispatched config's complete stored sequence through Lean from
    # its own initial_plan and require the exact final_plan before dispatching.
    if not replay_sequence(snap["target"], snap["lean_catalog"],
                           dispatch["initial_plan"], dispatch["tactic_sequence"],
                           dispatch["checked_plan"]):
        sys.exit("ABORT: dispatched config trace does not replay to its checked_plan")
    report["final_plan"] = dispatch["checked_plan"]

    # Final replay via the SAME verified dispatch (never bypasses it).
    final_rec = verified_dispatch(remote, args, run_dir, snap,
                                  dispatch["config_id"], dispatch["checked_plan"])
    report["final_replay"] = final_rec
    if final_rec["status"] != "ok":
        _write_report(run_dir, report)
        sys.exit(f"ABORT: final replay not ok (status={final_rec['status']}); "
                 "no win is claimed.")
    print(f"  final replay cfg{dispatch['config_id']}: "
          f"event={final_rec['event_median_ms']} graph={final_rec['graph_median_ms']} ms")
    report["benchmark"] = compare_with_survey(remote, args, run_dir, snap,
                                              final_rec["case"], args.seq, args.dim)
    _write_report(run_dir, report)


def _write_report(run_dir, report):
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
