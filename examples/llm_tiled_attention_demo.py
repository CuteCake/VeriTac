"""
VeriTac LLM-driven Metal tiled-attention *plan* demo.

Probes the local Metal device into a validated `Hardware.profile`, then has an
LLM (or a fixed mock) propose refinements to a tiling *plan* — `reuse_kv_storage`,
`set_query_tile`, `set_key_tile` — one tactic at a time.  Every proposal is
machine-checked by the Lean CLI (`check_attention_plan`); the plan changes only
when accepted, and only the Lean `final_plan` is authoritative.

`aliasKV = false` is a resource-legal *planning* state that stages `Q + K + V`
separately.  The installed vendor shader always aliases K and V, so only an
`aliasKV = true` plan is executable; this demo NEVER dispatches a
non-executable plan to the benchmark.

`--search` enumerates Q8/16/24 x K8/16, checks each candidate through Lean
(retaining its authoritative `checked_plan`, `tactic_sequence` and per-tactic
`verification_trace`), benchmarks every accepted executable candidate, and picks
the fastest *correct* (`ok`, finite positive timing) one.  The final report then
refers to that winner (plan, executable, tactic trace), the agent's original plan
is kept separately as `agent_final_plan`, and the winner is re-probed, re-checked
and freshly replayed for the reported comparison.

The benchmark stage shells out to the vendor-derived MLX steel attention
specialization (`benchmarks/attention/partitioned/steel_attention.py`) at the
exact accepted (query_tile, key_tile) for the requested seq/dim, and to the
vendor survey (`benchmarks/attention/metal_survey.py`) with the `mlx_fast` and
`mpsgraph` backends at float32.  Input q/k/v hashes must match across runs.

The OpenAI API key, model and an optional OpenAI-compatible base URL are read
from `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL`, the root `.env`, or
entered interactively (stdlib `urllib` only).  `--mock` never asks for a key.

Usage:
    python3 examples/llm_tiled_attention_demo.py --mock --no-benchmark   # smoke
    python3 examples/llm_tiled_attention_demo.py --mock                  # full benchmark
    python3 examples/llm_tiled_attention_demo.py --mock --search         # sweep + fastest
    OPENAI_API_KEY=sk-... python3 examples/llm_tiled_attention_demo.py --no-benchmark

Requires `lake build veritac` (Lean CLI).  Outputs are written under
`.lake/tiled_attention_demo/<run>`.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import llm_attention_demo as lad  # noqa: E402  (reuse LLM client + hardware helpers)

OUT = ROOT / ".lake" / "tiled_attention_demo"
VERITAC_BIN = ROOT / ".lake" / "build" / "bin" / "veritac"
STEEL_PY = ROOT / "benchmarks" / "attention" / "partitioned" / "steel_attention.py"
SURVEY_PY = ROOT / "benchmarks" / "attention" / "metal_survey.py"

DEFAULT_MODEL = lad.DEFAULT_MODEL
DEFAULT_BASE = lad.DEFAULT_BASE
FP32_BYTES = 4

# Mock schedule: start from an alias_kv=false planning baseline, reuse K/V
# storage to become executable, then widen the query tile.  "done" stops.
MOCK_SEQUENCE = [
    {"kind": "reuse_kv_storage"},
    {"kind": "set_query_tile", "value": 16},
]

SEARCH_QTS = (8, 16, 24)
SEARCH_KTS = (8, 16)

PLAN_LEGAL = (
    "query_tile 8/16/24/32 (one SIMD-32 group per 8 rows, threads = "
    "(query_tile/8)*32); key_tile 8/16/32; head_dim 192 or 256; seq_len > 0. "
    "alias_kv may be false (resource-legal planning state staging Q+K+V) but "
    "only alias_kv=true is executable by the vendor shader."
)


# ---------------------------------------------------------------------------
# Resource accounting (must mirror Plan.lean)
# ---------------------------------------------------------------------------

def q_b(qt, hd):
    return qt * (hd + 4) * 4


def k_b(kt, hd):
    return (kt + 4) * hd * 4


def v_b(kt, hd):
    return kt * (hd + 4) * 4


def plan_shared_bytes(qt, kt, hd, alias):
    return q_b(qt, hd) + (max(k_b(kt, hd), v_b(kt, hd)) if alias else k_b(kt, hd) + v_b(kt, hd))


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

def probe_metal() -> tuple[dict, str]:
    prof = lad.hw.probe(backend="metal")
    if prof is None:
        sys.exit("ERROR: could not probe a Metal hardware profile.")
    lad.hw.validate_schema(prof)
    h = lad.hw.profile_hash(prof)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hardware.profile.json", "w") as f:
        f.write(lad.hw.canonical_json(prof) + "\n")
    return prof, h


def initial_plan(seq_len: int, dim: int) -> dict:
    # Resource-legal, NON-executable planning baseline: Q8K8 alias_kv=false.
    return {
        "seq_len": seq_len,
        "head_dim": dim,
        "query_tile": 8,
        "key_tile": 8,
        "alias_kv": False,
    }


# ---------------------------------------------------------------------------
# Lean CLI
# ---------------------------------------------------------------------------

def call_check_plan(target: dict, plan: dict, tactics: list[dict]) -> dict:
    if not VERITAC_BIN.exists():
        sys.exit(f"ERROR: veritac binary not found at {VERITAC_BIN}. Run 'lake build veritac'.")
    payload = {
        "mode": "check_attention_plan",
        "schema_version": 1,
        "target": target,
        "initial_plan": plan,
        "tactics": tactics,
    }
    proc = subprocess.run(
        [str(VERITAC_BIN), "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()}
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON from CLI: {e}"}


def check_sequence(profile: dict, plan: dict, tactics: list[dict]) -> tuple[bool, dict, list[dict]]:
    """Apply tactics one at a time through Lean, recording every accepted
    intermediate plan.  Returns (all_ok, final_plan, trace_of_plans)."""
    current = plan
    trace: list[dict] = []
    for tac in tactics:
        res = call_check_plan(profile, current, [tac])
        if "error" in res:
            return False, current, trace
        if res.get("accepted") is not True:
            return False, current, trace
        fl = res.get("final_plan")
        if not isinstance(fl, dict):
            return False, current, trace
        current = fl
        trace.append({"tactic": tac, "final_plan": fl,
                      "executable": bool(res.get("executable")),
                      "shared_memory_bytes": res.get("shared_memory_bytes"),
                      "threads_per_threadgroup": res.get("threads_per_threadgroup")})
    return True, current, trace


def replay_plan(profile: dict, initial: dict, sequence: list[dict], expected_plan: dict) -> bool:
    """Replay a stored tactic sequence through Lean and check it reproduces the
    authoritative checked_plan exactly.  Every stored candidate trace must
    replay to the identical plan before it may be dispatched."""
    res = call_check_plan(profile, initial, sequence)
    if "error" in res or res.get("accepted") is not True:
        return False
    fl = res.get("final_plan")
    return isinstance(fl, dict) and fl == expected_plan


# ---------------------------------------------------------------------------
# LLM helpers (reused from the attention demo)
# ---------------------------------------------------------------------------

def ensure_completions_url(base: str) -> str:
    return lad.ensure_completions_url(base)


def load_env_file() -> dict:
    return lad.load_env_file()


def get_api_key(env: dict) -> tuple[str | None, str]:
    return lad.get_api_key(env)


def llm_propose(key: str, base_url: str, model: str, messages: list[dict]) -> str:
    return lad.llm_propose(key, base_url, model, messages)


def parse_llm_reply(reply: str) -> dict | None:
    return lad.parse_llm_reply(reply)


def mock_propose(step: int) -> dict | None:
    if step < len(MOCK_SEQUENCE):
        return MOCK_SEQUENCE[step]
    return {"kind": "done"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def make_system_prompt(profile: dict) -> str:
    fragment = lad.hw.render_prompt(profile)
    return (
        "You are the scheduling agent for VeriTac's verified Metal tiled-"
        "attention planner. Every tactic you propose is machine-checked by a "
        "Lean verifier; if it is invalid the verifier rejects it and you will "
        "see the error. Pick ONE tactic per turn.\n\n"
        f"{fragment}\n"
        "Shared-memory bytes: alias_kv=false -> Q + K + V "
        "(Q=qt*(hd+4)*4, K=(kt+4)*hd*4, V=kt*(hd+4)*4); alias_kv=true -> "
        "Q + max(K, V) (vendor shader aliases K/V).\n"
        f"Legal plan clauses:\n- {PLAN_LEGAL}\n"
        "Available tactic JSON (propose exactly one):\n"
        '- {"kind":"reuse_kv_storage"}\n'
        '- {"kind":"set_query_tile","value":<8|16|24|32>}\n'
        '- {"kind":"set_key_tile","value":<8|16|32>}\n'
        "When done improving, reply exactly: {\"kind\":\"done\"}.\n\n"
        "Reply with ONLY the JSON. No commentary."
    )


def build_user_message(plan: dict, last_result: str) -> str:
    alias = plan.get("alias_kv")
    return (
        "Current plan (JSON):\n"
        f"{json.dumps(plan, indent=2)}\n\n"
        f"Shared bytes: {plan_shared_bytes(plan['query_tile'], plan['key_tile'], plan['head_dim'], alias)}\n"
        f"Executable: {bool(alias)}\n\n"
        f"Last verifier feedback: {last_result}\n\n"
        "Propose the next single tactic (or done):"
    )


def run_agent(args, profile: dict, key: str | None, base_url: str) -> tuple[dict, list[dict], list[str]]:
    plan = initial_plan(args.seq, args.dim)
    trace: list[dict] = []
    loglines: list[str] = []
    last_result = "start"

    def log(line: str):
        loglines.append(line)
        print(line)

    messages: list[dict] | None = None
    if not args.mock:
        messages = [{"role": "system", "content": make_system_prompt(profile)}]

    for step in range(1, args.max_steps + 1):
        log("")
        log(f"── step {step}/{args.max_steps} ─" + "─" * 40)

        if args.mock:
            propose = mock_propose(step - 1)
            log(f"  llm (mock): {json.dumps(propose)}")
        else:
            user = build_user_message(plan, last_result)
            messages = messages[:1] + [{"role": "user", "content": user}]
            propose = None
            for _attempt in range(3):
                try:
                    reply = llm_propose(key, base_url, args.model, messages)
                except urllib.error.HTTPError as e:
                    body = e.read().decode()[:300]
                    log(f"  api error: HTTP {e.code}: {body}")
                    return plan, trace, loglines
                except Exception as e:  # noqa: BLE001
                    log(f"  api error: {type(e).__name__}: {e}")
                    return plan, trace, loglines
                propose = parse_llm_reply(reply)
                if propose is not None:
                    break
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user",
                                 "content": "Reply with ONLY one tactic JSON or "
                                            '{"kind":"done"}.'})
            if propose is None:
                log("  llm: <unparseable reply after retries>")
                break
            log(f"  llm: {json.dumps(propose)}")

        if not propose or propose.get("kind") == "done":
            log("  llm: done")
            break

        result = call_check_plan(profile, plan, [propose])
        if "error" in result:
            last_result = f"REJECTED: {result['error']}"
            log(f"  lean: ✗ {result['error']}")
            if not args.mock and messages is not None:
                messages.append({"role": "assistant", "content": json.dumps(propose)})
                messages.append({"role": "user",
                                 "content": f"The verifier rejected {json.dumps(propose)}: "
                                            f"{result['error']}. Propose something else "
                                            "or done."})
            continue
        if result.get("accepted") is not True:
            diag = result.get("diagnostic", "rejected")
            last_result = f"REJECTED: {diag}"
            log(f"  lean: ✗ {diag}")
            if not args.mock and messages is not None:
                messages.append({"role": "assistant", "content": json.dumps(propose)})
                messages.append({"role": "user",
                                 "content": f"The verifier rejected {json.dumps(propose)}: "
                                            f"{diag}. Propose something else or done."})
            continue

        fl = result.get("final_plan")
        if not isinstance(fl, dict):
            sys.exit(f"ABORT: accepted response missing final_plan: {result!r}")
        plan = fl
        trace.append({"tactic": propose, "final_plan": fl,
                      "executable": bool(result.get("executable")),
                      "shared_memory_bytes": result.get("shared_memory_bytes"),
                      "threads_per_threadgroup": result.get("threads_per_threadgroup")})
        last_result = f"OK (accepted {json.dumps(propose)})"
        log(f"  lean: ✓ accepted {json.dumps(propose)}")
        alias = plan.get("alias_kv")
        log(f"  plan: q={plan['query_tile']} k={plan['key_tile']} "
            f"threads={plan['threads_per_threadgroup']} "
            f"shared={plan_shared_bytes(plan['query_tile'], plan['key_tile'], plan['head_dim'], alias)}B "
            f"executable={bool(alias)}")

    log("")
    log("══ agent finished ═" + "═" * 42)
    return plan, trace, loglines


# ---------------------------------------------------------------------------
# Re-probe + final check
# ---------------------------------------------------------------------------

def final_verify(profile: dict, initial_hash: str, plan: dict) -> None:
    print("\n── final verification ─" + "─" * 40)
    prof2, h2 = probe_metal()
    if h2 != initial_hash:
        sys.exit(f"ABORT: hardware profile changed (hash {initial_hash[:12]} -> {h2[:12]}).")
    print(f"  re-probe: profile hash stable ({h2[:12]}…)")
    result = call_check_plan(profile, plan, [])
    if "error" in result:
        sys.exit(f"ABORT: final empty-tactics check failed: {result['error']}")
    if result.get("accepted") is not True:
        sys.exit(f"ABORT: final plan rejected: {result.get('diagnostic', 'rejected')}")
    fl = result.get("final_plan")
    if not isinstance(fl, dict):
        sys.exit(f"ABORT: final empty-tactics check missing final_plan: {result!r}")
    if fl != plan:
        sys.exit("ABORT: final empty-tactics check returned a plan that does not "
                 "exactly match the requested plan.")
    print(f"  final plan accepted: q={plan['query_tile']} k={plan['key_tile']} "
          f"executable={bool(plan.get('alias_kv'))} "
          f"shared={plan_shared_bytes(plan['query_tile'], plan['key_tile'], plan['head_dim'], plan.get('alias_kv'))}B")


def require_executable(plan: dict, label: str) -> None:
    if not plan.get("alias_kv"):
        sys.exit(f"ABORT: {label} is NOT executable (alias_kv=false). "
                 "The vendor shader always aliases K/V; never dispatch a "
                 "non-executable plan.")


# ---------------------------------------------------------------------------
# Candidate checking + selection (testable without GPU)
# ---------------------------------------------------------------------------

def build_candidate_sequence(qt: int, kt: int) -> list[dict]:
    return [{"kind": "reuse_kv_storage"},
            {"kind": "set_query_tile", "value": qt},
            {"kind": "set_key_tile", "value": kt}]


def check_candidate(profile: dict, seq: int, dim: int, qt: int, kt: int) -> dict:
    """Check a (qt,kt) candidate through Lean, retaining the authoritative
    checked_plan, tactic_sequence and verification_trace.  Returns a record whose
    `status` is one of:
      - "ok_candidate": accepted, executable, ready to benchmark
      - "not_executable": accepted by Lean but alias_kv=false (never dispatch)
      - "rejected": a tactic was rejected (resource/legality)
      - "error": Lean CLI error
    The qt/kt used for benchmarking come from the checked_plan (authoritative),
    after asserting seq/dim/tile/executable invariants."""
    sequence = build_candidate_sequence(qt, kt)
    current = initial_plan(seq, dim)
    trace: list[dict] = []
    for tac in sequence:
        res = call_check_plan(profile, current, [tac])
        if "error" in res:
            return {"qt": qt, "kt": kt, "tactic_sequence": sequence,
                    "verification_trace": trace, "status": "error",
                    "error": res["error"]}
        if res.get("accepted") is not True:
            return {"qt": qt, "kt": kt, "tactic_sequence": sequence,
                    "verification_trace": trace, "status": "rejected",
                    "diagnostic": res.get("diagnostic", "rejected")}
        fl = res.get("final_plan")
        if not isinstance(fl, dict):
            return {"qt": qt, "kt": kt, "tactic_sequence": sequence,
                    "verification_trace": trace, "status": "error",
                    "error": "final_plan missing in accepted response"}
        current = fl
        trace.append({"tactic": tac, "final_plan": fl,
                      "executable": bool(res.get("executable")),
                      "shared_memory_bytes": res.get("shared_memory_bytes")})
    # Authoritative values come from the checked_plan (Lean's final_plan).
    if current["seq_len"] != seq or current["head_dim"] != dim:
        return {"qt": qt, "kt": kt, "tactic_sequence": sequence,
                "verification_trace": trace, "status": "error",
                "error": "checked_plan seq/dim mismatch"}
    checked_qt, checked_kt = current["query_tile"], current["key_tile"]
    alias = current.get("alias_kv")
    record = {
        "qt": checked_qt, "kt": checked_kt,
        "tactic_sequence": sequence,
        "checked_plan": current,
        "verification_trace": trace,
        "executable": bool(alias),
        "shared_memory_bytes": res.get("shared_memory_bytes"),
        "threads_per_threadgroup": current["threads_per_threadgroup"],
    }
    if not alias:
        record["status"] = "not_executable"
        return record
    if checked_qt != qt or checked_kt != kt:
        return {"qt": qt, "kt": kt, "tactic_sequence": sequence,
                "verification_trace": trace, "status": "error",
                "error": "checked_plan tile mismatch"}
    record["status"] = "ok_candidate"
    return record


def _median_ms(rec: dict) -> float | None:
    wt = rec.get("wall_timing") or {}
    m = wt.get("median_ms")
    return None if m is None else float(m)


def _finite_positive(v) -> bool:
    return v is not None and math.isfinite(v) and v > 0


def apply_benchmark(record: dict, bench_fn, args, run_dir: Path) -> dict:
    """Benchmark one ok_candidate via `bench_fn(args, run_dir, qt, kt)` using the
    authoritative checked_plan tiles, and record status + timings.  Only called
    for accepted, executable candidates."""
    qt, kt = record["qt"], record["kt"]
    rec = bench_fn(args, run_dir, qt, kt)
    r = dict(record)
    r["status"] = rec.get("status")
    r["wall_median_ms"] = _median_ms(rec)
    gpu_avail = bool(rec.get("gpu_time_available"))
    r["gpu_time_available"] = gpu_avail
    r["gpu_median_ms"] = (rec.get("gpu_timing") or {}).get("median_ms") if gpu_avail else None
    r["bench_inputs_sha256"] = rec.get("inputs_sha256")
    return r


def run_search_candidates(profile: dict, seq: int, dim: int, qts, kts, bench_fn, args, run_dir: Path) -> list[dict]:
    """Enumerate (qt,kt), check each through Lean, benchmark accepted executable
    ones, and return all candidate records (including rejections / losses)."""
    records: list[dict] = []
    for qt in qts:
        for kt in kts:
            record = check_candidate(profile, seq, dim, qt, kt)
            if record["status"] == "ok_candidate":
                record = apply_benchmark(record, bench_fn, args, run_dir)
            records.append(record)
    return records


def select_winner(records: list[dict]) -> dict | None:
    """Fastest candidate with status 'ok' and finite positive wall timing.  This
    excludes rejected, non-executable, numeric_failed, failed and zero/NaN
    timings."""
    good = [r for r in records
            if r.get("status") == "ok" and r.get("executable") is True
            and (r.get("checked_plan") or {}).get("alias_kv") is True
            and _finite_positive(r.get("wall_median_ms"))]
    if not good:
        return None
    return min(good, key=lambda r: r["wall_median_ms"])


def real_bench(args, run_dir: Path, qt: int, kt: int) -> dict:
    return steel_single(args, run_dir, qt, kt)


# ---------------------------------------------------------------------------
# Benchmark stage
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    print(f"  [run] {label}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ERROR: {label} failed ({proc.returncode})\n{proc.stdout}\n{proc.stderr}")


def _find_case(doc: dict, seq: int, dim: int, dtype: str | None = None) -> dict | None:
    for case in doc.get("cases", []):
        if case.get("seq") == seq and case.get("dim") == dim:
            if dtype is None or case.get("dtype") == dtype:
                return case
    return None


def steel_single(args, run_dir: Path, qt: int, kt: int) -> dict:
    """Run the vendor-derived steel attention specialization for one config and
    return its result record (status / medians)."""
    out = run_dir / f"steel_qt{qt}_kt{kt}.json"
    build = run_dir / f"steel_build_qt{qt}_kt{kt}"
    cmd = [
        sys.executable, str(STEEL_PY), "--output", str(out),
        "--build-dir", str(build),
        "--seqs", str(args.seq), "--dims", str(args.dim),
        "--qts", str(qt), "--kts", str(kt),
        "--seed", str(args.seed), "--heads", "8", "--batch", "1",
        "--warmup", str(args.warmup), "--samples", str(args.samples),
        "--inner", str(args.inner),
    ]
    _run(cmd, f"steel_attention.py qt{qt} kt{kt}")
    with open(out) as f:
        doc = json.load(f)
    case = _find_case(doc, args.seq, args.dim)
    if case is None:
        return {"status": "failed", "error": {"type": "MissingCase",
                                               "message": "no case in steel output"}}
    rec = (case.get("results") or {}).get(f"qt{qt}_kt{kt}") or {}
    return dict(rec, inputs_sha256=case.get("inputs_sha256"),
                source_provenance=doc.get("vendor"))


def run_survey(args, run_dir: Path) -> dict:
    out = run_dir / "metal_survey.json"
    build = run_dir / "vendor_build"
    cmd = [
        sys.executable, str(SURVEY_PY), "--output", str(out),
        "--build-dir", str(build),
        "--seqs", str(args.seq), "--dims", str(args.dim),
        "--dtypes", "float32", "--backends", "mlx_fast,mpsgraph",
        "--heads", "8", "--batch", "1",
        "--warmup", str(args.warmup), "--samples", str(args.samples),
        "--inner", str(args.inner), "--seed", str(args.seed),
    ]
    _run(cmd, "metal_survey.py (mlx_fast, mpsgraph)")
    with open(out) as f:
        return json.load(f)


def run_benchmark(args, profile: dict, checked_plan: dict, run_dir: Path) -> dict:
    """Freshly benchmark an (already verified) executable checked_plan against
    the vendor survey and return the comparison report.  Requires correct
    (`ok`) status and finite positive wall timing."""
    require_executable(checked_plan, "benchmark plan")
    seq, dim = args.seq, args.dim
    qt, kt = checked_plan["query_tile"], checked_plan["key_tile"]
    if checked_plan["seq_len"] != seq or checked_plan["head_dim"] != dim:
        sys.exit(f"ABORT: benchmark plan seq/dim mismatch ({checked_plan})")
    print(f"  candidate qt{qt} kt{kt}: running fresh replay")

    cand = steel_single(args, run_dir, qt, kt)
    if cand.get("status") != "ok":
        sys.exit(f"ABORT: candidate qt{qt} kt{kt} not ok: "
                 f"{cand.get('status')} {cand.get('error')}")
    cw = _median_ms(cand)
    if not _finite_positive(cw):
        sys.exit(f"ABORT: candidate qt{qt} kt{kt} missing/invalid wall timing: {cw}")
    print(f"  candidate qt{qt} kt{kt}: status=ok wall={cw} ms")

    surv_doc = run_survey(args, run_dir)
    surv_case = _find_case(surv_doc, seq, dim, "float32")
    if surv_case is None:
        sys.exit("ABORT: survey JSON missing the requested seq/dim case.")

    for field in ("q", "k", "v"):
        ch = (cand.get("inputs_sha256") or {}).get(field)
        sh = (surv_case.get("inputs_sha256") or {}).get(field)
        if not ch or ch != sh:
            sys.exit(f"ABORT: input hash mismatch for {field} between candidate "
                     f"and survey ({ch} vs {sh}).")

    cg = None
    if cand.get("gpu_time_available"):
        gt = cand.get("gpu_timing") or {}
        cg = gt.get("median_ms")

    vendor_medians: dict[str, float] = {}
    losses = []
    failed = []
    for backend in ("mlx_fast", "mpsgraph"):
        rec = (surv_case.get("results") or {}).get(backend) or {}
        st = rec.get("status")
        med = _median_ms(rec)
        if st == "ok" and med is not None:
            vendor_medians[backend] = med
            print(f"  vendor {backend}: status=ok wall median={med} ms")
        elif st == "numeric_failed":
            losses.append(backend)
            print(f"  vendor {backend}: LOSS (numeric_failed)")
        else:
            failed.append(backend)
            print(f"  vendor {backend}: failed/unsupported ({rec.get('error', {}).get('type', st)})")

    report = {
        "qt": qt, "kt": kt, "seq": seq, "dim": dim,
        "checked_plan": checked_plan,
        "candidate": cand,
        "vendor_results": surv_case.get("results", {}),
        "candidate_wall_median_ms": cw, "candidate_gpu_median_ms": cg,
        "vendor_wall_medians_ms": vendor_medians,
        "input_hashes_verified": True,
        "numeric_failed": losses, "failed_unsupported": failed,
    }
    if vendor_medians:
        fastest = min(vendor_medians, key=vendor_medians.get)
        fmed = vendor_medians[fastest]
        report["fastest_vendor"] = fastest
        report["fastest_vendor_wall_median_ms"] = fmed
        report["candidate_over_vendor_ratio"] = cw / fmed
        print(f"  fastest vendor: {fastest} ({fmed} ms)")
        print(f"  candidate/vendor ratio: {cw / fmed:.3f}x "
              f"(candidate {'faster' if cw < fmed else 'slower'})")
    if losses:
        print(f"  losses: {', '.join(losses)}")
    if failed:
        print(f"  failed/unsupported: {', '.join(failed)}")
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true",
                    help="no API: run a fixed schedule (trace preview)")
    ap.add_argument("--search", action="store_true",
                    help="enumerate Q8/16/24 x K8/16, check each via Lean, "
                         "benchmark every accepted candidate, pick fastest correct")
    ap.add_argument("--seq", type=int, default=2048, help="sequence length (default 2048)")
    ap.add_argument("--dim", type=int, default=192, choices=[192, 256],
                    help="head dimension (default 192)")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-benchmark", action="store_true",
                    help="skip the Metal candidate/vendor benchmark stage")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    for name, minimum in (("seq", 1), ("max_steps", 1), ("samples", 1), ("inner", 1)):
        if getattr(args, name) < minimum:
            ap.error(f"--{name.replace('_', '-')} must be >= {minimum} "
                     f"(got {getattr(args, name)})")
    if args.warmup < 0:
        ap.error(f"--warmup must be >= 0 (got {args.warmup})")

    print("╭─ VeriTac · LLM-driven Metal tiled-attention plan demo ─" + "─" * 22 + "╮")
    print("│  every tactic is verified by the Lean verifier before applying.")
    print("╰" + "─" * 60 + "╯")
    print("")

    env = load_env_file()
    args.model = (args.model or os.environ.get("OPENAI_MODEL")
                  or env.get("OPENAI_MODEL") or DEFAULT_MODEL)
    base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                or env.get("OPENAI_BASE_URL") or DEFAULT_BASE)
    base_url = ensure_completions_url(base_url)
    key: str | None = None
    if not args.mock:
        key, _src = get_api_key(env)
        if not key:
            print("No API key — running in --mock mode instead (trace preview).")
            args.mock = True

    profile, profile_hash = probe_metal()
    print(f"hardware: {profile.get('device_name')} "
          f"backend={profile.get('backend')} simd_width={profile.get('simd_width')} "
          f"threads={profile.get('max_threads_per_threadgroup')} "
          f"shared_limit={profile.get('max_threadgroup_memory_bytes')}B")
    print(f"profile hash: {profile_hash[:16]}…")
    print(f"attention seq={args.seq} dim={args.dim}, "
          f"{'mock schedule' if args.mock else 'LLM agent (' + args.model + ')'}"
          f"{' + search' if args.search else ''}")
    print("")

    OUT.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(tempfile.mkdtemp(prefix=f"run_{run_stamp}_", dir=OUT))
    print(f"outputs: {run_dir}")

    plan, agent_trace, loglines = run_agent(args, profile, key, base_url)
    alias = plan.get("alias_kv")
    print(f"\nagent final plan: q={plan['query_tile']} k={plan['key_tile']} "
          f"threads={plan['threads_per_threadgroup']} "
          f"shared={plan_shared_bytes(plan['query_tile'], plan['key_tile'], plan['head_dim'], alias)}B "
          f"executable={bool(alias)}")

    report = {
        "mode": "check_attention_plan",
        "seed": args.seed,
        "seq": args.seq, "dim": args.dim,
        "search": args.search,
        "hardware": {"device_name": profile.get("device_name"),
                     "profile_hash": profile_hash,
                     "normalized_profile": profile},
        "agent_final_plan": plan,
        "agent_tactic_trace": agent_trace,
        "label": "vendor-derived MLX steel attention small-tile specialization "
                 "(template-instantiated from installed mlx, not original)",
        "benchmark": None,
    }

    if args.no_benchmark:
        print("\n--no-benchmark: skipping Metal candidate/vendor benchmark.")
        final_verify(profile, profile_hash, plan)
        report["final_plan"] = plan
        report["final_executable"] = bool(alias)
        report["tactic_trace"] = agent_trace
        if not alias:
            print("  NOTE: agent final plan is NOT executable (alias_kv=false); it "
                  "would never be dispatched. Run reuse_kv_storage to make it executable.")
    elif args.search:
        print("\n── search + benchmark ─" + "─" * 44)
        records = run_search_candidates(profile, args.seq, args.dim,
                                        SEARCH_QTS, SEARCH_KTS, real_bench, args, run_dir)
        report["candidates"] = records
        for r in records:
            print(f"    qt{r['qt']} kt{r['kt']}: status={r.get('status')} "
                  f"wall={r.get('wall_median_ms')} ms")
        winner = select_winner(records)
        if winner is None:
            sys.exit("ABORT: no correct (ok) executable candidate to benchmark.")
        print(f"  [search] winner: qt{winner['qt']} kt{winner['kt']} "
              f"(selection wall={winner['wall_median_ms']} ms)")

        # Every stored trace must replay to its checked_plan before dispatch.
        if not replay_plan(profile, initial_plan(args.seq, args.dim),
                           winner["tactic_sequence"], winner["checked_plan"]):
            sys.exit("ABORT: winner trace does not replay to its checked_plan.")

        # Re-probe + empty-tactics check on the winner, then fresh replay.
        final_verify(profile, profile_hash, winner["checked_plan"])
        bench = run_benchmark(args, profile, winner["checked_plan"], run_dir)

        report["final_plan"] = winner["checked_plan"]
        report["final_executable"] = True
        report["tactic_trace"] = winner["verification_trace"]
        report["winner"] = winner
        report["selection_timing_ms"] = winner["wall_median_ms"]
        report["benchmark"] = bench
    else:
        final_verify(profile, profile_hash, plan)
        bench = run_benchmark(args, profile, plan, run_dir)
        report["final_plan"] = plan
        report["final_executable"] = bool(alias)
        report["tactic_trace"] = agent_trace
        report["benchmark"] = bench

    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(run_dir / "trace.log", "w") as f:
        f.write("\n".join(loglines) + "\n")
    print(f"\nreport: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
