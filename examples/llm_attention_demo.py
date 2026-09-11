"""
VeriTac LLM-driven Metal attention-launch demo.

Probes the local Metal device into a validated `Hardware.profile`, then has an
LLM (or a fixed mock) propose refinements to an attention launch — mapping,
query tile, key tile — one tactic at a time.  Every proposal is machine-checked
by the Lean CLI (`check_attention_tactics`); state changes only when accepted.
Before benchmarking the target is re-probed (profile hash must be unchanged)
and the final launch is re-checked with an empty tactic list.

The benchmark stage shells out to the Metal candidate kernel
(`benchmarks/attention/metal_candidate/metal_candidate.py`) with the exact
accepted mapping/query/key tile, and to the vendor survey
(`benchmarks/attention/metal_survey.py`) with the `mlx_fast` and `mpsgraph`
backends at float32.  Input q/k/v hashes must match across both runs.

The OpenAI API key, model and an optional OpenAI-compatible base URL are read
from `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL`, the root `.env`,
or entered interactively (stdlib `urllib` only).  `--mock` never asks for a key.

Usage:
    .venv/bin/python examples/llm_attention_demo.py                 # interactive key
    .venv/bin/python examples/llm_attention_demo.py --mock          # fixed schedule
    .venv/bin/python examples/llm_attention_demo.py --seq 128 --dim 192 --max-steps 8
    OPENAI_API_KEY=sk-... .venv/bin/python examples/llm_attention_demo.py --no-benchmark

Requires `lake build veritac` (Lean CLI).  Outputs are written under
`.lake/attention_demo`.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Hardware import profile as hw  # noqa: E402

OUT = ROOT / ".lake" / "attention_demo"
VERITAC_BIN = ROOT / ".lake" / "build" / "bin" / "veritac"
CANDIDATE_PY = ROOT / "benchmarks" / "attention" / "metal_candidate" / "metal_candidate.py"
SURVEY_PY = ROOT / "benchmarks" / "attention" / "metal_survey.py"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE = "https://api.openai.com/v1/chat/completions"
FP32_BYTES = 4
MOCK_SEQUENCE = [
    {"kind": "set_mapping", "mapping": "simdgroup"},
    {"kind": "set_query_tile", "value": 8},
    {"kind": "set_key_tile", "value": 8},
]
LEGAL = (
    "scalar: query_tile 8 or 16, threads_per_threadgroup == query_tile; "
    "simdgroup: requires simd_width 32, query_tile 4 or 8, "
    "threads_per_threadgroup == query_tile * simd_width. "
    "key_tile 8 or 16. head_dim 192 or 256. dtype FP32 (4 bytes). "
    "stage_k and stage_v must both be true."
)


def ensure_completions_url(base: str) -> str:
    base = base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def load_env_file() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def get_api_key(env: dict) -> tuple[str | None, str]:
    key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if key:
        return key, "env/.env"
    print("No OPENAI_API_KEY found.")
    try:
        key = input("  Enter your OpenAI API key (kept in .env; Enter to skip): ").strip()
    except EOFError:
        return None, "none"
    if key:
        try:
            with open(ROOT / ".env", "a") as f:
                f.write(f"\nOPENAI_API_KEY={key}\n")
            print("  saved to .env")
        except OSError as e:
            print(f"  (could not write .env: {e})")
        return key, "typed"
    return None, "none"


def llm_propose(key: str, base_url: str, model: str, messages: list[dict]) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0 Safari/537.36"),
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def parse_llm_reply(reply: str) -> dict | None:
    reply = reply.strip()
    if reply.startswith("```"):
        lines = reply.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        reply = "\n".join(lines).strip()
    start = reply.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(reply)):
        if reply[i] == "{":
            depth += 1
        elif reply[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(reply[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

def probe_metal() -> tuple[dict, str]:
    prof = hw.probe(backend="metal")
    if prof is None:
        sys.exit("ERROR: could not probe a Metal hardware profile.")
    hw.validate_schema(prof)
    h = hw.profile_hash(prof)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hardware.profile.json", "w") as f:
        f.write(hw.canonical_json(prof) + "\n")
    return prof, h


def shared_bytes(launch: dict) -> int:
    staged = launch["key_tile"] * launch["head_dim"] * FP32_BYTES
    return 2 * staged + launch.get("extra_shared_bytes", 0)


def derived_threads(launch: dict, simd_width: int) -> int:
    if launch["mapping"] == "simdgroup":
        return launch["query_tile"] * simd_width
    return launch["query_tile"]


def initial_launch(dim: int) -> dict:
    return {
        "mapping": "scalar",
        "head_dim": dim,
        "query_tile": 8,
        "key_tile": 16,
        "threads_per_threadgroup": 8,
        "dtype_bytes": FP32_BYTES,
        "stage_k": True,
        "stage_v": True,
        "extra_shared_bytes": 0,
    }


# ---------------------------------------------------------------------------
# Lean CLI
# ---------------------------------------------------------------------------

def call_check_tactics(target: dict, launch: dict, tactics: list[dict]) -> dict:
    if not VERITAC_BIN.exists():
        sys.exit(f"ERROR: veritac binary not found at {VERITAC_BIN}. Run 'lake build veritac'.")
    payload = {
        "mode": "check_attention_tactics",
        "schema_version": 1,
        "target": target,
        "initial_launch": launch,
        "tactics": tactics,
    }
    proc = subprocess.run(
        [str(VERITAC_BIN), "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()}
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON from CLI: {e}"}


def apply_tactic(launch: dict, tac: dict, simd_width: int) -> dict:
    l = dict(launch)
    kind = tac.get("kind")
    if kind == "set_mapping":
        l["mapping"] = tac["mapping"]
    elif kind == "set_query_tile":
        l["query_tile"] = tac["value"]
    elif kind == "set_key_tile":
        l["key_tile"] = tac["value"]
    else:
        raise ValueError(f"unknown tactic {tac!r}")
    l["threads_per_threadgroup"] = derived_threads(l, simd_width)
    return l


def mock_propose(step: int) -> dict | None:
    if step < len(MOCK_SEQUENCE):
        return MOCK_SEQUENCE[step]
    return {"kind": "done"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def make_system_prompt(profile: dict) -> str:
    fragment = hw.render_prompt(profile)
    return (
        "You are the scheduling agent for VeriTac's verified Metal attention-"
        "launch configurator. Every tactic you propose is machine-checked by a "
        "Lean verifier; if it is invalid the verifier rejects it and you will "
        "see the error. Pick ONE tactic per turn.\n\n"
        f"{fragment}\n"
        "Shared-memory byte formula: shared_bytes = 2 * key_tile * head_dim * "
        "dtype_bytes(4) + extra_shared_bytes (both K and V staged).\n"
        "Legal mappings/tiles:\n"
        f"- {LEGAL}\n"
        "Available tactic JSON (propose exactly one):\n"
        '- {"kind":"set_mapping","mapping":"scalar"} or '
        '{"kind":"set_mapping","mapping":"simdgroup"}\n'
        '- {"kind":"set_query_tile","value":<tile>}\n'
        '- {"kind":"set_key_tile","value":<8|16>}\n'
        "When done improving, reply exactly: {\"kind\":\"done\"}.\n\n"
        "Reply with ONLY the JSON. No commentary."
    )


def build_user_message(launch: dict, last_result: str) -> str:
    return (
        "Current launch (JSON):\n"
        f"{json.dumps(launch, indent=2)}\n\n"
        f"Shared bytes: {shared_bytes(launch)}\n\n"
        f"Last verifier feedback: {last_result}\n\n"
        "Propose the next single tactic (or done):"
    )


def run_agent(args, profile: dict, key: str | None, base_url: str) -> tuple[dict, list[str]]:
    simd_width = profile.get("simd_width", 0) or 0
    launch = initial_launch(args.dim)
    trace: list[str] = []
    last_result = "start"

    def log(line: str):
        trace.append(line)
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
            user = build_user_message(launch, last_result)
            messages = messages[:1] + [{"role": "user", "content": user}]
            propose = None
            for _attempt in range(3):
                try:
                    reply = llm_propose(key, base_url, args.model, messages)
                except urllib.error.HTTPError as e:
                    body = e.read().decode()[:300]
                    log(f"  api error: HTTP {e.code}: {body}")
                    log("  (check your key / quota / base URL)")
                    return launch, trace
                except Exception as e:  # noqa: BLE001
                    log(f"  api error: {type(e).__name__}: {e}")
                    return launch, trace
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

        result = call_check_tactics(profile, launch, [propose])
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
            continue

        fl = result.get("final_launch")
        if not isinstance(fl, dict):
            sys.exit(f"ABORT: accepted response missing final_launch: {result!r}")
        expected = apply_tactic(launch, propose, simd_width)
        if fl != expected:
            sys.exit(f"ABORT: final_launch mismatch (got {fl!r}, expected {expected!r})")
        launch = fl
        last_result = f"OK (accepted {json.dumps(propose)})"
        log(f"  lean: ✓ accepted {json.dumps(propose)}")
        log(f"  launch: mapping={launch['mapping']} q={launch['query_tile']} "
            f"k={launch['key_tile']} threads={launch['threads_per_threadgroup']} "
            f"shared={shared_bytes(launch)}B")

    log("")
    log("══ agent finished ═" + "═" * 42)
    return launch, trace


# ---------------------------------------------------------------------------
# Re-probe + final check
# ---------------------------------------------------------------------------

def final_verify(profile: dict, initial_hash: str, launch: dict) -> None:
    print("\n── final verification ─" + "─" * 40)
    prof2, h2 = probe_metal()
    if h2 != initial_hash:
        sys.exit(f"ABORT: hardware profile changed (hash {initial_hash[:12]} -> {h2[:12]}).")
    print(f"  re-probe: profile hash stable ({h2[:12]}…)")
    result = call_check_tactics(profile, launch, [])
    if "error" in result:
        sys.exit(f"ABORT: final empty-tactics check failed: {result['error']}")
    if result.get("accepted") is not True:
        sys.exit(f"ABORT: final launch rejected: {result.get('diagnostic', 'rejected')}")
    fl = result.get("final_launch")
    if not isinstance(fl, dict):
        sys.exit(f"ABORT: final empty-tactics check missing final_launch: {result!r}")
    if fl != launch:
        sys.exit("ABORT: final empty-tactics check returned a launch that does not "
                 "exactly match the requested launch.")
    print(f"  final launch accepted: mapping={launch['mapping']} "
          f"q={launch['query_tile']} k={launch['key_tile']} "
          f"shared={shared_bytes(launch)}B")


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


def _median_ms(rec: dict) -> float | None:
    wt = rec.get("wall_timing") or {}
    m = wt.get("median_ms")
    return None if m is None else float(m)


def run_benchmark(args, profile: dict, launch: dict) -> None:
    print("\n── benchmark ─" + "─" * 48)
    OUT.mkdir(parents=True, exist_ok=True)
    cand_out = OUT / "candidate.json"
    surv_out = OUT / "survey.json"
    seq, dim = args.seq, args.dim
    mapping = launch["mapping"]
    qt, kt = launch["query_tile"], launch["key_tile"]

    cand_cmd = [
        sys.executable, str(CANDIDATE_PY), "--output", str(cand_out),
        "--build-dir", str(OUT / "candidate_build"),
        "--seqs", str(seq), "--dims", str(dim),
        "--mappings", mapping, "--query-tiles", str(qt), "--key-tiles", str(kt),
        "--warmup", str(args.warmup), "--samples", str(args.samples),
        "--inner", str(args.inner), "--seed", str(args.seed),
    ]
    surv_cmd = [
        sys.executable, str(SURVEY_PY), "--output", str(surv_out),
        "--build-dir", str(OUT / "vendor_build"),
        "--seqs", str(seq), "--dims", str(dim),
        "--dtypes", "float32", "--backends", "mlx_fast,mpsgraph",
        "--warmup", str(args.warmup), "--samples", str(args.samples),
        "--inner", str(args.inner), "--seed", str(args.seed),
    ]

    _run(cand_cmd, "metal_candidate.py")
    _run(surv_cmd, "metal_survey.py")

    with open(cand_out) as f:
        cand_doc = json.load(f)
    with open(surv_out) as f:
        surv_doc = json.load(f)

    cand_case = _find_case(cand_doc, seq, dim)
    surv_case = _find_case(surv_doc, seq, dim, "float32")
    if cand_case is None or surv_case is None:
        sys.exit("ABORT: benchmark JSON missing the requested seq/dim case.")

    for field in ("q", "k", "v"):
        ch = cand_case.get("inputs_sha256", {}).get(field)
        sh = surv_case.get("inputs_sha256", {}).get(field)
        if not ch or ch != sh:
            sys.exit(f"ABORT: input hash mismatch for {field} between candidate "
                     f"and survey ({ch} vs {sh}).")

    cand_key = f"{mapping}_qt{qt}_kt{kt}"
    cand = (cand_case.get("results") or {}).get(cand_key)
    if cand is None:
        sys.exit(f"ABORT: candidate result missing for {cand_key}.")

    print(f"  candidate ({mapping} q{qt} k{kt}):")
    print(f"    status={cand.get('status')}")
    cw = _median_ms(cand)
    cg = None
    if cand.get("gpu_time_available"):
        gt = cand.get("gpu_timing") or {}
        cg = gt.get("median_ms")
    print(f"    wall median={cw} ms" if cw is not None else "    wall median=unavailable")
    if cg is not None:
        print(f"    gpu  median={cg} ms")
    if cw is None:
        sys.exit("ABORT: candidate wall median unavailable; cannot compare.")

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
            err = (rec.get("error") or {}).get("type", st)
            print(f"  vendor {backend}: failed/unsupported ({err})")

    if not vendor_medians:
        print("  no usable vendor median; no ratio computed.")
        return

    fastest = min(vendor_medians, key=vendor_medians.get)
    fmed = vendor_medians[fastest]
    print(f"  fastest vendor: {fastest} ({fmed} ms)")
    print(f"  candidate/vendor ratio: {cw / fmed:.3f}x (candidate {'faster' if cw < fmed else 'slower'})")
    if losses:
        print(f"  losses: {', '.join(losses)}")
    if failed:
        print(f"  failed/unsupported: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true",
                    help="no API: run a fixed schedule (trace preview)")
    ap.add_argument("--seq", type=int, default=128, help="sequence length (default 128)")
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

    print("╭─ VeriTac · LLM-driven Metal attention demo ─" + "─" * 34 + "╮")
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
          f"{'mock schedule' if args.mock else 'LLM agent (' + args.model + ')'}")
    print("")

    launch, _trace = run_agent(args, profile, key, base_url)
    print(f"\nfinal launch: mapping={launch['mapping']} "
          f"q={launch['query_tile']} k={launch['key_tile']} "
          f"threads={launch['threads_per_threadgroup']} "
          f"shared={shared_bytes(launch)}B")

    final_verify(profile, profile_hash, launch)

    if args.no_benchmark:
        print("\n--no-benchmark: skipping Metal candidate/vendor benchmark.")
        return
    run_benchmark(args, profile, launch)


if __name__ == "__main__":
    main()
