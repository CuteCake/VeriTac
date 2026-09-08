"""
VeriTac LLM-driven GEMM demo.

The OpenAI API key, model ID and an optional OpenAI-compatible `OPENAI_BASE_URL`
are read from `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL`, root `.env`,
or entered interactively. Each step an LLM proposes one tactic; the Lean CLI verifies and
applies it; the full trace — proposed tactic, current statement tree, and any
error — is printed as the demo runs. When the LLM says "done", the final
schedule is benchmarked against the baseline.

Usage:
    .venv/bin/python examples/llm_gemm_demo.py            # interactive key entry
    .venv/bin/python examples/llm_gemm_demo.py --mock     # no API: fixed schedule
    .venv/bin/python examples/llm_gemm_demo.py --dim 128 --max-steps 8
    OPENAI_API_KEY=sk-... .venv/bin/python examples/llm_gemm_demo.py

Requires `lake build veritac` (Lean CLI) and numpy (install via
`python3 -m venv .venv && .venv/bin/pip install numpy`).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Search.interface import make_matmul_stmt, call_lean  # noqa: E402
from CodeGen.runner import benchmark_matmul  # noqa: E402

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE = "https://api.openai.com/v1/chat/completions"
OPENMP_NOTE = ("OpenMP compiler found; parallel/vectorize pragmas are live. "
               "Without one, pragmas are inert no-ops (still correct).")


def ensure_completions_url(base: str) -> str:
    """Normalize OPENAI_BASE_URL to a full chat-completions endpoint.

    It may be given as a bare API base (`https://host/api/v1`) or a complete
    endpoint (`https://host/api/v1/chat/completions`); we always POST to the
    latter, otherwise the request hits a non-POST route (HTTP 405).
    """
    base = base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


# ---------------------------------------------------------------------------
# Config: API key from env / .env / interactive prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM client (stdlib only, OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

def llm_propose(key: str, base_url: str, model: str, messages: list[dict]) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    # A browser-like User-Agent is required: many gateways sit behind Cloudflare
    # bot management, which 403/1010-blocks the default "Python-urllib/3.x".
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
    """Extract a JSON object from the LLM reply (tolerates fences/prose)."""
    reply = reply.strip()
    # try fenced block first
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
# Statement tree rendering for the trace / LLM context
# ---------------------------------------------------------------------------

def sexpr_str(e) -> str:
    if not isinstance(e, dict):
        return str(e)
    tag = e.get("tag")
    if tag == "lit":
        return str(e["val"])
    if tag == "var":
        return e["name"]
    if tag in ("add", "mul", "div", "mod"):
        op = {"add": "+", "mul": "*", "div": "/", "mod": "%"}[tag]
        return f"({sexpr_str(e['left'])}{op}{sexpr_str(e['right'])})"
    if tag == "bufRead":
        return f"{e['buf']}[{','.join(sexpr_str(i) for i in e['indices'])}]"
    return "?"


def dump_stmt(node, depth: int = 0, out: list[str] | None = None) -> list[str]:
    out = out if out is not None else []
    pad = "  " * depth
    tag = node.get("tag")
    if tag == "loop":
        out.append(f"{pad}loop {node['var']} [{sexpr_str(node['lo'])}, {sexpr_str(node['hi'])}) "
                   f"ann={node['ann']}")
        dump_stmt(node["body"], depth + 1, out)
    elif tag == "seq":
        dump_stmt(node["s1"], depth, out)
        dump_stmt(node["s2"], depth, out)
    elif tag == "bufWrite":
        idx = ",".join(sexpr_str(i) for i in node["indices"])
        out.append(f"{pad}{node['buf']}[{idx}] = {sexpr_str(node['val'])}")
    elif tag == "alloc":
        out.append(f"{pad}alloc {node['buf']}[{','.join(sexpr_str(s) for s in node['shape'])}]")
        dump_stmt(node["body"], depth + 1, out)
    elif tag == "skip":
        out.append(f"{pad}skip")
    else:
        out.append(f"{pad}{tag}")
    return out


# ---------------------------------------------------------------------------
# Mock proposer (no API key; fixed reasonable schedule with feedback)
# ---------------------------------------------------------------------------

MOCK_SEQUENCE = [
    {"kind": "tile", "vars": ["i0"], "int_params": [32], "str_params": []},
    {"kind": "tile", "vars": ["i1"], "int_params": [32], "str_params": []},
    {"kind": "parallel", "vars": ["i0_outer"], "int_params": [], "str_params": []},
    {"kind": "vectorize", "vars": ["i2"], "int_params": [], "str_params": []},
]


def mock_propose(step_index: int, schedule_so_far: list[dict], last_error: str | None,
                 seen: set[int]) -> dict:
    if last_error and step_index in seen:
        # a previously successful tactic failed on replay; skip it
        for i in range(step_index, len(MOCK_SEQUENCE)):
            canon = json.dumps(MOCK_SEQUENCE[i], sort_keys=True)
            if canon not in seen:
                seen.add(canon)
                return MOCK_SEQUENCE[i]
        return {"kind": "done"}
    if step_index < len(MOCK_SEQUENCE):
        return MOCK_SEQUENCE[step_index]
    return {"kind": "done"}


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the scheduling agent for VeriTac, a verified tensor compiler for matmul \
loop nests. Every tactic you propose is machine-checked by a Lean verifier — if \
it is invalid (bad tile size, unsafe reorder, non-independent parallel axis, ...) \
the verifier rejects it and you will see the error. Pick ONE tactic per turn.

Available tactic JSON:
- {"kind":"tile","vars":["<loopVar>"],"int_params":[<tileSize>]} \
  split a loop into outer/inner. tileSize must be > 0 and must DIVIDE the loop \
  extent (e.g. 256 -> 32/64/128). After tiling, the inner is named `<v>_inner` \
  and the outer `<v>_outer`.
- {"kind":"reorder","vars":["<v1>","<v2>"],"int_params":[]} swap two adjacent \
  nested loops (rejected if not safe).
- {"kind":"fuse","vars":["<v1>","<v2>"],"int_params":[]} merge two nested loops.
- {"kind":"parallel","vars":["<loopVar>"],"int_params":[]} annotate a loop for \
  OpenMP parallel execution.
- {"kind":"vectorize","vars":["<loopVar>"],"int_params":[]} annotate for SIMD.
- {"kind":"unroll","vars":["<loopVar>"],"int_params":[]} unroll (needs literal bounds).

Goal: speed up `C[i0][i1] += A[i0][i2] * B[i2][i1]` (matmul, dims are square). \
A good verified schedule: tile the two outer axes, annotate the outermost loop \
parallel. To make an inner axis vectorizable, first use reorder to move the k-loop \
(i2) OUTSIDE of a tiled axis — e.g. after tiling i1 into i1_outer/i1_inner, \
`reorder i1_inner i2` puts i2 above i1_inner so i1_inner becomes the innermost \
loop, and only then vectorize i1_inner. Do NOT vectorize/parallelize the \
reduction loop over k (i2) — the verifier's codegen guard will suppress it.

When you are done improving, reply exactly: {"kind":"done"}.

Reply with ONLY the JSON. No commentary."""


def build_user_message(schedule: list[dict], stmt_json: dict,
                       last_result: str) -> str:
    tree = "\n".join(dump_stmt(stmt_json))
    sched = json.dumps(schedule, indent=0) if schedule else "(none yet)"
    return (
        f"Schedule so far: {sched}\n\n"
        f"Current statement:\n{tree}\n\n"
        f"Last action result: {last_result}\n\n"
        "Propose the next tactic (or done):"
    )


def run_agent(args, key: str | None, base_url: str) -> tuple[list[dict], dict, str]:
    """Run the agent loop; returns (schedule, final_stmt, trace)."""
    stmt = make_matmul_stmt(args.dim, args.dim, args.dim)
    schedule: list[dict] = []
    trace_lines: list[str] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_result = "start"

    def log(line: str):
        trace_lines.append(line)
        print(line)

    for step in range(1, args.max_steps + 1):
        log("")
        log(f"── step {step}/{args.max_steps} ─" + "─" * 40)

        # --- propose ---
        if args.mock:
            propose = mock_propose(step - 1, schedule, None, set())
            log(f"  llm (mock): {json.dumps(propose)}")
        else:
            user = build_user_message(schedule, stmt, last_result)
            messages = messages[:1] + [{"role": "user", "content": user}]
            propose = None
            for attempt in range(3):
                try:
                    reply = llm_propose(key, base_url, args.model, messages)
                except urllib.error.HTTPError as e:
                    body = e.read().decode()[:300]
                    log(f"  api error: HTTP {e.code}: {body}")
                    log("  (check your key / quota / base URL)")
                    return schedule, stmt, "API ERROR — see trace above"
                except Exception as e:  # network etc.
                    log(f"  api error: {type(e).__name__}: {e}")
                    return schedule, stmt, "API ERROR — see trace above"
                propose = parse_llm_reply(reply)
                if propose is not None:
                    break
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user",
                                 "content": "Your reply was not valid JSON. "
                                            "Reply with ONLY one tactic JSON or "
                                            '{"kind":"done"}.'})
            if propose is None:
                log("  llm: <unparseable reply after retries>")
                break
            log(f"  llm: {json.dumps(propose)}")

        if propose.get("kind") == "done" or not propose:
            log("  llm: done")
            break

        # --- verify + apply via Lean CLI ---
        # Apply only the *new* tactic to the current statement (the Lean CLI
        # supports sequence application too, but replaying the whole schedule
        # would re-run already-applied transformations on the transformed IR).
        r = call_lean(stmt, [propose])
        if "error" in r:
            last_result = f"REJECTED: {r['error']}"
            log(f"  lean: ✗ {r['error']}")
            log(f"  stmt: unchanged — {json.dumps(propose)} not applied")
            messages.append({"role": "assistant", "content": json.dumps(propose)})
            messages.append({"role": "user", "content": f"The verifier rejected "
                            f"{json.dumps(propose)}: {r['error']}. Propose something "
                            f"else or done."})
            continue
        schedule = schedule + [propose]
        stmt = r["stmt"]
        last_result = f"OK (applied {r['applied']} tactics)"
        log(f"  lean: ✓ applied")
        log("  stmt:")
        for tree_line in dump_stmt(stmt):
            log(f"      {tree_line}")

    log("")
    log("══ agent finished ═" + "═" * 42)
    return schedule, stmt, "\n".join(trace_lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--mock", action="store_true",
                    help="no API: run a fixed schedule end-to-end (trace preview)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    print("╭─ VeriTac · LLM-driven GEMM demo ─" + "─" * 40 + "╮")
    print("│  every tactic is verified by the Lean verifier before applying.")
    print("╰" + "─" * 60 + "╯")
    print("")

    env = load_env_file()
    args.model = (args.model or os.environ.get("OPENAI_MODEL")
                  or env.get("OPENAI_MODEL") or DEFAULT_MODEL)
    key: str | None = None
    base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL")
                or env.get("OPENAI_BASE_URL") or DEFAULT_BASE)
    base_url = ensure_completions_url(base_url)
    if not args.mock:
        key, _src = get_api_key(env)
        if not key:
            print("No API key — running in --mock mode instead (trace preview).")
            args.mock = True

    print(f"matmul {args.dim}^3, {args.runs} runs, "
          f"{'mock schedule' if args.mock else 'LLM agent (' + args.model + ')'}")
    print("")

    # baseline
    print("── baseline ─" + "─" * 46)
    base_stmt = make_matmul_stmt(args.dim, args.dim, args.dim)
    b0 = benchmark_matmul(base_stmt, args.dim, args.dim, args.dim, num_runs=args.runs)
    print(f"  baseline C time: {b0['c_time_ms']:.3f} ms   correct: {b0['correct']}")

    # agent
    schedule, final_stmt, trace = run_agent(args, key, base_url)

    if not schedule:
        print("\nNo tactics applied — nothing to benchmark.")
        return

    # final benchmark
    print("\n── final benchmark ─" + "─" * 40)
    b1 = benchmark_matmul(final_stmt, args.dim, args.dim, args.dim, num_runs=args.runs)
    print(f"  optimized C time: {b1['c_time_ms']:.3f} ms   correct: {b1['correct']}")
    if b1["c_time_ms"] and b0["c_time_ms"]:
        print(f"  speedup (C vs C): {b0['c_time_ms'] / b1['c_time_ms']:.2f}x")
    print(f"  numpy (OpenBLAS, multithreaded) reference: {b1['numpy_time_ms']:.3f} ms")
    print("\n  final schedule:")
    for t in schedule:
        print(f"    {t['kind']} {t['vars']}{t['int_params'] or ''}")


if __name__ == "__main__":
    main()