"""Blind live-search harness: untrusted model proposals for Gemmini GEMM programs.

The model runs in an isolated OpenCode session (a fresh session per call, all
built-in tools disabled, every permission denied, no plugins, no MCP servers,
sharing disabled) inside a fresh system-temporary directory outside this
repository.  The prompt contains only the workload plan, the exact instruction
semantics, and the exact response schema -- never a baseline implementation,
templates, B-reuse strategy hints, or repository paths.

Pipeline per round: model call -> strict proposal parse (duplicate JSON keys,
bool-as-int, unexpected/missing command fields, and contract changes are
rejected) -> exact per-field type validation derived from the backend command
classes -> Lean program checker as a fast round filter -> structured feedback
(candidate syntax, checker acceptance, and metrics only; no prescribed repair).
Candidates are scored on ACTUAL commands by (total DMA input bytes, command
count).  The final winner alone receives a real kernel certificate through
lowering.emit_checked_candidate; "native accepted" and "certified" are kept
strictly separate statuses.

Integrity: sha256 hashes of the Lean sources, the built checker binaries, the
toolchain pin, and this controller are frozen at run start and re-verified
before each round's validation, before final certification, and at report
time; any drift aborts the run fail-closed.

Isolation support (verified live against OpenCode 1.18.31): the inline
OPENCODE_CONFIG_CONTENT env var is honored as a runtime config override, and
{"tools": {"*": false}, "permission": {"*": "deny"}, "plugin": [], "mcp": {}})
resolves correctly -- a live probe confirmed the model can emit no executing
tool calls.  Residual limitation, documented instead of claimed away: OpenCode
merges user-global config files (~/.config/opencode) and managed settings that
this harness cannot un-merge, so globally declared plugins/MCP servers may
still be loaded (their execution remains denied).  The harness therefore
claims denied tool EXECUTION, not a hermetic sandbox; probe_isolation()
re-verifies the inline config at run start and the report records whether
isolation was enforced.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import backend as g
from . import certificate as certificates
from . import encoding as enc
from . import lowering
from . import diagnostics

DEFAULT_MODEL = "dgxspark-glm/glm-5.3-flash"
DEFAULT_MAX_ROUNDS = 3
DEFAULT_MODEL_TIMEOUT = 900.0
DEFAULT_PROOF_TIMEOUT = 300.0
DEFAULT_MAX_COMMANDS = 4096
DEFAULT_INTERPRET_MACS_LIMIT = 4_000_000
RUN_SCHEMA = "veritac_blind_search_run_v1"
FEEDBACK_SCHEMA = "veritac_blind_search_feedback_v1"
OPENCODE_VERSION_PROBED = "1.18.31 (verified live during development)"

STATUS_MODEL_TIMEOUT = "model_timeout"
STATUS_MODEL_PROCESS_FAILURE = "model_process_failure"
STATUS_MODEL_MALFORMED = "model_malformed"
STATUS_PROPOSAL_REJECTED = "proposal_rejected"
STATUS_NATIVE_REJECTED = "native_rejected"
STATUS_NATIVE_ACCEPTED = "native_accepted"
STATUS_CERTIFIED = "certified"
STATUS_CERTIFICATION_FAILED = "certification_failed"

PLAN_KEYS = ("m", "n", "k", "dim", "scratchpad_rows", "accumulator_rows",
             "schedule")

COMMAND_CLASSES = {cls.kind: cls for cls in (
    g.ConfigEx, g.ConfigLd, g.ConfigSt, g.Mvin,
    g.Preload, g.Compute, g.Mvout, g.Fence)}

# Exact field types demanded BEFORE any backend call.  int/bool/str use
# type(...) identity so JSON bools never pass as ints; "number" allows int or
# float (never bool).  The table is cross-checked against the actual backend
# command class signatures at import time below.
FIELD_TYPES = {
    "config_ex": {"dataflow": int},
    "config_ld": {"slot": int, "stride_bytes": int, "scale": "number"},
    "config_st": {"stride_bytes": int},
    "mvin": {"slot": int, "buf": str, "offset": int, "spad_addr": int,
             "cols": int, "rows": int},
    "preload": {"bd_spad_addr": int, "out_addr": int},
    "compute": {"accumulated": bool, "a_spad_addr": int, "bd_spad_addr": int},
    "mvout": {"buf_offset": int, "acc_addr": int, "cols": int, "rows": int},
    "fence": {},
}


class BlindSearchError(Exception):
    pass


class ContractViolation(BlindSearchError):
    pass


class IntegrityDrift(BlindSearchError):
    pass


def _check_field_table():
    for kind, fields in FIELD_TYPES.items():
        cls = COMMAND_CLASSES[kind]
        if cls.__init__ is object.__init__ and not fields:
            continue
        params = [name for name in inspect.signature(cls.__init__).parameters
                  if name != "self"]
        if sorted(params) != sorted(fields):
            raise RuntimeError(
                "FIELD_TYPES %r disagrees with backend class %r (fields %r)"
                % (kind, cls.__name__, params))


_check_field_table()


# ---------------------------------------------------------------------------
# Strict JSON: duplicate keys are always a contract violation.
# ---------------------------------------------------------------------------

def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %r" % (key,))
        result[key] = value
    return result


def strict_json_loads(text):
    def bad_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=bad_constant)


# ---------------------------------------------------------------------------
# Task input: exactly {id, plan}; the plan is the immutable contract.
# ---------------------------------------------------------------------------

def load_task(path):
    text = Path(path).read_text()
    try:
        doc = strict_json_loads(text)
    except ValueError as exc:
        raise ContractViolation("task JSON invalid: %s" % exc)
    if not isinstance(doc, dict) or set(doc) != {"id", "plan"}:
        raise ContractViolation(
            "task must be a JSON object with exactly the keys {id, plan} "
            "(got %r)" % (sorted(doc) if isinstance(doc, dict) else type(doc).__name__))
    task_id = doc["id"]
    if (type(task_id) is not str or not task_id or len(task_id) > 80
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id)):
        raise ContractViolation("task id must be a safe alphanumeric identifier")
    plan = doc["plan"]
    if not isinstance(plan, dict) or set(plan) != set(PLAN_KEYS):
        raise ContractViolation(
            "plan must have exactly the fields %s" % (list(PLAN_KEYS),))
    for key in PLAN_KEYS[:-1]:
        if type(plan[key]) is not int:
            raise ContractViolation(
                "plan field %r must be an integer (bool-as-int rejected)" % key)
    if type(plan["schedule"]) is not str:
        raise ContractViolation("plan schedule must be a string")
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ContractViolation("plan rejected: %s" % reason)
    return {"id": task_id, "plan": dict(plan)}


# ---------------------------------------------------------------------------
# Command validation: exact schema + exact types, before any backend call.
# ---------------------------------------------------------------------------

def validate_command(entry):
    if not isinstance(entry, dict):
        raise ValueError("command must be a JSON object")
    kind = entry.get("kind")
    if type(kind) is not str or kind not in COMMAND_CLASSES:
        raise ValueError("unknown command kind %r" % (kind,))
    expected = set(FIELD_TYPES[kind])
    actual = set(entry) - {"kind"}
    if actual != expected:
        raise ValueError("command %r fields %s violate the exact schema %s"
                         % (kind, sorted(actual), sorted(expected)))
    kwargs = {}
    for name, typ in FIELD_TYPES[kind].items():
        value = entry[name]
        if typ is int:
            if type(value) is not int or not 0 <= value < (1 << 64):
                raise ValueError("command %r field %r must be an integer "
                                 "(bool-as-int rejected)" % (kind, name))
        elif typ is bool:
            if type(value) is not bool:
                raise ValueError("command %r field %r must be a Boolean"
                                 % (kind, name))
        elif typ is str:
            if type(value) is not str:
                raise ValueError("command %r field %r must be a string"
                                 % (kind, name))
        elif typ == "number":
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("command %r field %r must be a number"
                                 % (kind, name))
        kwargs[name] = value
    command = COMMAND_CLASSES[kind](**kwargs)
    enc.command_dict(command)  # canonical form the encoding backend will see
    return command


def validate_subset(command):
    """Cheap semantic pre-checks of the supported subset (fail fast, before
    the checker; the Lean checker remains the authority)."""
    c = enc.command_dict(command)
    kind = c["kind"]
    if kind == "config_ex" and c["dataflow"] != g.WEIGHT_STATIONARY:
        raise ValueError("only weight-stationary dataflow=%d is supported"
                         % g.WEIGHT_STATIONARY)
    if kind == "config_ld":
        if c["slot"] not in (0, 1):
            raise ValueError("config_ld slot must be 0 or 1")
        if c["scale"] != 1.0:
            raise ValueError("only identity load scale 1.0 is supported")
    if kind == "mvin" and (c["slot"] not in (0, 1) or c["buf"] not in ("A", "B")):
        raise ValueError("mvin slot must be 0/1 and buf must be 'A' or 'B'")


def parse_proposal(text, max_commands):
    """Extract the proposal from free model text.

    Returns (commands, rationale).  Raises ProposalRejected when a JSON object
    parsed but violated the contract, MalformedProposal when no JSON object
    could be parsed at all.
    """
    contract_error = None
    for candidate in extract_json_candidates(text):
        try:
            doc = strict_json_loads(candidate)
        except ValueError:
            continue  # unparseable candidate (e.g. truncated); keep scanning
        if not isinstance(doc, dict):
            contract_error = "proposal JSON is not an object"
            continue
        if set(doc) != {"commands", "rationale"}:
            contract_error = (
                "proposal keys %s violate the exact schema {commands, rationale}; "
                "the task contract is fixed and cannot be changed"
                % sorted(doc))
            continue
        rationale = doc["rationale"]
        if type(rationale) is not str or not rationale.strip():
            contract_error = "rationale must be a non-empty string"
            continue
        entries = doc["commands"]
        if not isinstance(entries, list):
            contract_error = "commands must be a list"
            continue
        if not entries:
            contract_error = "commands list is empty"
            continue
        if len(entries) > max_commands:
            raise ProposalRejected(
                "commands list has %d entries, exceeding the bound %d"
                % (len(entries), max_commands))
        commands = []
        for index, entry in enumerate(entries):
            try:
                command = validate_command(entry)
                validate_subset(command)
            except (ValueError, TypeError) as exc:
                raise ProposalRejected("command %d rejected: %s" % (index, exc))
            commands.append(command)
        return commands, rationale
    if contract_error is not None:
        raise ProposalRejected(contract_error)
    raise MalformedProposal("no parsable JSON object found in the model reply")


class ProposalRejected(BlindSearchError):
    pass


class MalformedProposal(BlindSearchError):
    pass


def extract_json_candidates(text):
    """Accept one complete object, optionally in one JSON code fence.

    Never discard one proposed contract and silently select another object.
    """
    if not text:
        return
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4].strip()
    yield text


# ---------------------------------------------------------------------------
# Integrity freezing: Lean sources, built checkers, toolchain, controller.
# ---------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path):
    digest = hashlib.sha256()
    root = Path(path)
    for item in sorted(p for p in root.rglob("*") if not p.is_dir()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_hashes(targets):
    frozen = {}
    for name, path in targets.items():
        path = Path(path)
        if path.is_dir():
            frozen[name] = {"path": str(path), "kind": "dir",
                            "sha256": sha256_tree(path)}
        elif path.is_file():
            frozen[name] = {"path": str(path), "kind": "file",
                            "sha256": sha256_file(path)}
        else:
            frozen[name] = {"path": str(path), "kind": "missing",
                            "sha256": None}
    return frozen


def verify_hashes(frozen, targets):
    drifted = []
    for name, record in frozen.items():
        path = Path(targets[name])
        if record["kind"] == "dir" and path.is_dir():
            digest = sha256_tree(path)
        elif record["kind"] == "file" and path.is_file():
            digest = sha256_file(path)
        elif record["kind"] == "missing" and not path.exists():
            digest = None
        else:
            digest = "changed-kind-or-availability"
        if digest != record["sha256"]:
            drifted.append(name)
    if drifted:
        raise IntegrityDrift(
            "integrity check failed at %s: hashed inputs drifted: %s "
            "(aborting fail-closed)" % (stage_name[0], drifted))


stage_name = ["run"]


def default_targets(repo_root):
    root = Path(repo_root)
    return {
        "lean_sources": root / "VeriTac",
        "lean_toolchain": root / "lean-toolchain",
        "lean_manifest": root / "lake-manifest.json",
        "program_checker_source": root / "GemminiProgramMain.lean",
        "plan_checker_source": root / "GemminiMain.lean",
        "lean_built_library": root / ".lake/build/lib/lean/VeriTac",
        "checker_plan": Path(g.checker_binary_path()),
        "checker_program": Path(g.program_checker_binary_path()),
        "controller_backend": Path(g.__file__),
        "controller_encoding": Path(enc.__file__),
        "controller_certificate": Path(certificates.__file__),
        "controller_lowering": Path(lowering.__file__),
        "controller_blind_search": Path(__file__),
        "controller_diagnostics": Path(diagnostics.__file__),
    }


# ---------------------------------------------------------------------------
# Isolated model calls (fresh OpenCode session, all tools/plugins disabled).
# ---------------------------------------------------------------------------

BUILTIN_TOOLS_DISABLED = (
    "bash", "edit", "write", "read", "grep", "glob", "apply_patch", "skill",
    "todowrite", "webfetch", "websearch", "question", "task", "lsp")


def isolation_config():
    return {
        "$schema": "https://opencode.ai/config.json",
        "tools": {name: False for name in BUILTIN_TOOLS_DISABLED} | {"*": False},
        "permission": {"*": "deny"},
        "plugin": [],
        "mcp": {},
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
    }


def isolation_note():
    return (
        "Verified against OpenCode %s: OPENCODE_CONFIG_CONTENT is honored as an "
        "inline config override; tools.{\"*\": false} disables every tool, "
        "permission.{\"*\": \"deny\"} denies execution, plugin [] and --pure drop "
        "plugins, mcp {} drops MCP servers, and a live probe confirmed the model "
        "can emit no executing tool calls. Residual limitation: OpenCode merges "
        "user-global config (~/.config/opencode) and managed settings that this "
        "harness cannot un-merge, so globally declared plugins/MCP servers may "
        "still be loaded (their execution remains denied). Claim: denied tool "
        "EXECUTION in a fresh session and fresh temp cwd; NOT a hermetic sandbox."
        % OPENCODE_VERSION_PROBED)


def probe_isolation(opencode_path=None, timeout=60):
    """Re-verify at run start that the inline isolation config is honored."""
    opencode_path = opencode_path or shutil.which("opencode")
    if not opencode_path:
        return {"supported": False, "reason": "opencode executable not found on PATH"}
    env = dict(os.environ)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(isolation_config(), sort_keys=True)
    try:
        with tempfile.TemporaryDirectory(prefix="veritac-isolation-probe-") as cwd:
            proc = subprocess.run([opencode_path, "debug", "config"], env=env, cwd=cwd,
                                  capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"supported": False, "reason": "isolation probe failed: %s" % exc}
    try:
        resolved = json.loads(proc.stdout)
    except ValueError:
        return {"supported": False,
                "reason": "`opencode debug config` emitted non-JSON output"}
    if not isinstance(resolved, dict):
        return {"supported": False, "reason": "unexpected resolved config type"}
    tools = resolved.get("tools")
    permission = resolved.get("permission")
    honored = (proc.returncode == 0 and isinstance(tools, dict) and tools.get("*") is False
               and all(v is False for v in tools.values())
               and isinstance(permission, dict) and permission.get("*") == "deny"
               and all(v == "deny" for v in permission.values())
               and resolved.get("plugin") == [] and resolved.get("mcp") == {})
    return {"supported": bool(honored),
            "resolved": {key: resolved.get(key)
                         for key in ("tools", "permission", "plugin", "mcp")},
            "reason": None if honored else
            "inline isolation config was not fully honored by this OpenCode "
            "version; enforced isolation is NOT claimed for this run"}


def _default_model_runner(argv, cwd, env, timeout):
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)
    return {"returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr}


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def call_model(prompt, model, timeout, runner=None):
    """One fresh OpenCode session: `opencode run --format json --model ...`
    with the inline isolation config, in a fresh temp dir outside the repo."""
    runner = runner or _default_model_runner
    opencode_path = shutil.which("opencode")
    if not opencode_path:
        raise BlindSearchError("opencode executable not found on PATH")
    workdir = tempfile.mkdtemp(prefix="veritac-blind-model-")
    env = dict(os.environ)
    config = isolation_config()
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, sort_keys=True)
    argv = [opencode_path, "run", "--format", "json", "--model", model,
            "--pure", prompt]
    record = {"argv": argv, "cwd": workdir, "config": config,
              "cwd_policy": "fresh system-temp directory outside the repository"}
    start = time.monotonic()
    try:
        try:
            result = runner(argv, workdir, env, timeout)
        except subprocess.TimeoutExpired as exc:
            record.update(timed_out=True, returncode=None,
                          stdout=_as_text(exc.stdout), stderr=_as_text(exc.stderr),
                          error="model call timed out after %ss" % timeout)
        except OSError as exc:
            record.update(timed_out=False, returncode=None, stdout="", stderr="",
                          error="model call failed to run: %s" % exc)
        else:
            record.update(timed_out=False, returncode=result["returncode"],
                          stdout=result["stdout"], stderr=result["stderr"],
                          error=None)
    finally:
        record["duration_seconds"] = round(time.monotonic() - start, 3)
        shutil.rmtree(workdir, ignore_errors=True)
    return record


def parse_stream_events(stdout):
    """Robust parse of the --format json event stream; never fabricates usage."""
    events, unparsed, texts, tool_parts, errors = [], [], [], [], []
    usage, session_id = None, None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            unparsed.append(line)
            continue
        try:
            event = json.loads(line)
        except ValueError:
            unparsed.append(line)
            continue
        if not isinstance(event, dict):
            unparsed.append(line)
            continue
        events.append(event)
        if isinstance(event.get("sessionID"), str):
            session_id = event["sessionID"]
        kind = event.get("type")
        part = event.get("part")
        if kind == "text" and isinstance(part, dict):
            texts.append(str(part.get("text", "")))
        elif (kind in ("tool", "tool_use") or
              isinstance(part, dict) and part.get("type") == "tool"):
            tool_parts.append(part)
        elif kind == "step_finish" and isinstance(part, dict):
            usage = _merge_usage(usage, part.get("tokens"), part.get("cost"))
        elif kind == "error":
            errors.append(event.get("error"))
    return {"events": events, "unparsed_lines": unparsed,
            "text": "\n".join(texts), "tool_parts": tool_parts,
            "errors": errors, "usage": usage, "session_id": session_id}


def _merge_usage(current, tokens, cost):
    current = dict(current) if current else {}
    if isinstance(tokens, dict):
        for key in ("input", "output", "reasoning", "total"):
            value = tokens.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                current[key] = current.get(key, 0) + value
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            merged = current.setdefault("cache", {})
            for key in ("read", "write"):
                value = cache.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    merged[key] = merged.get(key, 0) + value
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        current["cost"] = current.get("cost", 0) + cost
    return current or None


# ---------------------------------------------------------------------------
# Metrics from ACTUAL commands; the fixed selection objective.
# ---------------------------------------------------------------------------

def candidate_metrics(commands):
    counts = g.modeled_counts(commands)
    return {"command_count": len(commands),
            "dma_input_bytes": counts["dma_in_elems_int8"],
            "instructions": counts["instructions"],
            "counts": counts}


def objective(metrics):
    """Fixed contract: minimize total DMA input bytes, then command count."""
    return (metrics["dma_input_bytes"], metrics["command_count"])


# ---------------------------------------------------------------------------
# Prompt: complete instruction semantics + exact schema + plan.  No baseline,
# no templates, no B-reuse hints, no repo paths, no tool execution.
# ---------------------------------------------------------------------------

def instruction_text(plan):
    m, n, k = plan["m"], plan["n"], plan["k"]
    garbage = g.GARBAGE_ADDR
    return f"""You are proposing a Gemmini tile-engine instruction stream for one fixed workload. You have no tools, no files, and no execution: your entire reply must be a single JSON object and nothing else.

Workload plan (fixed, exact; it cannot be changed, renegotiated, or restated as alternatives):
{json.dumps(plan, sort_keys=True)}

Semantics: C = A * B. A is an {m} x {k} row-major int8 matrix (element offset = row * {k} + column). B is a {k} x {n} row-major int8 matrix (element offset = row * {n} + column). C is an {m} x {n} int32 matrix (element offset = row * {n} + column). Every C element must equal the dot product of the corresponding A row and B column.

Command schema: "commands" is a list of command objects executed in order. Each command has a "kind" field plus exactly these fields:
- {{"kind": "config_ex", "dataflow": int}}: set engine dataflow. Only weight-stationary (dataflow = 1) is supported. Must precede any compute.
- {{"kind": "config_ld", "slot": int, "stride_bytes": int, "scale": number}}: configure one of the two input DMA streams; mvin slot 0 uses the slot-0 stream, mvin slot 1 the slot-1 stream. stride_bytes is the byte distance between consecutive loaded rows in DRAM. scale must be 1.0.
- {{"kind": "config_st", "stride_bytes": int}}: configure the output DMA stream; stride_bytes is the byte distance between consecutive stored rows of C (int32 elements). Must precede any mvout.
- {{"kind": "mvin", "slot": int, "buf": str, "offset": int, "spad_addr": int, "cols": int, "rows": int}}: load a rows x cols tile of A (buf = "A") or B (buf = "B") starting at the given element offset into the scratchpad starting at row spad_addr. slot must be 0 for buf "A" and 1 for buf "B". cols and rows must be 16.
- {{"kind": "preload", "bd_spad_addr": int, "out_addr": int}}: stage the B operand and output destination for the next compute. bd_spad_addr is a scratchpad row holding a B tile, or {garbage} to keep the B tile latched by the most recent compute. out_addr bit layout: bit 31 set = accumulator space (required), bit 29 set = full 32-bit C width (required), bit 30 = 1 accumulate onto / 0 overwrite the addressed rows; the low 29 bits are the starting accumulator row.
- {{"kind": "compute", "accumulated": bool, "a_spad_addr": int, "bd_spad_addr": int}}: multiply the 16x16 A tile at scratchpad row a_spad_addr by the staged B tile and write (overwrite mode) or add (accumulate mode) the product into the staged output rows. accumulated = false computes with the B tile staged by the most recent preload; accumulated = true selects the B tile latched by a previous compute and requires the most recent preload to specify GARBAGE_ADDR. This Boolean ONLY selects the B source: output addition versus overwrite is controlled independently by out_addr bit 30. A compute consumes its pending preload; every compute requires a new preload. bd_spad_addr must be {garbage}.
- {{"kind": "mvout", "buf_offset": int, "acc_addr": int, "cols": int, "rows": int}}: store a 16x16 accumulator tile (acc_addr uses the same bit layout as out_addr) to C starting at element offset buf_offset. cols and rows must be 16.
- {{"kind": "fence"}}: establish completion; include a final fence. Additional fences are permitted.

Hardware capacities fixed by the plan: the scratchpad has {plan['scratchpad_rows']} rows (scratchpad row addresses in [0, {plan['scratchpad_rows']})); the accumulator space has {plan['accumulator_rows']} rows (accumulator row indices in [0, {plan['accumulator_rows']})). Tiles are always 16 x 16 (the engine dimension). Scratchpad and accumulator tile starts must be multiples of 16, with start + 16 no greater than capacity. Scratchpad contents persist until overwritten; accumulator contents persist until overwritten. All referenced scratchpad tiles must have been loaded. Accumulate-mode compute and mvout require initialized accumulator contents; overwrite-mode compute initializes the destination without a prior load. Element values are int8 in [-128, 127]. The supported arithmetic contract is K * 16384 <= 2147483647; each accepted output contains exactly the K ordered products for its row and column, in increasing K-index order. The current validator conservatively rejects differently ordered reductions even if algebraically equivalent. Commands have sequential completion in the declared model; no asynchronous overlap semantics are assumed. Input/output buffers are fixed, disjoint, initialized/mapped by the caller, and the accelerator is exclusively owned.

Response schema (exact; no extra keys, no duplicate keys, no text outside the JSON):
{{"commands": [<command objects>], "rationale": "<non-empty string>"}}

Objective: minimize total DMA input bytes (the sum of all mvin tile bytes for A and B), then minimize the number of commands. An external checker decides correctness; a stream that does not compute the exact GEMM scores zero regardless of its traffic."""


def build_feedback(task_id, plan, receipts):
    entries = []
    for receipt in receipts:
        entries.append({
            "round": receipt["round"],
            "status": receipt["status"],
            "syntax_ok": receipt["status"] in (
                STATUS_NATIVE_ACCEPTED, STATUS_NATIVE_REJECTED),
            "error": receipt["reason"] if receipt["status"] in (
                STATUS_MODEL_MALFORMED, STATUS_PROPOSAL_REJECTED,
                STATUS_NATIVE_REJECTED) else None,
            "checker": receipt.get("checker"),
            "metrics": receipt.get("metrics"),
            "own_previous_proposal": receipt.get("proposal"),
            "advisory_diagnostic": receipt.get("advisory_diagnostic"),
        })
    return {"schema": FEEDBACK_SCHEMA, "task_id": task_id, "plan": plan,
            "objective": "minimize total DMA input bytes, then command count",
            "previous_rounds": entries,
            "instructions": "Return a new proposal in exactly the same schema "
                            "and under exactly the same rules as the original task."}


def build_prompt(plan, feedback=None):
    text = instruction_text(plan)
    if feedback is not None:
        text += ("\n\nStructured feedback from previous rounds (factual status, "
                 "checker acceptance, and metrics only):\n"
                 + json.dumps(feedback, sort_keys=True, indent=2))
    return text


# ---------------------------------------------------------------------------
# Injectable backend entry points (defaults are the real Lean-backed ones).
# ---------------------------------------------------------------------------

def _default_program_check(plan, commands):
    accepted, reason, _ = g.check_program(plan, commands)
    return accepted, reason


def _default_certifier(plan, commands, output, timeout):
    return lowering.emit_checked_candidate(plan, commands, output,
                                           timeout=timeout)


@dataclass
class RunConfig:
    task_id: str
    plan: dict
    model: str = DEFAULT_MODEL
    max_rounds: int = DEFAULT_MAX_ROUNDS
    model_timeout: float = DEFAULT_MODEL_TIMEOUT
    proof_timeout: float = DEFAULT_PROOF_TIMEOUT
    max_commands: int = DEFAULT_MAX_COMMANDS
    interpret_macs_limit: int = DEFAULT_INTERPRET_MACS_LIMIT

    def __post_init__(self):
        if type(self.task_id) is not str or not self.task_id.strip():
            raise ContractViolation("task_id must be a non-empty string")
        for name, value in (("max_rounds", self.max_rounds),
                            ("max_commands", self.max_commands),
                            ("interpret_macs_limit", self.interpret_macs_limit)):
            if type(value) is not int or value < 1:
                raise ContractViolation("%s must be a positive integer" % name)
        for name, value in (("model_timeout", self.model_timeout),
                            ("proof_timeout", self.proof_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ContractViolation("%s must be a positive number" % name)
        ok, reason = g.validate_plan(self.plan)
        if not ok:
            raise ContractViolation("plan rejected: %s" % reason)


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BlindSearch:
    """One live blind-search experiment in a fresh, never-overwritten folder."""

    def __init__(self, config, out_dir, *, repo_root=None, program_check=None,
                 certifier=None, model_runner=None):
        self.config = config
        self.out_dir = Path(out_dir)
        self.repo_root = Path(repo_root) if repo_root else _default_repo_root()
        self.program_check = program_check or _default_program_check
        self.certifier = certifier or _default_certifier
        self.model_runner = model_runner
        self.synthetic = any(value is not None for value in (program_check, certifier, model_runner))
        self.task_contract = json.dumps({"id": config.task_id, "plan": config.plan}, sort_keys=True)
        self.frozen = None
        self.receipts = []
        self.integrity_checks = []
        self.isolation = None
        self.report = None

    # -- layout -----------------------------------------------------------

    def prepare(self):
        if self.out_dir.exists():
            raise BlindSearchError(
                "refusing to overwrite existing experiment directory: %s"
                % self.out_dir)
        self.out_dir.mkdir(parents=True)
        for name in ("prompts", "model_raw", "proposals", "candidates",
                     "receipts", "feedback", "integrity", "certification"):
            (self.out_dir / name).mkdir()
        stage_name[0] = "run-start"
        self.frozen = freeze_hashes(default_targets(self.repo_root))
        missing = [name for name, value in self.frozen.items() if value["kind"] == "missing"]
        if missing:
            raise BlindSearchError("required verification inputs missing: " + str(missing))
        self._write_json("integrity/frozen_hashes.json", self.frozen)
        stage_name[0] = "isolation-probe"
        self.isolation = probe_isolation()
        self._write_json("integrity/isolation_probe.json", self.isolation)
        if not self.isolation.get("supported"):
            raise BlindSearchError("isolation preflight failed: " + str(self.isolation.get("reason")))
        self._write_json("task.json",
                         {"id": self.config.task_id, "plan": self.config.plan})
        self._write_json("run_config.json", {
            "schema": RUN_SCHEMA, "task_id": self.config.task_id,
            "plan": self.config.plan, "model": self.config.model,
            "max_rounds": self.config.max_rounds,
            "model_timeout_seconds": self.config.model_timeout,
            "proof_timeout_seconds": self.config.proof_timeout,
            "max_commands": self.config.max_commands,
            "interpret_macs_limit": self.config.interpret_macs_limit,
            "objective": "minimize total DMA input bytes, then command count",
            "isolation": {"config": isolation_config(),
                          "note": isolation_note(),
                          "enforced": bool(self.isolation.get("supported")),
                          "probe_reason": self.isolation.get("reason")},
            "started_utc": _utcnow(),
        })
        self._verify_integrity("after-prepare")
        return self.out_dir

    def _write_json(self, relpath, payload):
        path = self.out_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def _verify_integrity(self, stage):
        stage_name[0] = stage
        verify_hashes(self.frozen, default_targets(self.repo_root))
        current = json.dumps({"id": self.config.task_id, "plan": self.config.plan}, sort_keys=True)
        saved = json.dumps(strict_json_loads((self.out_dir / "task.json").read_text()), sort_keys=True)
        if current != self.task_contract or saved != self.task_contract:
            raise IntegrityDrift("task contract changed during experiment")
        self.integrity_checks.append({"stage": stage, "verified": True,
                                      "utc": _utcnow()})

    # -- rounds -------------------------------------------------------------

    def run(self):
        if self.frozen is None or not self.out_dir.exists():
            self.prepare()
        plan = self.config.plan
        for index in range(1, self.config.max_rounds + 1):
            feedback = build_feedback(self.config.task_id, plan,
                                      self.receipts) if self.receipts else None
            prompt = build_prompt(plan, feedback)
            receipt = self._run_round(index, prompt)
            self.receipts.append(receipt)
            self._write_json("receipts/round_%03d.receipt.json" % index, receipt)
        self._verify_integrity("final-certification")
        report = self._finalize()
        self._verify_integrity("report")
        report["integrity"]["final_verified"] = True
        self.report = report
        self._write_json("report.json", report)
        return report

    def _run_round(self, index, prompt):
        plan = self.config.plan
        receipt = {"round": index, "started_utc": _utcnow(),
                   "prompt_sha256": hashlib.sha256(
                       prompt.encode()).hexdigest()}
        (self.out_dir / "prompts" / ("round_%03d.prompt.txt" % index)) \
            .write_text(prompt)
        call = call_model(prompt, self.config.model,
                          self.config.model_timeout, runner=self.model_runner)
        parsed = parse_stream_events(call["stdout"])
        receipt["model"] = {
            "model": self.config.model, "argv_tail": call["argv"][1:-1],
            "duration_seconds": call["duration_seconds"],
            "returncode": call["returncode"], "timed_out": call["timed_out"],
            "error": call["error"],
            "config_sha256": hashlib.sha256(json.dumps(
                call["config"], sort_keys=True).encode()).hexdigest(),
            "cwd_policy": call["cwd_policy"],
            "stderr_tail": call["stderr"][-2000:],
        }
        receipt["events"] = {
            "count": len(parsed["events"]),
            "unparsed_count": len(parsed["unparsed_lines"]),
            "tool_part_count": len(parsed["tool_parts"]),
            "tool_parts": parsed["tool_parts"],
            "errors": parsed["errors"],
            "session_id": parsed["session_id"],
            "reply_text_chars": len(parsed["text"]),
            "reply_text_sha256": hashlib.sha256(
                parsed["text"].encode()).hexdigest(),
        }
        receipt["usage"] = parsed["usage"]  # None when the stream had none
        (self.out_dir / "model_raw" / ("round_%03d.stdout.txt" % index)) \
            .write_text(call["stdout"] or "")
        (self.out_dir / "model_raw" / ("round_%03d.stderr.txt" % index)) \
            .write_text(call["stderr"] or "")
        (self.out_dir / "model_raw" / ("round_%03d.events.json" % index)) \
            .write_text(json.dumps(parsed["events"], indent=2) + "\n")
        (self.out_dir / "model_raw" / ("round_%03d.meta.json" % index)) \
            .write_text(json.dumps(
                {"argv": call["argv"], "config": call["config"],
                 "duration_seconds": call["duration_seconds"],
                 "returncode": call["returncode"],
                 "timed_out": call["timed_out"], "error": call["error"]},
                indent=2) + "\n")

        if call["timed_out"]:
            receipt.update(status=STATUS_MODEL_TIMEOUT,
                           reason=call["error"])
            return receipt
        if call["error"] is not None or call["returncode"] != 0:
            detail = call["error"] or ("exit %d: %s" % (
                call["returncode"], (call["stderr"] or call["stdout"])
                .strip()[-400:]))
            receipt.update(status=STATUS_MODEL_PROCESS_FAILURE, reason=detail)
            return receipt

        if parsed["tool_parts"]:
            receipt.update(status="isolation_violation", reason="model emitted a tool event; proposal discarded")
            self._write_json("receipts/round_%03d.receipt.json" % index, receipt)
            raise BlindSearchError(receipt["reason"])
        if parsed["errors"]:
            receipt.update(status=STATUS_MODEL_PROCESS_FAILURE, reason="OpenCode error event: " + str(parsed["errors"]))
            return receipt

        # Validation stage: integrity must hold before anything is checked.
        self._verify_integrity("round-%03d-validation" % index)
        try:
            commands, rationale = parse_proposal(
                parsed["text"], self.config.max_commands)
        except MalformedProposal as exc:
            receipt.update(status=STATUS_MODEL_MALFORMED, reason=str(exc))
            return receipt
        except ProposalRejected as exc:
            receipt.update(status=STATUS_PROPOSAL_REJECTED, reason=str(exc))
            return receipt

        proposal_path = self.out_dir / "proposals" / (
            "round_%03d.proposal.json" % index)
        proposal_path.write_text(json.dumps(
            {"schema": "veritac_blind_search_proposal_v1",
             "round": index,
             "commands": [enc.command_dict(c) for c in commands],
             "rationale": rationale}, indent=2) + "\n")
        commands_path = self.out_dir / "candidates" / (
            "round_%03d.commands.json" % index)
        commands_path.write_text(json.dumps(
            g.serialize_commands(commands), indent=2) + "\n")
        receipt.update(
            proposal={"commands": g.serialize_commands(commands), "rationale": rationale},
            proposal_path=str(proposal_path),
            proposal_sha256=sha256_file(proposal_path),
            commands_path=str(commands_path),
            command_sha256=lowering.command_hash(commands),
            metrics=candidate_metrics(commands))
        receipt["rationale_sha256"] = hashlib.sha256(
            rationale.encode()).hexdigest()

        check_started = time.monotonic()
        try:
            accepted, reason = self.program_check(plan, commands)
        except subprocess.TimeoutExpired:
            accepted, reason = False, "checker timed out"
        except OSError as error:
            accepted, reason = False, "checker failed to run: " + str(error)
        receipt["checker_seconds"] = round(time.monotonic() - check_started, 3)
        receipt["checker"] = {"accepted": accepted, "reason": reason}
        if not accepted:
            status = STATUS_NATIVE_REJECTED
            if "timed out" in reason.lower():
                status = "checker_timeout"
            elif any(text in reason.lower() for text in ("failed to run", "exited ", "not found", "non-json", "missing accepted", "without checker", "echoed different")):
                status = "checker_process_failure"
            receipt.update(status=status, reason=reason)
            if status == STATUS_NATIVE_REJECTED:
                receipt["advisory_diagnostic"] = diagnostics.diagnose(plan, commands)
            return receipt
        receipt.update(status=STATUS_NATIVE_ACCEPTED, reason=reason)
        receipt["interpretation"] = self._interpret(commands)
        return receipt

    def _interpret(self, commands):
        """Diagnostic-only exact interpretation on deterministic inputs;
        never substitutes for the Lean checker or the kernel certificate."""
        plan = self.config.plan
        macs = plan["m"] * plan["n"] * plan["k"]
        if macs > self.config.interpret_macs_limit:
            return {"ran": False,
                    "reason": "skipped: %d modeled MACs exceed the "
                              "interpretation bound %d"
                              % (macs, self.config.interpret_macs_limit)}
        A, B = g.deterministic_inputs(plan["m"], plan["n"], plan["k"], 0)
        try:
            C, _stats = g.run_commands(commands, plan, A, B)
        except g.GemminiError as exc:
            return {"ran": True, "matches_reference": False, "error": str(exc)}
        reference = g.scalar_matmul(A, B, plan["m"], plan["n"], plan["k"])
        return {"ran": True, "matches_reference": C == reference}

    # -- winner + certification ---------------------------------------------

    def _select_winner(self):
        best = None
        for receipt in self.receipts:
            if receipt["status"] != STATUS_NATIVE_ACCEPTED:
                continue
            key = objective(receipt["metrics"])
            if best is None or key < best[0]:
                best = (key, receipt)
        return best[1] if best else None

    def _finalize(self):
        plan = self.config.plan
        winner = self._select_winner()
        report = {
            "schema": RUN_SCHEMA,
            "evidence_kind": "synthetic_test" if self.synthetic else "live_model",
            "task_id": self.config.task_id,
            "plan": plan,
            "plan_sha256": hashlib.sha256(json.dumps(
                plan, sort_keys=True).encode()).hexdigest(),
            "model": self.config.model,
            "limits": {"max_rounds": self.config.max_rounds,
                       "model_timeout_seconds": self.config.model_timeout,
                       "proof_timeout_seconds": self.config.proof_timeout,
                       "max_commands": self.config.max_commands},
            "objective": "minimize total DMA input bytes, then command count",
            "isolation": {"config": isolation_config(),
                          "note": isolation_note(),
                          "enforced": bool(self.isolation.get("supported")),
                          "probe_reason": self.isolation.get("reason")},
            "integrity": {"frozen": self.frozen,
                          "checks": self.integrity_checks},
            "rounds": self.receipts,
            "winner": None,
            "status": "no_certified_winner",
        }
        if winner is None:
            return report
        commands = _reload_commands(Path(winner["commands_path"]))
        if lowering.command_hash(commands) != winner["command_sha256"] or candidate_metrics(commands) != winner["metrics"]:
            raise IntegrityDrift("winning candidate bytes/metrics changed before certification")
        key = objective(winner["metrics"])
        report["winner"] = {
            "round": winner["round"],
            "objective": list(key),
            "metrics": winner["metrics"],
            "command_sha256": winner["command_sha256"],
            "proposal_path": winner["proposal_path"],
            "commands_path": winner["commands_path"],
            "native_accepted": True,
            "certified": False,
        }
        cert_dir = self.out_dir / "certification" / "winner"
        try:
            cert = self.certifier(plan, commands, cert_dir,
                                         timeout=self.config.proof_timeout)
        except (OSError, ValueError) as exc:
            report["winner"].update(
                certified=False,
                reason="certification errored: %s" % exc)
            report["status"] = "certification_failed"
            return report
        report["winner"]["certification"] = cert
        report["winner"]["certification_path"] = str(cert_dir)
        if cert.get("accepted"):
            report["winner"]["certified"] = True
            report["winner"]["reason"] = cert.get("reason")
            report["status"] = "certified"
        else:
            report["winner"]["reason"] = cert.get("reason")
            report["status"] = "certification_failed"
        return report


def _reload_commands(path):
    """Reload the exact commands from the archived candidate file so the
    certified bytes correspond to the archived proposal."""
    CLASSES = COMMAND_CLASSES
    document = json.loads(Path(path).read_text())
    commands = []
    for entry in document:
        kind = entry["kind"]
        commands.append(CLASSES[kind](**{k: v for k, v in entry.items()
                                         if k != "kind"}))
    return commands


def _default_repo_root():
    return Path(__file__).resolve().parents[2]
