"""Bounded live proposal search. Protocol certificates, never machine-code proofs.

The transport is reused from the existing Gemmini experiment; only its generic
OpenCode isolation/event helpers are used. Proposal/checker adapters are local.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from specializations.gemmini_gemm.blind_search import (
    call_model, parse_stream_events, probe_isolation, strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = list((ROOT / "VeriTac/Tenstorrent").glob("*.lean"))
    paths += list((ROOT / "specializations/tenstorrent_protocol").glob("*.py"))
    paths += list((ROOT / ".lake/build/lib/lean/VeriTac/Tenstorrent").glob("*.olean"))
    paths += [ROOT / p for p in ("lean-toolchain", "lake-manifest.json", "lakefile.lean",
                               "specializations/gemmini_gemm/blind_search.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def extract_proposal(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
        if not match:
            raise ValueError("reply must be one JSON object or one JSON fenced block")
        text = match.group(1)
    if len(text) > 100_000:
        raise ValueError("reply too large")
    obj = strict_json_loads(text)
    if not isinstance(obj, dict) or set(obj) != {"program", "rationale"}:
        raise ValueError("expected exactly program and rationale")
    if not isinstance(obj["rationale"], str) or len(obj["rationale"]) > 20000:
        raise ValueError("rationale must be a bounded string")
    if not isinstance(obj["program"], list) or not 1 <= len(obj["program"]) <= 128:
        raise ValueError("program must contain 1..128 instructions")
    return obj


def make_prompt(task, contract, history):
    return (
        "Design a correct restricted Blackhole asynchronous page-copy protocol. "
        "The task and acceptance rules are fixed. You have no tools. "
        "Return only JSON {\"program\":[...],\"rationale\":\"...\"}. "
        "Use 1..128 flat instructions with the exact op fields below. "
        "There is no optimized reference implementation. "
        "Minimize explicit wait_reads+wait_writes count, then command count, "
        "then peak live slots, while meeting the exact output specification. "
        "Only correct programs count. All allowed asynchronous completion "
        "orders must be safe, not just one favorable timing.\n\n"
        + contract + "\n\nTASK:\n" + json.dumps(task, sort_keys=True)
        + "\n\nYOUR PREVIOUS PROPOSALS AND FACTUAL FEEDBACK:\n"
        + json.dumps(history, sort_keys=True)
    )


def run(task, output, contract, *, rounds=3, model="dgxspark-glm/glm-5.3-flash",
        model_timeout=600, proof_timeout=600, model_call=None):
    from .adapter import validate_task, evaluate, certify
    task = validate_task(task)
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("rounds must be 1..8")
    if not isinstance(model, str) or not model:
        raise ValueError("model required")
    for limit in (model_timeout, proof_timeout):
        if type(limit) not in (int, float) or not 0 < limit <= 3600:
            raise ValueError("timeouts must be positive and at most 3600 seconds")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Make task immutable to caller mutation and check code integrity before each call.
    task = json.loads(json.dumps(task))
    task_hash = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    frozen = fingerprint()
    config = {"task": task, "task_sha256": task_hash, "rounds": rounds,
              "model": model, "model_timeout": model_timeout, "proof_timeout": proof_timeout,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "restricted protocol IR; NOT final machine-code proof",
              "synthetic_model": model_call is not None,
              "contract_sha256": hashlib.sha256(contract.encode()).hexdigest()}
    dump(output / "config.json", config)
    dump(output / "fingerprint.json", frozen)
    (output / "contract.txt").write_text(contract)
    isolation = probe_isolation() if model_call is None else {"supported": True, "synthetic": True}
    dump(output / "isolation.json", isolation)
    if not isolation["supported"]:
        raise RuntimeError("model isolation unavailable")
    history, accepted = [], []
    for index in range(1, rounds + 1):
        if fingerprint() != frozen:
            raise RuntimeError("source/build fingerprint changed during registered search")
        folder = output / f"round_{index:03d}"
        folder.mkdir()
        prompt = make_prompt(task, contract, history)
        (folder / "prompt.txt").write_text(prompt)
        record = (model_call or call_model)(prompt, model, model_timeout)
        (folder / "stdout.jsonl").write_text(record.get("stdout", ""))
        (folder / "stderr.txt").write_text(record.get("stderr", ""))
        parsed = parse_stream_events(record.get("stdout", ""))
        receipt = {"round": index, "model": {k: v for k, v in record.items()
                   if k not in ("stdout", "stderr", "argv", "config")},
                   "usage": parsed["usage"], "proposal": None,
                   "tool_count": len(parsed["tool_parts"])}
        if record.get("timed_out"):
            receipt.update(status="model_timeout", reason=record.get("error"))
        elif record.get("returncode") != 0 or parsed["errors"] or parsed["unparsed_lines"]:
            receipt.update(status="model_failure", reason="model process/event stream failure")
        elif parsed["tool_parts"]:
            receipt.update(status="model_failure", reason="tool invocation in tool-denied proposal")
        else:
            try:
                proposal = extract_proposal(parsed["text"])
                receipt["proposal"] = proposal
                verdict = evaluate(task, proposal["program"])
                receipt.update(status="native_accepted" if verdict["accepted"] else "native_rejected",
                               verdict=verdict)
                if verdict["accepted"]:
                    accepted.append((tuple(verdict["objective"]), index, proposal["program"]))
            except (ValueError, TypeError, KeyError, RecursionError) as error:
                receipt.update(status="malformed", reason=str(error))
        dump(folder / "receipt.json", receipt)
        history.append({k: v for k, v in receipt.items() if k not in ("model", "usage", "tool_count")})
    if fingerprint() != frozen:
        raise RuntimeError("source/build fingerprint changed before certification")
    attempts, winner = [], None
    for score, index, program in sorted(accepted):
        # Reload the exact recorded proposal, avoiding any hidden in-memory repair.
        recorded = json.loads((output / f"round_{index:03d}/receipt.json").read_text())["proposal"]["program"]
        if recorded != program:
            raise RuntimeError("proposal artifact changed")
        result = certify(task, recorded, output / f"certificate_round_{index:03d}", timeout=proof_timeout)
        attempts.append({"round": index, "objective": list(score), "certificate": result})
        if result["accepted"]:
            winner = attempts[-1]
            break
    report = {"config": config, "status": "certified" if winner else "no_certified_winner",
              "rounds": history, "winner": winner, "certification_attempts": attempts,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    dump(output / "report.json", report)
    return report
