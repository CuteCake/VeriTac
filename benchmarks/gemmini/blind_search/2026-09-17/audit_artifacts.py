"""Audit completed experiment evidence without trusting winner labels alone.

This rechecks source/byte/task binding and the native predicate. It does not
substitute for the recorded successful Lean kernel proof process, nor claim
that native acceptance alone is a certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from specializations.gemmini_gemm import backend as g, blind_search as b, certificate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_run(folder, task):
    report = json.loads((folder / "report.json").read_text())
    if report["task_id"] != task["id"] or report["plan"] != task["plan"]:
        raise ValueError("task contract differs from preregistration")
    if report.get("evidence_kind") != "live_model":
        raise ValueError("synthetic evidence in live suite")
    result = {"task_id": task["id"], "status": report["status"], "byte_certified": False}
    winner = report.get("winner")
    if report["status"] != "certified":
        if winner and winner.get("certified"):
            raise ValueError("inconsistent certification flags")
        return result
    if not winner or not winner.get("certified"):
        raise ValueError("missing certified winner")
    cert = winner["certification"]
    proof = cert["kernel_certificate"]
    if not cert["accepted"] or not proof["accepted"]:
        raise ValueError("certificate was not accepted")
    request = json.loads(Path(cert["request"]).read_text())
    code = Path(proof["binary"]).read_bytes()
    if request["plan"] != task["plan"] or code.hex() != request["encoding"]["bytes_hex"]:
        raise ValueError("request/task/binary binding mismatch")
    if sha(proof["binary"]) != proof["binary_sha256"] or sha(proof["proof"]) != proof["proof_sha256"]:
        raise ValueError("certificate artifact hash mismatch")
    text = Path(proof["proof"]).read_text()
    match = re.search(r"def code : List UInt8 := (\[[0-9,\s]*\])", text)
    if not match or bytes(json.loads(match.group(1))) != code:
        raise ValueError("the theorem's byte list is not the delivered binary")
    commands = [b.validate_command(c) for c in request["commands"]]
    receipt = next(r for r in report["rounds"] if r["round"] == winner["round"])
    raw = folder / "model_raw" / ("round_%03d.stdout.txt" % winner["round"])
    response = b.parse_stream_events(raw.read_text())
    if response["tool_parts"] or response["errors"]:
        raise ValueError("winning response contains a tool or error event")
    if hashlib.sha256(response["text"].encode()).hexdigest() != receipt["events"]["reply_text_sha256"]:
        raise ValueError("original model reply changed")
    if report.get("proposal_format", "flat") == "compact":
        from specializations.gemmini_gemm.compact_protocol import parse_compact
        proposed, _ = parse_compact(response["text"], task["plan"], report["limits"]["max_commands"])
    else:
        proposed, _ = b.parse_proposal(response["text"], report["limits"]["max_commands"])
    if b.lowering.command_hash(proposed) != b.lowering.command_hash(commands):
        raise ValueError("delivered commands are not the model's actual proposal")
    if sha(winner["proposal_path"]) != receipt["proposal_sha256"]:
        raise ValueError("archived proposal changed")
    if certificate.certificate_source(task["plan"], commands, request["encoding"]) != text:
        raise ValueError("proof does not match the task/commands/bytes declaration")
    if b.lowering.command_hash(commands) != winner["command_sha256"]:
        raise ValueError("winner commands changed")
    if b.candidate_metrics(commands) != winner["metrics"]:
        raise ValueError("winner metrics changed")
    log = Path(proof["log"]).read_text()
    if "'SubmittedKernel.correct' depends on axioms:" not in log or any(
            bad in log for bad in ("sorryAx", "Lean.ofReduceBool")):
        raise ValueError("missing or unacceptable axiom audit")
    axiom_match = re.search(r"'SubmittedKernel.correct' depends on axioms:\s*\[([^\]]*)\]", log)
    if not axiom_match:
        raise ValueError("cannot parse the correctness theorem's axiom list")
    axioms = {name.strip() for name in axiom_match.group(1).split(",") if name.strip()}
    if not axioms <= {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("unexpected axioms: " + str(sorted(axioms)))
    accepted, reason, _ = g.gemmini_program_check(request)
    if not accepted:
        raise ValueError("native byte replay failed: " + reason)
    local = json.loads((folder / "local_execution.json").read_text())
    if not local["success"] or local["binary_sha256"] != proof["binary_sha256"]:
        raise ValueError("missing/mismatched local execution evidence")
    result.update(byte_certified=True, binary_sha256=proof["binary_sha256"],
                  original_model_reply_matches_delivered_program=True,
                  audited_axioms=sorted(axioms),
                  theorem_bytes_equal_delivered_bytes=True, task_and_command_binding=True,
                  native_byte_replay=True, local_execution=True)
    return result


def audit(runs):
    results, errors, pending = [], [], []
    for path in sorted((HERE / "tasks").glob("*.json")):
        task = b.load_task(path)
        folder = runs / task["id"]
        if not (folder / "report.json").is_file():
            pending.append(task["id"])
            continue
        try:
            results.append(audit_run(folder, task))
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append({"task_id": task["id"], "error": str(error)})
    return {"artifact_integrity": not errors, "complete": not pending,
            "certified_cases": sum(r["byte_certified"] for r in results),
            "checked_outcomes": results, "pending": pending, "errors": errors,
            "scope": "binding audit and native replay; kernel proofs were checked during certification; no physical conformance claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.runs)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
    raise SystemExit(0 if result["artifact_integrity"] else 1)
