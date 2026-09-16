#!/usr/bin/env python3
"""Replay the recorded, real OpenCode capacity challenge (no model call)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CodeGen import gemmini as g
from CodeGen.gemmini_lowering import validate_rewrite

ARCHIVE = ROOT / "benchmarks/gemmini/ai_repair/2026-09-16"
CLASSES = {c.kind: c for c in (
    g.ConfigEx, g.ConfigLd, g.ConfigSt, g.Mvin, g.Preload, g.Compute, g.Mvout, g.Fence)}


def commands(document):
    return [CLASSES[c["kind"]](**{k: v for k, v in c.items() if k != "kind"})
            for c in document["commands"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".lake/gemmini_ai_replay")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-only", action="store_true",
                        help="Replay native decisions only; skip new kernel certificates.")
    args = parser.parse_args()
    trace = json.loads((ARCHIVE / "trace.json").read_text())
    for name, digest in trace["files_sha256"].items():
        if hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"archived interaction hash mismatch: {name}")
    baseline, proposal, repair = [
        json.loads((ARCHIVE / f"{name}.json").read_text())
        for name in ("baseline", "proposal", "repair")]
    if not baseline["plan"] == proposal["plan"] == repair["plan"]:
        raise RuntimeError("proposal changed the workload or target capacities")
    decisions = {}
    for name, doc, expected in (("baseline", baseline, True),
                                ("proposal", proposal, False), ("repair", repair, True)):
        accepted, reason, _ = g.check_program(doc["plan"], commands(doc))
        decisions[name] = {"accepted": accepted, "reason": reason}
        if accepted != expected:
            raise RuntimeError(f"unexpected {name} decision: {reason}")
    if not args.check_only:
        edge = validate_rewrite(baseline["plan"], commands(baseline), commands(repair),
                                args.out, timeout=args.timeout)
        if not edge["accepted"]:
            raise RuntimeError(f"kernel replay failed; inspect {args.out}")
        decisions["kernel_edge"] = {"accepted": True, "path": str(args.out / "edge.json")}
    print(json.dumps({"recorded_interaction": str(ARCHIVE / "visible-transcript.json"),
                      "target_unchanged": True, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
