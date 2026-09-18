#!/usr/bin/env python3
"""Live blind-search harness for Gemmini GEMM programs (prepare/run/report).

Runs bounded rounds of fresh, tool-isolated OpenCode sessions that propose
Gemmini command streams for one fixed task {id, plan}, validates each proposal
against the exact contract, filters with the Lean program checker, and
kernel-certifies only the final winner via emit_checked_candidate.  Standard
library only.

Exit codes: 0 = certified winner; 3 = completed without a certified winner;
4 = integrity drift abort; 2 = contract/usage error.

Examples:
  python3 examples/gemmini_blind_search.py prepare --task task.json
  python3 examples/gemmini_blind_search.py run --task task.json --rounds 3
  python3 examples/gemmini_blind_search.py report --experiment <dir>
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm import blind_search as blind

DEFAULT_OUT_ROOT = ROOT / ".lake" / "blind_search"


def fresh_experiment_dir(out_root, task_id):
    out_root = Path(out_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(64):
        suffix = "".join(random.choice(
            "0123456789abcdef") for _ in range(6))
        candidate = out_root / ("%s-%s-%s" % (task_id, stamp, suffix))
        if not candidate.exists():
            return candidate
    raise blind.BlindSearchError("could not allocate a fresh experiment folder")


def build_config(args):
    task = blind.load_task(args.task)
    return blind.RunConfig(
        task_id=task["id"], plan=task["plan"], model=args.model,
        max_rounds=args.rounds, model_timeout=args.model_timeout,
        proof_timeout=args.proof_timeout, max_commands=args.max_commands,
        proposal_format=args.proposal_format)


def make_controller(args, out_dir):
    config = build_config(args)
    if args.out is not None:
        out_dir = Path(args.out)
    return blind.BlindSearch(config, out_dir)


def cmd_prepare(args):
    out_dir = (Path(args.out) if args.out is not None
               else fresh_experiment_dir(DEFAULT_OUT_ROOT, blind.load_task(args.task)["id"]))
    controller = make_controller(args, out_dir)
    controller.prepare()
    print(json.dumps({
        "experiment": str(controller.out_dir),
        "task_id": controller.config.task_id,
        "frozen_hashes": sorted(controller.frozen),
        "isolation_enforced": bool(controller.isolation.get("supported")),
        "note": "inspection snapshot only; `run` creates a separate fresh experiment folder",
    }, indent=2))
    return 0


def cmd_run(args):
    out_dir = (Path(args.out) if args.out is not None
               else fresh_experiment_dir(DEFAULT_OUT_ROOT, blind.load_task(args.task)["id"]))
    controller = make_controller(args, out_dir)
    try:
        report = controller.run()
    except blind.IntegrityDrift as exc:
        if controller.out_dir.exists():
            controller._write_json("abort.json", {"status": "integrity_drift", "error": str(exc)})
        print("ABORT: %s" % exc, file=sys.stderr)
        return 4
    except blind.BlindSearchError as exc:
        if controller.out_dir.exists():
            controller._write_json("abort.json", {"status": "controller_failure", "error": str(exc)})
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    winner = report["winner"]
    summary = {
        "experiment": str(controller.out_dir),
        "task_id": report["task_id"],
        "status": report["status"],
        "rounds": [{"round": r["round"], "status": r["status"]}
                   for r in report["rounds"]],
        "isolation_enforced": report["isolation"]["enforced"],
        "winner": None if winner is None else {
            "round": winner["round"], "certified": winner["certified"],
            "objective": winner["objective"],
            "certification_path": winner.get("certification_path")},
    }
    print(json.dumps(summary, indent=2))
    if report["status"] == "certified":
        return 0
    return 3


def cmd_report(args):
    experiment = Path(args.experiment)
    report_path = experiment / "report.json"
    if not report_path.is_file():
        print("ERROR: no report.json under %s" % experiment, file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    frozen = report["integrity"]["frozen"]
    targets = {name: Path(record["path"]) for name, record in frozen.items()}
    try:
        blind.stage_name[0] = "report"
        blind.verify_hashes(frozen, targets)
        integrity = {"verified": True, "stage": "report"}
    except blind.IntegrityDrift as exc:
        integrity = {"verified": False, "error": str(exc)}
    print(json.dumps({"experiment": str(experiment),
                      "status": report["status"],
                      "task_id": report["task_id"],
                      "winner": report["winner"],
                      "integrity": integrity,
                      "isolation": report["isolation"]}, indent=2))
    return 0 if integrity["verified"] else 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(child):
        child.add_argument("--task", type=Path, required=True,
                           help="task JSON with exactly {id, plan}")
        child.add_argument("--out", type=Path, default=None,
                           help="experiment directory (default: fresh unique "
                                "folder under %s; never overwrites)" % DEFAULT_OUT_ROOT)
        child.add_argument("--model", default=blind.DEFAULT_MODEL,
                           help="OpenCode model id (default: %(default)s)")
        child.add_argument("--rounds", type=int, default=blind.DEFAULT_MAX_ROUNDS)
        child.add_argument("--model-timeout", type=float,
                           default=blind.DEFAULT_MODEL_TIMEOUT)
        child.add_argument("--proof-timeout", type=float,
                           default=blind.DEFAULT_PROOF_TIMEOUT)
        child.add_argument("--max-commands", type=int,
                           default=blind.DEFAULT_MAX_COMMANDS)
        child.add_argument("--proposal-format", choices=("flat", "compact"), default="flat")

    add_common(sub.add_parser("prepare", help="freeze hashes + isolate probe"))
    add_common(sub.add_parser("run", help="full live search"))
    report = sub.add_parser("report", help="print the final report")
    report.add_argument("--experiment", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command in ("prepare", "run") and args.rounds < 1:
        parser.error("--rounds must be >= 1")
    try:
        return {"prepare": cmd_prepare, "run": cmd_run,
                "report": cmd_report}[args.command](args)
    except (blind.BlindSearchError, OSError, ValueError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
