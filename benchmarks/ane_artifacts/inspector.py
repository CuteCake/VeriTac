"""Read-only compiled-ANE artifact inspector (CLI).

Usage:
    python3 inspector.py inspect PATH [PATH ...] [--json]

Each PATH may be a single file or a run-directory.  Only the explicit inputs
are read.  Output is a single JSON document on stdout; diagnostics on stderr.

Exit status:
    0   inspection completed (all requested inputs parsed; unknown formats
        are reported, not treated as errors; an empty directory yields
        artifact_count 0 with status ok)
    2   one or more requested inputs were malformed / truncated / unreadable
    1   usage error
"""

import argparse
import json
import os
import sys

from aneinspector import __version__ as VERSION
from aneinspector.inspect import _InspectorError, inspect_file
from aneinspector import walk


def _status_for(result):
    status = result.get("status", "ok")
    if status in ("malformed",):
        return "malformed"
    if status in ("unreadable",):
        return "unreadable"
    if status == "unknown_format":
        return "unknown_format"
    return "ok"


def _build_report(inputs, json_out):
    report = {
        "inspector": {
            "name": "ane_artifact_inspector",
            "version": VERSION,
        },
        "inputs": inputs,
        "artifacts": [],
        "security_notes": [],
        "coverage": {
            "semantic_scope": "container_only",
            "instruction_semantics": "instruction_semantics_unmodeled",
        },
        "status": "ok",
        "errors": [],
    }

    per_file_hash_inputs = []
    hard_error = False

    for inp in inputs:
        try:
            files, notes, _ = walk.collect(inp)
        except ValueError as exc:
            report["errors"].append({"input": inp, "message": str(exc)})
            report["status"] = "error"
            hard_error = True
            continue

        report["security_notes"].extend(notes)
        if not files:
            report["artifacts"].append({
                "path": inp,
                "kind": "directory",
                "artifact_count": 0,
                "note": "no artifacts found",
            })
            continue

        dir_artifacts = []
        for f in files:
            try:
                art = inspect_file(f)
            except _InspectorError as exc:
                art = {
                    "path": f,
                    "status": "unreadable",
                    "error": {"reason": exc.reason, "detail": exc.detail},
                }
                hard_error = True
            dir_artifacts.append(art)
            st = art.get("status", "ok")
            if st == "malformed":
                hard_error = True
                report["errors"].append({
                    "input": f,
                    "reason": art.get("error", {}).get("reason"),
                    "detail": art.get("error", {}).get("detail"),
                })
            per_file_hash_inputs.append(
                "%s:%s" % (os.path.basename(f), art.get("sha256", "")))

        report["artifacts"].append({
            "path": inp,
            "kind": "file" if len(files) == 1 and os.path.isfile(inp) else "directory",
            "artifact_count": len(files),
            "artifacts": dir_artifacts,
        })

    if hard_error:
        report["status"] = "error"

    # Input fingerprint: hash of the sorted per-artifact name:sha256 lines.
    import hashlib
    digest = hashlib.sha256()
    for line in sorted(per_file_hash_inputs):
        digest.update(line.encode("utf-8") + b"\n")
    report["input_hash"] = digest.hexdigest()

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ane_artifact_inspector",
        description="Read-only bounded inspector for compiled ANE artifacts.")
    parser.add_argument("inspect", nargs="?", help="'inspect' subcommand")
    parser.add_argument("paths", nargs="*", help="explicit files or run-dirs")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON (default)")
    parser.add_argument("--version", action="store_true", help="print version")
    args = parser.parse_args(argv)

    if args.version:
        print(VERSION)
        return 0

    if args.inspect != "inspect" or not args.paths:
        parser.print_help()
        return 1

    report = _build_report(args.paths, args.json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
