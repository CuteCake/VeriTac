"""Compile and execute generated GEMM C with upstream Gemmini/Spike.

Run this on the Linux host prepared by setup_spike.sh. Command counts come
from Gemmini's executed instruction trace; Spike cycles are NOT hardware time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trace_counts(stdout: str) -> dict:
    counts: Counter = Counter()
    loads: Counter = Counter()
    for line in stdout.splitlines():
        match = re.match(r"GEMMINI: mvin - (0x[0-9a-f]+) cols and (0x[0-9a-f]+) rows .* to addr (0x[0-9a-f]+)$", line)
        if match:
            cols, rows, address = (int(value, 16) for value in match.groups())
            counts["mvin"] += 1
            loads[f"0x{address:08x}"] += 1
            # int8 scratchpad transfers only; accumulator loads can have a
            # distinct configured width, so don't infer their byte count here.
            if not address & (1 << 31):
                counts["scratchpad_input_bytes"] += cols * rows
        elif line.startswith("GEMMINI: mvout - "):
            counts["mvout"] += 1
        elif line.startswith("GEMMINI: preload - "):
            counts["preload"] += 1
        elif line.startswith("GEMMINI: compute - preload = "):
            counts["compute"] += 1
    return {"operations": dict(counts), "mvin_by_local_address": dict(loads)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--success-marker", default="VERITAC_GEMMINI_PASS")
    args = parser.parse_args()
    root, source, out = args.root.resolve(), args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prefix = root / "prefix"
    upstream = root / "gemmini-rocc-tests"
    multiarch = subprocess.check_output(["dpkg-architecture", "-qDEB_HOST_MULTIARCH"], text=True).strip()
    env = dict(os.environ)
    env["PATH"] = f"{prefix / 'usr/bin'}:{prefix / 'bin'}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{prefix / 'usr/lib' / multiarch}:" + env.get("LD_LIBRARY_PATH", "")
    binary = out / "kernel"
    cc = prefix / "usr/bin/riscv64-linux-gnu-gcc-13"
    spike = prefix / "bin/spike"
    pk = root / "pk-build/pk"
    artifacts = {
        "source": source,
        "gemmini_h": upstream / "include/gemmini.h",
        "gemmini_params": upstream / "include/gemmini_params.h",
        "simulator_params": root / "libgemmini/gemmini_params.h",
        "xcustom": upstream / "rocc-software/src/xcustom.h",
        "compiler": cc,
        "spike": spike,
        "proxy_kernel": pk,
        "extension": prefix / "lib/libgemmini.so",
    }
    report = {
        "schema_version": 1,
        "backend": "upstream_gemmini_spike",
        "performance_claim": "none; functional simulator, executed command counts only",
        "success": False,
        "artifacts": {key: {"path": str(path), "sha256": sha256(path)} for key, path in artifacts.items()},
    }
    compile_cmd = [str(cc), f"--sysroot={prefix}", "-static", "-O2", "-march=rv64gc", "-mabi=lp64d", "-DBAREMETAL", "-I", str(upstream), str(source), "-lm", "-o", str(binary)]
    run_cmd = [str(spike), "--log=/dev/null", "--log-commits", "--extension=gemmini", str(pk), str(binary)]
    report.update(compile_command=compile_cmd, run_command=run_cmd)
    try:
        if sha256(artifacts["gemmini_params"]) != sha256(artifacts["simulator_params"]):
            raise RuntimeError("Gemmini header and simulator parameter files differ")
        compiled = subprocess.run(compile_cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        (out / "compile.stdout").write_text(compiled.stdout)
        (out / "compile.stderr").write_text(compiled.stderr)
        report["compile_exit_code"] = compiled.returncode
        if compiled.returncode:
            raise RuntimeError("C compilation failed; see compile.stderr")
        report["binary_sha256"] = sha256(binary)
        ran = subprocess.run(run_cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        (out / "spike.stdout").write_text(ran.stdout)
        (out / "spike.stderr").write_text(ran.stderr)
        report.update(exit_code=ran.returncode, executed_commands=trace_counts(ran.stdout),
                      stdout_sha256=sha256(out / "spike.stdout"), stderr_sha256=sha256(out / "spike.stderr"))
        if ran.returncode:
            raise RuntimeError("Spike execution failed; see spike.stderr/stdout")
        if args.success_marker not in ran.stdout.splitlines():
            raise RuntimeError("Kernel did not emit the required full-output validation marker")
        if not report["executed_commands"]["operations"].get("compute"):
            raise RuntimeError("No executed Gemmini compute instructions in trace")
        report["success"] = True
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
