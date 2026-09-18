"""Runner for the emitted Metalium protocol bundle.

Pipeline: parse task/proposal (native checker protocol.py) -> emit bundle ->
CPU reference (Python mirror of input_gen.hpp) -> optional build (CMake,
package or add_subdirectory mode) -> optional run -> checksum cross-check of
device outputs against the CPU reference -> results.json + logs.

Exit codes: 0 verified OK; 2 protocol rejected; 3 build environment
unavailable (or --emit-only not requested and tt-metal not locatable);
4 build failed; 5 run failed; 6 checksum/verify mismatch; 7 internal error.

ttsim results are recorded as runtime evidence only: never treated as proof
(the formal authority is the Lean worker) and never as a timing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from specializations.tenstorrent_protocol import protocol
from specializations.tenstorrent_protocol.runtime import emit_cpp

SEED = 0x12345678

EXIT_OK = 0
EXIT_PROTOCOL = 2
EXIT_NO_ENV = 3
EXIT_BUILD = 4
EXIT_RUN = 5
EXIT_VERIFY = 6
EXIT_INTERNAL = 7


# ---------------------------------------------------------------------------
# CPU reference: byte-exact Python mirror of input_gen.hpp
# ---------------------------------------------------------------------------

def fill_input(n_bytes: int, seed: int = SEED, pattern: str = "lcg") -> bytes:
    if pattern == "zero": return bytes(n_bytes)
    if pattern == "ones": return bytes([255]) * n_bytes
    if pattern == "alternating": return bytes(170 if i % 2 == 0 else 85 for i in range(n_bytes))
    if pattern != "lcg": raise ValueError("unknown input pattern")
    x = seed & 0xFFFFFFFF
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


def fnv1a64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def compute_reference(task: protocol.Task, seed: int = SEED, pattern: str = "lcg") -> dict:
    """Deterministic full input and the exact expected output mapping
    (output page d == input page expected[d])."""
    input_bytes = fill_input(task.input_pages * task.page_bytes, seed, pattern)
    pages = [input_bytes[i * task.page_bytes : (i + 1) * task.page_bytes]
             for i in range(task.input_pages)]
    expected_pages = [pages[src] for src in task.expected]
    expected_output = b"".join(expected_pages)
    return {
        "input": input_bytes,
        "expected_output": expected_output,
        "input_checksum": format(fnv1a64(input_bytes), "016x"),
        "expected_output_checksum": format(fnv1a64(expected_output), "016x"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, cwd=None, env=None, timeout=1800):
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, time.time() - t0


def locate_tt_metal(args) -> dict | None:
    """Locate a locally built tt-metal. Never touches the network."""
    home = args.tt_metal_home or os.environ.get("TT_METAL_HOME")
    if not home or not Path(home).is_dir():
        return None
    info = {"home": str(Path(home).resolve()), "mode": None, "prefix": None,
            "source": None}
    source = args.tt_metal_source or os.environ.get("TT_METAL_SOURCE") or home
    if args.mode == "package":
        info["mode"] = "package"
    elif args.mode == "add_subdirectory" or args.tt_metal_source or (
            args.mode == "auto" and not _has_package_config(home)):
        info["mode"] = "add_subdirectory"
        info["source"] = str(Path(source).resolve())
        if not (Path(source) / "CMakeLists.txt").is_file():
            return None
    else:
        info["mode"] = "package"
    if info["mode"] == "package":
        prefix = args.cmake_prefix or os.environ.get("TT_METAL_INSTALL_PREFIX")
        if prefix:
            info["prefix"] = prefix
        else:
            for c in (Path(home) / "install", Path(home) / "build"):
                if _has_package_config(str(c)):
                    info["prefix"] = str(c)
                    break
        if not info["prefix"]:
            return None
    return info


def _has_package_config(home: str) -> bool:
    for sub in ("build", "install", "."):
        for libdir in ("lib", "lib64"):
            for package in ("tt-metalium", "TT-Metalium"):
                path = Path(home) / sub / libdir / "cmake" / package
                if path.is_dir() and (any(path.glob("*config.cmake")) or any(path.glob("*Config.cmake"))):
                    return True
    return False


def _parse_checksums(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        for key in ("INPUT_CHECKSUM", "EXPECTED_OUTPUT_CHECKSUM", "OUTPUT_CHECKSUM"):
            if line.startswith(key + " "):
                out[key] = line.split()[1]
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args) -> int:
    report = {
        "schema": "veritac-tenstorrent-runtime-results-v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "internal-error",
        "exit_code": EXIT_INTERNAL,
        "ttsim_disclaimer": "runtime evidence only; not proof, not timing",
    }
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    src_dir = out_dir / "src"
    try:
        from specializations.tenstorrent_protocol.search import strict_json_loads
        from specializations.tenstorrent_protocol.adapter import validate_task
        task = protocol.parse_task(validate_task(strict_json_loads(Path(args.task).read_text())))
        proposal = protocol.parse_proposal(strict_json_loads(Path(args.proposal).read_text()))
    except (protocol.ProtocolError, ValueError) as e:
        report.update(status="protocol-rejected", reason=str(e), exit_code=EXIT_PROTOCOL)
        _write_report(out_dir, report)
        print(f"protocol rejected: {e}", file=sys.stderr)
        return EXIT_PROTOCOL

    report["task"] = emit_cpp.task_to_dict(task)
    report["proposal_rationale"] = proposal.rationale
    report["program_len"] = len(proposal.program)

    checker = protocol.check(task, proposal)
    report["protocol_check"] = checker.to_dict()
    if checker.status != protocol.STATUS_ACCEPTED:
        report.update(status="protocol-rejected", exit_code=EXIT_PROTOCOL)
        _write_report(out_dir, report)
        print(f"protocol check: {checker.status}: {checker.reason}", file=sys.stderr)
        return EXIT_PROTOCOL

    # Emit bundle.
    src_dir.mkdir(parents=True, exist_ok=True)
    bundle = emit_cpp.emit_bundle(task, proposal)
    artifacts = {}
    for rel, content in bundle.items():
        p = src_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        artifacts[rel] = {"sha256": _sha256(p), "bytes": len(content)}
    report["emitted_artifacts"] = artifacts

    # CPU reference (Python mirror of input_gen.hpp).
    seed = getattr(args, "seed", SEED)
    pattern = getattr(args, "pattern", "lcg")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32")
    report["input_seed"], report["input_pattern"] = seed, pattern
    ref = compute_reference(task, seed, pattern)
    report["cpu_reference"] = {
        "input_sha256": hashlib.sha256(ref["input"]).hexdigest(),
        "expected_output_sha256": hashlib.sha256(ref["expected_output"]).hexdigest(),
        "input_checksum_fnv1a64": ref["input_checksum"],
        "expected_output_checksum_fnv1a64": ref["expected_output_checksum"],
        "input_bytes": task.input_pages * task.page_bytes,
        "output_bytes": len(task.expected) * task.page_bytes,
    }

    if args.emit_only:
        report.update(status="emitted-not-run", exit_code=EXIT_OK)
        _write_report(out_dir, report)
        print(f"emitted bundle at {src_dir} (not built/run)")
        return EXIT_OK

    env_info = locate_tt_metal(args)
    if env_info is None:
        report.update(
            status="not-run",
            reason="no locally built tt-metal found; pass --tt-metal-home/--tt-metal-source "
                   "or set TT_METAL_HOME (controller owns installation)",
            exit_code=EXIT_NO_ENV)
        _write_report(out_dir, report)
        print("no tt-metal build environment; bundle emitted, nothing built/run",
              file=sys.stderr)
        return EXIT_NO_ENV
    report["tt_metal_env"] = env_info

    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cmd = ["cmake", "-S", str(src_dir), "-B", str(build_dir)]
    if env_info["mode"] == "add_subdirectory":
        cmake_cmd += [f"-DTT_METAL_SOURCE={env_info['source']}"]
    else:
        cmake_cmd += [f"-DCMAKE_PREFIX_PATH={env_info['prefix']}"]
    proc, dt = _run(cmake_cmd)
    report["cmake_configure"] = {"cmd": cmake_cmd, "exit": proc.returncode,
                                 "seconds": round(dt, 1), "log": proc.stdout[-20000:]}
    if proc.returncode != 0:
        report.update(status="build-failed", exit_code=EXIT_BUILD)
        _write_report(out_dir, report)
        return EXIT_BUILD

    proc, dt = _run(["cmake", "--build", str(build_dir), "-j", str(args.jobs)])
    report["cmake_build"] = {"cmd": ["cmake", "--build", "<build>", "-j", str(args.jobs)],
                             "exit": proc.returncode, "seconds": round(dt, 1),
                             "log": proc.stdout[-20000:]}
    if proc.returncode != 0:
        report.update(status="build-failed", exit_code=EXIT_BUILD)
        _write_report(out_dir, report)
        return EXIT_BUILD

    binary = build_dir / "veritac_tt_protocol"
    if not binary.is_file():
        report.update(status="build-failed",
                      reason="binary veritac_tt_protocol not produced",
                      exit_code=EXIT_BUILD)
        _write_report(out_dir, report)
        return EXIT_BUILD
    report["binary_sha256"] = _sha256(binary)

    run_env = dict(os.environ)
    run_env.setdefault("TT_METAL_HOME", env_info["home"])
    run_env.setdefault("TT_METAL_RUNTIME_ROOT", env_info["home"])
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        run_env[k] = v
    proc, dt = _run([str(binary), str(seed), pattern], cwd=out_dir, env=run_env, timeout=args.timeout)
    report["run"] = {"exit": proc.returncode, "seconds": round(dt, 1),
                     "stdout": proc.stdout[-20000:], "env_passthrough": args.env or []}
    (out_dir / "run.log").write_text(proc.stdout)
    if proc.returncode != 0:
        report.update(status="run-failed", exit_code=EXIT_RUN)
        _write_report(out_dir, report)
        return EXIT_RUN

    # Cross-check device-reported checksums against the CPU reference.
    sums = _parse_checksums(proc.stdout)
    report["device_checksums"] = sums
    checks = {
        "INPUT_CHECKSUM": ref["input_checksum"],
        "EXPECTED_OUTPUT_CHECKSUM": ref["expected_output_checksum"],
        "OUTPUT_CHECKSUM": ref["expected_output_checksum"],
    }
    mismatches = {k: (sums.get(k), v) for k, v in checks.items() if sums.get(k) != v}
    raw_output = out_dir / "output.bin"
    full_match = raw_output.is_file() and raw_output.read_bytes() == ref["expected_output"]
    report["full_output_matches_python_reference"] = full_match
    report["output_sha256"] = _sha256(raw_output) if raw_output.is_file() else None
    if mismatches or "VERIFY OK" not in proc.stdout or not full_match:
        report["checksum_mismatches"] = mismatches
        report.update(status="verify-mismatch", exit_code=EXIT_VERIFY)
        _write_report(out_dir, report)
        return EXIT_VERIFY

    report.update(status="verified", exit_code=EXIT_OK,
                  output_checksum_fnv1a64=sums["OUTPUT_CHECKSUM"])
    report["note"] = (
        "Device outputs bit-match the CPU reference on the executed schedule. "
        "This is runtime evidence only: not a Lean proof, not a ttsim-proof, "
        "and not a performance/timing claim.")
    _write_report(out_dir, report)
    print(f"verified: outputs match CPU reference; results in {out_dir / 'results.json'}")
    return EXIT_OK


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="task JSON path")
    p.add_argument("--proposal", required=True, help="proposal JSON path")
    p.add_argument("--out", required=True, help="artifact output directory")
    p.add_argument("--emit-only", action="store_true",
                   help="emit bundle + CPU reference only; skip build/run")
    p.add_argument("--tt-metal-home", default=None,
                   help="locally built tt-metal tree (default: $TT_METAL_HOME)")
    p.add_argument("--tt-metal-source", default=None,
                   help="force add_subdirectory mode against this checkout")
    p.add_argument("--cmake-prefix", default=None,
                   help="CMAKE_PREFIX_PATH for find_package(TT-Metalium)")
    p.add_argument("--mode", choices=["auto", "package", "add_subdirectory"],
                   default="auto")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=lambda x: int(x, 0), default=SEED)
    p.add_argument("--pattern", choices=["lcg", "zero", "ones", "alternating"], default="lcg")
    p.add_argument("--timeout", type=int, default=600,
                   help="binary run timeout seconds")
    p.add_argument("--env", action="append", metavar="K=V",
                   help="extra env for the binary (repeatable), e.g. ttsim hooks")
    args = p.parse_args(argv)
    try:
        return run(args)
    except Exception as e:  # pragma: no cover - safety net
        print(f"internal error: {e!r}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
