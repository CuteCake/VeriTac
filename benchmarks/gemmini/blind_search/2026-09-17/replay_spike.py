"""Execute one certified run on the existing private Gemmini/Spike host.

Only requests, certified bytes, and the standalone runner are copied. Large
trace logs remain in a unique remote directory; the small report is retrieved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[4]
HOST = "cake@spark-379a.tail64c925.ts.net"
TOOLCHAIN = "/home/cake/.cache/veritac-gemmini"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checked(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, timeout=300, **kwargs)


def replay(run):
    run = Path(run).resolve()
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "certified" or report.get("evidence_kind") != "live_model":
        raise ValueError("requires a certified live-model report")
    cert = report["winner"]["certification"]
    proof = cert["kernel_certificate"]
    binary = Path(proof["binary"])
    request = Path(cert["request"])
    if digest(binary) != proof["binary_sha256"]:
        raise ValueError("certified binary changed")
    payload = json.loads(request.read_text())
    if payload["plan"] != report["plan"] or bytes.fromhex(payload["encoding"]["bytes_hex"]) != binary.read_bytes():
        raise ValueError("request does not bind the certified task and bytes")
    if digest(Path(proof["proof"])) != proof["proof_sha256"]:
        raise ValueError("proof source changed")
    output = run / "spike"
    output.mkdir(exist_ok=False)
    remote = TOOLCHAIN + "/blind-20260917-" + uuid.uuid4().hex
    runner = ROOT / "specializations/gemmini_gemm/runtime/run_raw_spike.py"
    metadata = {"host": HOST, "remote": remote, "binary_sha256": digest(binary),
                "request_sha256": digest(request), "runner_sha256": digest(runner),
                "scope": "functional execution and counts, not hardware latency"}
    (output / "transport.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        run_checked(["ssh", "-o", "BatchMode=yes", HOST, "mkdir " + shlex.quote(remote)])
        for source, target in ((binary, "kernel.bin"), (request, "request.json"), (runner, "run_raw_spike.py")):
            run_checked(["scp", "-q", str(source), HOST + ":" + remote + "/" + target])
        argv = ["python3", remote + "/run_raw_spike.py", "--root", TOOLCHAIN,
                "--request", remote + "/request.json", "--binary", remote + "/kernel.bin",
                "--output", remote + "/execution", "--timeout", "180"]
        command = " ".join(shlex.quote(arg) for arg in argv)
        executed = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command],
                                  text=True, capture_output=True, timeout=300)
        (output / "runner.stdout").write_text(executed.stdout)
        (output / "runner.stderr").write_text(executed.stderr)
        metadata["exit_code"] = executed.returncode
        run_checked(["scp", "-q", HOST + ":" + remote + "/execution/report.json", str(output / "report.json")])
        observed = json.loads((output / "report.json").read_text())
        metadata["success"] = bool(executed.returncode == 0 and observed.get("success")
                                   and observed.get("binary_sha256") == proof["binary_sha256"]
                                   and observed.get("plan") == report["plan"])
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        metadata.update(success=False, error=str(error))
    (output / "transport.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return metadata["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if replay(args.run) else 1)
