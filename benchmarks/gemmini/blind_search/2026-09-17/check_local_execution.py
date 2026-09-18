"""Local regression evidence, explicitly separate from upstream Spike.

Interpret the actual RV64 body using the independent test interpreter, compare
its observed operand packets with pinned macro operands, and run the command
model against scalar GEMM. This does not establish physical conformance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm import blind_search as b
from tests.test_gemmini_machine_bytes import execute_register_program, expected_packets


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(run):
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    winner = report.get("winner")
    if report.get("status") != "certified" or not winner or not winner.get("certified"):
        raise ValueError("certified winner required")
    certification = winner["certification"]
    request = json.loads(Path(certification["request"]).read_text())
    proof = certification["kernel_certificate"]
    code = Path(proof["binary"]).read_bytes()
    if digest(code) != proof["binary_sha256"] or code.hex() != request["encoding"]["bytes_hex"]:
        raise ValueError("byte binding changed")
    plan = report["plan"]
    if request["plan"] != plan:
        raise ValueError("task binding changed")
    commands = [b.validate_command(c) for c in request["commands"]]
    actual = execute_register_program(code)
    match = actual == expected_packets(commands, request["encoding"]["bases"])
    result = {"scope": "local independent RV64 operand interpreter plus Python command model; NOT Spike or silicon",
              "binary_sha256": digest(code), "operand_packets_match": match,
              "packets_observed": len(actual), "cases": [], "success": match,
              "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in (
                  "tests/test_gemmini_machine_bytes.py", "specializations/gemmini_gemm/backend.py")}}
    for mode, seed in (("zero", 0), ("max", 0), ("min", 0), ("minmax", 0), ("lcg", 0), ("lcg", 917)):
        a, bb = g.deterministic_inputs(plan["m"], plan["n"], plan["k"], seed, mode)
        actual_c, stats = g.run_commands(commands, plan, a, bb)
        expected = g.scalar_matmul(a, bb, plan["m"], plan["n"], plan["k"])
        ok = actual_c == expected
        result["cases"].append({"mode": mode, "seed": seed, "all_outputs_match": ok,
                                "output_sha256": digest(json.dumps(actual_c).encode()), "stats": stats})
        result["success"] &= ok
    output = run / "local_execution.json"
    if output.exists():
        raise ValueError("refusing to overwrite local execution evidence")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run": str(run), "success": result["success"]}))
    return result["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    raise SystemExit(0 if check(parser.parse_args().run) else 1)
