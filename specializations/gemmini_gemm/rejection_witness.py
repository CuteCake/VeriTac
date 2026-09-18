"""Sparse counterexamples for output mismatches, with optional kernel proof.

This is separate from acceptance and was not fed to the registered experiments.
Product-order differences alone do not imply a numerical error: coefficients
are normalized before searching, and a reported counterexample is executed.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from . import backend as g, diagnostics as d, encoding, certificate


def _coefficients(terms):
    return Counter(tuple(sorted(tuple(ref) for ref in term)) for term in terms)


def find_sparse_counterexample(plan, commands, max_trials=12):
    if type(max_trials) is not int or max_trials <= 0:
        raise ValueError("max_trials must be a positive integer")
    diagnostic = d.diagnose(plan, commands)
    result = {"advisory_only": True, "diagnostic": diagnostic, "found": False}
    if diagnostic["status"] != d.STATUS_OUTPUT_MISMATCH or not diagnostic.get("facts", {}).get("written"):
        return {**result, "reason": "no supported written-output mismatch to investigate"}
    parsed_plan, _ = d._parse_plan(plan)
    parsed_commands, _ = d._parse_commands(commands)
    state = d._State()
    for instruction in parsed_commands:
        if d._step(parsed_plan, state, instruction) is not None:
            return {**result, "reason": "advisory execution inconclusive"}
    facts = diagnostic["facts"]
    row, col, cell = facts["row"], facts["col"], facts["cell_index"]
    delta = _coefficients(state.output[cell])
    delta.subtract(_coefficients(d._expected_terms(parsed_plan, row, col)))
    different = [term for term, value in delta.items() if value]
    if not different:
        return {**result, "reason": "normalized product coefficients agree; no numerical error is asserted"}
    attempted = set()
    for term in different:
        refs = tuple(sorted(set(term)))
        for enabled in [(ref,) for ref in refs] + [refs]:
            if enabled in attempted:
                continue
            if len(attempted) >= max_trials:
                return {**result, "reason": "counterexample search budget exhausted"}
            attempted.add(enabled)
            a, b = [0] * (plan["m"] * plan["k"]), [0] * (plan["k"] * plan["n"])
            for buf, index in enabled:
                (a if buf == "a" else b)[index] = 1
            try:
                actual, _ = g.run_commands(commands, plan, a, b)
            except g.GemminiError as error:
                return {**result, "reason": "numeric command model could not execute: " + str(error)}
            expected = sum(a[row * plan["k"] + kk] * b[kk * plan["n"] + col] for kk in range(plan["k"]))
            if actual[cell] != expected:
                return {**result, "found": True, "row": row, "col": col,
                        "actual": actual[cell], "expected": expected,
                        "sparse_inputs": {"default": 0, "ones": [[buf, index] for buf, index in enabled]},
                        "trials": len(attempted), "reason": "executed sparse input produces a different output"}
    return {**result, "reason": "no executed counterexample found"}


def _input_definition(name, indices):
    expression = "0"
    for index in reversed(indices):
        expression = f"if i = {index} then 1 else ({expression})"
    source = f"def {name} (i : Nat) : Int := {expression}\n"
    source += f"theorem {name}_domain (i : Nat) : VeriTac.GemminiExact.Int8 ({name} i) := by\n"
    if indices:
        source += f"  unfold {name}\n  split_ifs <;> norm_num [VeriTac.GemminiExact.Int8]\n"
    else:
        source += f"  norm_num [{name}, VeriTac.GemminiExact.Int8]\n"
    return source


def witness_source(plan, commands, artifact, witness):
    source = certificate.certificate_source(plan, commands, artifact).split("theorem symbolicAccepted", 1)[0]
    ones = witness["sparse_inputs"]["ones"]
    source += _input_definition("inputA", [index for buf, index in ones if buf == "a"])
    source += _input_definition("inputB", [index for buf, index in ones if buf == "b"])
    row, col = witness["row"], witness["col"]
    source += f"""
theorem plan_ok : checkGemminiPlan plan = true := by decide +kernel
theorem layout_ok : Encoding.checkBases bases (bufferSizes plan) = true := by decide +kernel
theorem capacities_ok : (plan.scratchpadRows ≤ 16384 && plan.accumulatorRows ≤ 1024) = true := by decide +kernel
theorem cell_in_bounds : {row} < plan.m ∧ {col} < plan.n := by decide +kernel
def mismatch : Bool :=
  match runExecutable plan bases code inputA inputB with
  | none => false
  | some s => s.drained && s.written ({row} * plan.n + {col}) &&
      decide (s.output ({row} * plan.n + {col}) = ({witness['actual']} : Int)) &&
      decide (gemm plan inputA inputB {row} {col} = ({witness['expected']} : Int)) &&
      decide (s.output ({row} * plan.n + {col}) ≠ gemm plan inputA inputB {row} {col})
theorem counterexample : mismatch = true := by decide +kernel
#print axioms counterexample
#print axioms inputA_domain
#print axioms inputB_domain
end SubmittedKernel
"""
    return source


def prove_sparse_counterexample(plan, commands, output, timeout=300):
    witness = find_sparse_counterexample(plan, commands)
    if not witness["found"]:
        return {"kernel_proved": False, "witness": witness}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = encoding.emit_encoding(plan, commands)
    raw = bytes.fromhex(artifact["bytes_hex"])
    (output / "rejected.bin").write_bytes(raw)
    source = output / "counterexample.lean"
    source.write_text(witness_source(plan, commands, artifact, witness))
    request = g.build_program_request(plan, commands, artifact)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    started = time.monotonic()
    try:
        ran = subprocess.run(["lake", "env", "lean", "-s", "65536", str(source)],
                             cwd=certificate.ROOT, capture_output=True, text=True, timeout=timeout)
        log = ran.stdout + ran.stderr
        axioms = {}
        for name in ("counterexample", "inputA_domain", "inputB_domain"):
            qualified = "SubmittedKernel." + name
            match = re.search(r"'" + re.escape(qualified) + r"' depends on axioms:\s*\[([^\]]*)\]", log)
            if match:
                axioms[name] = sorted({a.strip() for a in match.group(1).split(",") if a.strip()})
            elif "'" + qualified + "' does not depend on any axioms" in log:
                axioms[name] = []
        proved = ran.returncode == 0 and len(axioms) == 3 and all(
            set(values) <= {"propext", "Classical.choice", "Quot.sound"} for values in axioms.values())
    except (OSError, subprocess.TimeoutExpired) as error:
        proved, log, axioms = False, str(error), {}
    (output / "counterexample.log").write_text(log)
    result = {"kind": "concrete rejection witness, not an accepted kernel", "kernel_proved": proved,
              "witness": witness, "binary_sha256": hashlib.sha256(raw).hexdigest(),
              "audited_axioms": axioms,
              "proof_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "kernel_check_seconds": round(time.monotonic() - started, 3)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
