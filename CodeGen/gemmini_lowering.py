"""Parameterized, untrusted Gemmini proposals and checked rewrite artifacts.

Plan.schedule describes the seed plan. The validator checks the actual program;
a rewritten baseline seed may therefore use batched B reuse without changing its
workload or target capacities.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

from CodeGen import gemmini as g
from CodeGen import gemmini_encoding as encoding
from CodeGen import gemmini_certificate as certificates


def gen_batched_reuse_b(plan, batch_rows):
    ok, why = g.validate_plan(plan)
    if not ok:
        raise ValueError(why)
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive number of 16-row tiles")
    m, n, k = plan["m"], plan["n"], plan["k"]
    commands = g._config_preamble(plan)
    for j in range(n // 16):
        for first in range(0, m // 16, batch_rows):
            count = min(batch_rows, m // 16 - first)
            for kk in range(k // 16):
                commands.append(g.Mvin(1, "B", (kk * n + j) * 16, 16, 16, 16))
                for local in range(count):
                    i = first + local
                    commands.append(g.Mvin(0, "A", (i * k + kk) * 16, 0, 16, 16))
                    out = (0xA0000000 if kk == 0 else 0xE0000000) + local * 16
                    commands.append(g.Preload(16 if local == 0 else g.GARBAGE_ADDR, out))
                    commands.append(g.Compute(local != 0, 0, g.GARBAGE_ADDR))
                    if kk + 1 == k // 16:
                        commands.append(g.Mvout((i * n + j) * 16, 0xE0000000 + local * 16, 16, 16))
    commands.append(g.Fence())
    return commands


def command_hash(commands):
    data = json.dumps(g.serialize_commands(commands), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def emit_checked_candidate(plan, commands, output, timeout=300):
    """Certify these exact commands/bytes; never regenerate from a schedule tag."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = encoding.emit_encoding(plan, commands)
    request = g.build_program_request(plan, commands, artifact)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    accepted, reason, receipt = g.gemmini_program_check(request)
    result = {"plan": plan, "command_sha256": command_hash(commands),
              "request": str(output / "request.json"), "accepted": False,
              "reason": reason, "checker_receipt": receipt}
    if accepted:
        proof = certificates.write_and_check(plan, commands, artifact, output, timeout=timeout)
        result.update(accepted=proof["accepted"], reason=proof["reason"], kernel_certificate=proof)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def edge_certificate_source(plan, before, after):
    fields = (("m", "m"), ("n", "n"), ("k", "k"), ("dim", "dim"),
              ("scratchpadRows", "scratchpad_rows"), ("accumulatorRows", "accumulator_rows"))
    literal = ", ".join(f"{name} := {encoding.nat(plan[key])}" for name, key in fields)
    literal += ", schedule := ." + {"baseline": "baseline", "reuse_b": "reuseB"}[plan["schedule"]]
    source = ",\n  ".join(certificates.lean_instruction(c) for c in before)
    destination = ",\n  ".join(certificates.lean_instruction(c) for c in after)
    return """import VeriTac.Gemmini.LoweringSound
open VeriTac.Gemmini
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedRewrite
""" + f"def plan : GemminiPlan := {{ {literal} }}\n" + \
        f"def source : Program := [\n  {source}\n]\n" + \
        f"def destination : Program := [\n  {destination}\n]\n" + """
theorem sourceAccepted : Symbolic.check plan source = true := by decide +kernel
theorem destinationAccepted : Symbolic.check plan destination = true := by decide +kernel
def edge : CheckedRewrite plan source := ⟨destination, sourceAccepted, destinationAccepted⟩

theorem equivalent (a b : Nat → Int)
    (ha : ∀ i < plan.m * plan.k, VeriTac.GemminiExact.Int8 (a i))
    (hb : ∀ i < plan.k * plan.n, VeriTac.GemminiExact.Int8 (b i)) :
    ∃ before after, runProgram plan a b source = some before ∧
      runProgram plan a b destination = some after ∧
      before.drained = true ∧ after.drained = true ∧
      ∀ r < plan.m, ∀ c < plan.n,
        before.output (r * plan.n + c) = after.output (r * plan.n + c) :=
  checkedRewrite_sound plan source edge a b ha hb
#print axioms equivalent
end SubmittedRewrite
"""


def validate_rewrite(plan, before, after, output, timeout=300):
    """A replayable edge with kernel-certified byte artifacts at both endpoints."""
    output = Path(output).resolve()
    source = emit_checked_candidate(plan, before, output / "source", timeout)
    destination = emit_checked_candidate(plan, after, output / "destination", timeout)
    result = {"kind": "exact_gemm_rewrite", "plan": plan, "source": source,
              "destination": destination, "accepted": False}
    if source["accepted"] and destination["accepted"]:
        proof = output / "rewrite.lean"
        proof.write_text(edge_certificate_source(plan, before, after))
        start = time.monotonic()
        try:
            run = subprocess.run(["lake", "env", "lean", "-s", "65536", str(proof)],
                                 cwd=certificates.ROOT, capture_output=True, text=True, timeout=timeout)
            log = run.stdout + run.stderr
            result["accepted"] = (run.returncode == 0 and
                "'SubmittedRewrite.equivalent' depends on axioms:" in log and
                "sorryAx" not in log and "Lean.ofReduceBool" not in log)
        except (OSError, subprocess.TimeoutExpired) as error:
            log = str(error)
        (output / "rewrite.log").write_text(log)
        result.update(proof=str(proof), proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
                      edge_check_seconds=round(time.monotonic() - start, 3))
    (output / "edge.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
