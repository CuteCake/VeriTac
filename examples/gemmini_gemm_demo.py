"""Lean-gated Gemmini GEMM search, exact checks, and reproducible C artifacts.

Run the emitted C on upstream Spike with benchmarks/gemmini/run_spike.py.
This local demo reports modeled traffic, never hardware speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specializations.gemmini_gemm import backend as g


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_demo(m=32, n=16, k=16, scratchpad_rows=32, accumulator_rows=1024, seed=0x12345678, output=None):
    out = Path(output) if output else ROOT / ".lake/gemmini_demo" / ("run_" + uuid.uuid4().hex)
    out.mkdir(parents=True, exist_ok=False)
    search = g.optimize(m, n, k, scratchpad_rows=scratchpad_rows, accumulator_rows=accumulator_rows)
    checker = Path(g.checker_binary_path())
    report = {"schema_version": 2, "backend": "gemmini_int8_ws",
              "performance_evidence": "modeled transfer bytes and instruction counts; no latency measurement",
              "proof_boundary": ("plan acceptance = Lean plan legality only; "
                                 "program acceptance additionally requires the "
                                 "Lean program checker to prove the exact "
                                 "command stream computes the plan GEMM and "
                                 "the Lean decoder to recover it from the "
                                 "emitted executable bytes; Python/C lowering "
                                 "is untrusted"),
              "checker_sha256": g.sha256_file(checker) if checker.is_file() else None,
              "backend_sha256": g.sha256_file(Path(g.__file__)),
              "candidates": search["candidates"], "winner": None}
    patterns = [("lcg", seed), ("lcg", seed ^ 0x9E3779B9), ("max", 0),
                ("minmax", 0), ("min", 0), ("zero", 0)]
    eligible = []
    for entry in report["candidates"]:
        if not entry["valid"] or not entry["check"]["accepted"]:
            continue
        plan = entry["plan"]
        commands = g.gen_commands(plan)
        encoded = g.serialize_commands(commands)
        entry["command_sha256"] = digest(encoded)
        entry["plan_sha256"] = digest(plan)
        command_path = out / (plan["schedule"] + "_commands.json")
        command_path.write_text(json.dumps(encoded, indent=2) + "\n")
        entry["commands_path"] = str(command_path.resolve())
        entry["validation"] = []
        for mode, case_seed in patterns:
            a, b = g.deterministic_inputs(m, n, k, case_seed, mode)
            expected = g.scalar_matmul(a, b, m, n, k)
            got, stats = g.run_commands(commands, plan, a, b)
            entry["validation"].append({"mode": mode, "seed": case_seed,
                "input_sha256": digest([a, b]), "output_sha256": digest(got),
                "all_outputs_equal": got == expected, "interpreter_stats": stats})
        entry["numeric_passed"] = all(v["all_outputs_equal"] for v in entry["validation"])
        if entry["numeric_passed"]:
            # Program acceptance: emit commands + executable bytes and have the
            # Lean program checker verify the concrete artifact end to end.
            # A candidate is only 'verified' when program_accepted is true.
            entry["program"] = g.emit_program_artifacts(plan, str(out), seed=seed)
            entry["program_accepted"] = entry["program"]["program_accepted"]
            entry["program_reason"] = entry["program"]["program_reason"]
            if entry["program_accepted"]:
                eligible.append(entry)
    if eligible:
        winner = min(eligible, key=lambda e: e["objective"])
        # Fresh acceptance and replay of the chosen artifact, after search.
        accepted, reason, checked = g.gemmini_check(winner["plan"])
        if not accepted:
            raise RuntimeError("winner replay rejected: " + reason)
        a, b = g.deterministic_inputs(m, n, k, seed)
        result, _ = g.run_commands(g.gen_commands(checked), checked, a, b)
        if result != g.scalar_matmul(a, b, m, n, k):
            raise RuntimeError("winner replay failed numerical validation")
        # Program-level replay: the winner must pass program acceptance again
        # on the actual emitted commands before anything is called verified.
        commands = g.gen_commands(checked)
        prog_ok, prog_reason, cert = g.check_program(checked, commands)
        if not prog_ok:
            raise RuntimeError("winner program acceptance failed: " + prog_reason)
        report["winner"] = checked
        report["winner_source"] = winner["program"]["source"]
        report["winner_program_accepted"] = True
        from specializations.gemmini_gemm.certificate import write_and_check
        replay = write_and_check(checked, commands, winner["program"]["encoding"], out / "winner_replay")
        if not replay["accepted"]:
            raise RuntimeError("winner kernel certificate replay failed: " + replay["reason"])
        report["winner_certificate"] = cert
        report["winner_kernel_certificate"] = replay
        report["winner_binary"] = replay["binary"]
        report["winner_output_sha256"] = digest(result)
    # A deterministic rejection/repair illustration, not a live LLM transcript.
    proposed = {"m": m, "n": n, "k": k, "dim": g.DIM,
                "scratchpad_rows": scratchpad_rows, "accumulator_rows": g.DIM - 1,
                "schedule": "baseline"}
    accepted, reason, _ = g.gemmini_check(proposed)
    repaired = {**proposed, "accumulator_rows": g.DIM}
    repaired_ok, repaired_reason, _ = g.gemmini_check(repaired)
    report["illustrative_repair"] = {
            "kind": "scripted capacity rejection and repair, not live LLM",
            "proposal": proposed, "accepted": accepted, "reason": reason,
                "repair": repaired, "repair_accepted": repaired_ok, "repair_reason": repaired_reason}
    # 'verified' requires program acceptance of the actual emitted commands and
    # executable bytes; plan acceptance alone never sets it.
    report["verified"] = report.get("winner_program_accepted", False)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", 32), ("n", 16), ("k", 16)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--scratchpad-rows", type=int, default=32)
    p.add_argument("--accumulator-rows", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0x12345678)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 0 <= args.seed <= 0xFFFFFFFF:
        p.error("seed must be an unsigned 32-bit integer")
    report, path = run_demo(**vars(args))
    for c in report["candidates"]:
        counts = c["counts"]
        print(c["plan"]["schedule"], "Lean plan:", c["check"]["reason"],
              "modeled input bytes:", counts["dma_in_elems_int8"] if counts else None)
        if "program_accepted" in c:
            print("  program acceptance:", c["program_accepted"], "-",
                  (c["program_reason"] or "")[:100])
    print("Winner:", report["winner"]["schedule"] if report["winner"] else "none")
    print("Verified (program-level):", report["verified"])
    print("Report:", path.resolve())
    if not report["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
