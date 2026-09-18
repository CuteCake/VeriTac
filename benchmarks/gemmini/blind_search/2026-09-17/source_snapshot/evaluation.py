"""Post-proposal controls; never supplied to the blind optimizer.

These are modeled traffic comparisons, not measured hardware performance.
An unmatched command stream is not, by itself, a novel optimization.
"""
from . import backend as g
from .lowering import gen_batched_reuse_b, command_hash


def objective(commands):
    counts = g.modeled_counts(commands)
    return (counts["dma_in_elems_int8"], counts["instructions"])


def compare_controls(plan, commands, *, timeout=60):
    """Check all controls under the candidate's exact, unchanged target."""
    accepted, reason, _ = g.check_program(plan, commands, timeout=timeout)
    if not accepted:
        raise ValueError("candidate failed independent validation: " + reason)
    controls = []
    proposals = [("baseline", g.gen_commands(plan))]
    for batch in range(1, min(plan["m"] // 16, plan["accumulator_rows"] // 16) + 1):
        proposals.append((f"existing_batched_b_{batch}", gen_batched_reuse_b(plan, batch)))
    for name, program in proposals:
        ok, why, _ = g.check_program(plan, program, timeout=timeout)
        controls.append({"name": name, "native_accepted": ok, "reason": why,
                         "counts": g.modeled_counts(program),
                         "objective": list(objective(program)),
                         "command_sha256": command_hash(program)})
    admissible = [c for c in controls if c["native_accepted"]]
    if not admissible:
        raise ValueError("no validated control")
    best = min(admissible, key=lambda c: c["objective"])
    counts = g.modeled_counts(commands)
    baseline = next(c for c in controls if c["name"] == "baseline")
    baseline_bytes = baseline["counts"]["dma_in_elems_int8"]
    best_bytes = best["counts"]["dma_in_elems_int8"]
    digest = command_hash(commands)
    return {
        "scope": "native-validated modeled traffic; separate kernel certificate required",
        "counts": counts, "objective": list(objective(commands)), "controls": controls,
        "best_existing_control": best["name"],
        "input_reduction_vs_baseline": 1 - counts["dma_in_elems_int8"] / baseline_bytes,
        "input_reduction_vs_best_existing": 1 - counts["dma_in_elems_int8"] / best_bytes,
        "beats_best_existing_objective": list(objective(commands)) < best["objective"],
        "exact_matches": [c["name"] for c in admissible if c["command_sha256"] == digest],
        "fewer_a_loads_than_existing_controls": counts["mvin_A"] < min(
            c["counts"]["mvin_A"] for c in admissible),
        "novelty": "requires manual mechanism review; a distinct hash is insufficient",
        "hardware_latency": None,
    }
