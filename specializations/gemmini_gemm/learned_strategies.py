"""Reusable, untrusted generalizations of observed blind-search schedules.

The batched A-reuse proposer below was generalized by the controller AFTER the live
columns_tight round-2 artifact had been byte-certified. It is not a template
shown to the blind model, and is excluded from the preregistered old-generator
controls. It must still pass the existing checker for each concrete task.
The resident-A proposer has its separate certified discovery recorded below.
"""
from . import backend as g
from .lowering import emit_checked_candidate, gen_batched_reuse_b, command_hash
from pathlib import Path

DISCOVERY = "benchmarks/gemmini/blind_search/2026-09-17/runs/columns_tight/proposals/round_002.proposal.json"
RESIDENT_DISCOVERY = "benchmarks/gemmini/blind_search/2026-09-17/runs/columns_cache/proposals/round_001.proposal.json"
SHARED_SLOT_DISCOVERY = "benchmarks/gemmini/blind_search/2026-09-17/runs/mixed_cache/proposals/round_003.proposal.json"
SERPENTINE_DISCOVERY = "benchmarks/gemmini/blind_search/2026-09-17/runs/single_acc/proposals/round_003.proposal.json"


def gen_batched_reuse_a(plan, batch_columns):
    """Hold several output-column tiles live and share their A input tile.

    Oversized batches are not silently reduced: the unchanged byte checker
    decides whether the proposed program fits the declared accumulator.
    """
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    if plan["dim"] != 16:
        raise ValueError("instruction model requires DIM=16")
    if type(batch_columns) is not int or batch_columns <= 0:
        raise ValueError("batch_columns must be a positive integer")
    m, n, k = (plan[name] for name in ("m", "n", "k"))
    commands = g._config_preamble(plan)
    for i in range(m // 16):
        for first in range(0, n // 16, batch_columns):
            count = min(batch_columns, n // 16 - first)
            for kk in range(k // 16):
                commands.append(g.Mvin(0, "A", (i * k + kk) * 16, 0, 16, 16))
                for local in range(count):
                    j = first + local
                    commands.append(g.Mvin(1, "B", (kk * n + j) * 16, 16, 16, 16))
                    out = (0xA0000000 if kk == 0 else 0xE0000000) + local * 16
                    commands.append(g.Preload(16, out))
                    commands.append(g.Compute(False, 0, g.GARBAGE_ADDR))
                    if kk + 1 == k // 16:
                        commands.append(g.Mvout((i * n + j) * 16, 0xE0000000 + local * 16, 16, 16))
    commands.append(g.Fence())
    return commands


def certify_batched_reuse_a(plan, batch_columns, output, timeout=300):
    """The public acceptance path uses an actual concrete byte certificate."""
    commands = gen_batched_reuse_a(plan, batch_columns)
    _reserve_output(output)
    return emit_checked_candidate(plan, commands, output, timeout)


def gen_resident_a(plan):
    """Keep one output-row block's complete A reduction resident across N.

    Generalized after columns_cache round 1 was certified. Requires K+16
    scratchpad rows and only one accumulator tile; actual accesses are still
    checked rather than accepting this resource estimate as authority.
    """
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    if plan["dim"] != 16:
        raise ValueError("instruction model requires DIM=16")
    m, n, k = (plan[name] for name in ("m", "n", "k"))
    commands = g._config_preamble(plan)
    for i in range(m // 16):
        for kk in range(k // 16):
            commands.append(g.Mvin(0, "A", (i * k + kk) * 16, kk * 16, 16, 16))
        for j in range(n // 16):
            for kk in range(k // 16):
                commands.append(g.Mvin(1, "B", (kk * n + j) * 16, k, 16, 16))
                commands.append(g.Preload(k, 0xA0000000 if kk == 0 else 0xE0000000))
                commands.append(g.Compute(False, kk * 16, g.GARBAGE_ADDR))
            commands.append(g.Mvout((i * n + j) * 16, 0xE0000000, 16, 16))
    commands.append(g.Fence())
    return commands


def certify_resident_a(plan, output, timeout=300):
    commands = gen_resident_a(plan)
    _reserve_output(output)
    return emit_checked_candidate(plan, commands, output, timeout)


def gen_resident_a_batched_rows(plan, batch_rows):
    """Cache the first row block's A; share one streaming slot using latched B.

    Generalized after mixed_cache round 3 was certified. Each row batch uses
    K+16 scratchpad rows and one accumulator tile per active output-row block.
    The streaming slot first holds B, then each remaining row block's A; the
    array's retained B value survives these scratchpad overwrites.
    """
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    if plan["dim"] != 16:
        raise ValueError("instruction model requires DIM=16")
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    m, n, k = (plan[name] for name in ("m", "n", "k"))
    commands = g._config_preamble(plan)
    for first in range(0, m // 16, batch_rows):
        count = min(batch_rows, m // 16 - first)
        for kk in range(k // 16):
            commands.append(g.Mvin(0, "A", (first * k + kk) * 16, kk * 16, 16, 16))
        for j in range(n // 16):
            for kk in range(k // 16):
                mode = 0xA0000000 if kk == 0 else 0xE0000000
                commands.append(g.Mvin(1, "B", (kk * n + j) * 16, k, 16, 16))
                commands.append(g.Preload(k, mode))
                commands.append(g.Compute(False, kk * 16, g.GARBAGE_ADDR))
                for local in range(1, count):
                    i = first + local
                    commands.append(g.Mvin(0, "A", (i * k + kk) * 16, k, 16, 16))
                    commands.append(g.Preload(g.GARBAGE_ADDR, mode + local * 16))
                    commands.append(g.Compute(True, k, g.GARBAGE_ADDR))
            for local in range(count):
                commands.append(g.Mvout(((first + local) * n + j) * 16,
                                        0xE0000000 + local * 16, 16, 16))
    commands.append(g.Fence())
    return commands


def certify_resident_a_batched_rows(plan, batch_rows, output, timeout=300):
    commands = gen_resident_a_batched_rows(plan, batch_rows)
    _reserve_output(output)
    return emit_checked_candidate(plan, commands, output, timeout)


def gen_serpentine_b_cache(plan):
    """Cache B by column and retain the last row's A at each column turn.

    Generalized from the actual single_acc round-3 command stream (not its
    inaccurate cost rationale). Slots rotate as dead B tiles become A storage,
    then consumed A tiles become the next column's B storage. Uses K+16
    scratchpad rows and a single accumulator tile. Single-row tasks use the
    separately discovered resident-A pattern instead.
    """
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    if plan["dim"] != 16:
        raise ValueError("instruction model requires DIM=16")
    m, n, k = (plan[name] for name in ("m", "n", "k"))
    mt, nt, kt = m // 16, n // 16, k // 16
    if mt == 1:
        return gen_resident_a(plan)
    commands = g._config_preamble(plan)
    workspace = 0
    b_slots = [(kk + 1) * 16 for kk in range(kt)]

    def product(a_row, b_row, kk):
        commands.append(g.Preload(b_row, 0xA0000000 if kk == 0 else 0xE0000000))
        commands.append(g.Compute(False, a_row, g.GARBAGE_ADDR))

    def store(i, j):
        commands.append(g.Mvout((i * n + j) * 16, 0xE0000000, 16, 16))

    def row(i, j, retain_a):
        for kk in range(kt):
            a_row = b_slots[kk - 1] if retain_a and kk > 0 else workspace
            commands.append(g.Mvin(0, "A", (i * k + kk) * 16, a_row, 16, 16))
            product(a_row, b_slots[kk], kk)
        store(i, j)

    for kk in range(kt):
        commands.append(g.Mvin(1, "B", kk * n * 16, b_slots[kk], 16, 16))
    for i in range(mt):
        row(i, 0, i == mt - 1)
    for j in range(1, nt):
        a_positions = [workspace] + b_slots[:-1]
        next_b_slots = [b_slots[-1]] + a_positions[:-1]
        cached_row = mt - 1 if j % 2 else 0
        for kk in range(kt):
            commands.append(g.Mvin(1, "B", (kk * n + j) * 16, next_b_slots[kk], 16, 16))
            product(a_positions[kk], next_b_slots[kk], kk)
        store(cached_row, j)
        workspace, b_slots = a_positions[-1], next_b_slots
        remaining = range(mt - 2, -1, -1) if j % 2 else range(1, mt)
        last = 0 if j % 2 else mt - 1
        for i in remaining:
            row(i, j, i == last)
    commands.append(g.Fence())
    return commands


def certify_serpentine_b_cache(plan, output, timeout=300):
    commands = gen_serpentine_b_cache(plan)
    _reserve_output(output)
    return emit_checked_candidate(plan, commands, output, timeout)


def select_learned_strategy(plan):
    """Rank known recipes by actual traffic/counts, then validate in that order.

    This is an offline reuse path, not a blind model run or a global-optimality
    claim. Higher-cost recipes need no check once a cheaper one is validated.
    """
    plan = dict(plan)
    ok, reason = g.validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    proposals = [("baseline", g.gen_commands(plan))]
    row_batches = range(1, min(plan["m"] // 16, plan["accumulator_rows"] // 16) + 1)
    column_batches = range(1, min(plan["n"] // 16, plan["accumulator_rows"] // 16) + 1)
    for batch in row_batches:
        proposals.append((f"batched-b-{batch}", gen_batched_reuse_b(plan, batch)))
    for batch in column_batches:
        proposals.append((f"batched-a-{batch}", gen_batched_reuse_a(plan, batch)))
    if plan["scratchpad_rows"] >= plan["k"] + 16:
        proposals.append(("resident-a", gen_resident_a(plan)))
        for batch in row_batches:
            proposals.append((f"resident-a-batched-{batch}", gen_resident_a_batched_rows(plan, batch)))
        proposals.append(("serpentine-b", gen_serpentine_b_cache(plan)))
    unique = {}
    for name, commands in proposals:
        digest = command_hash(commands)
        if digest in unique:
            unique[digest]["aliases"].append(name)
            continue
        counts = g.modeled_counts(commands)
        unique[digest] = {"name": name, "aliases": [], "commands": commands,
                          "command_sha256": digest, "counts": counts,
                          "objective": [counts["dma_in_elems_int8"], counts["instructions"]],
                          "native_accepted": None}
    ordered = sorted(unique.values(), key=lambda candidate: candidate["objective"])
    for candidate in ordered:
        accepted, reason, _ = g.check_program(plan, candidate["commands"])
        candidate.update(native_accepted=accepted, reason=reason)
        if accepted:
            metadata = {"kind": "offline learned-recipe selection; no model call",
                        "objective": "modeled input DMA bytes, then command count",
                        "global_optimality": "not claimed", "winner": candidate["name"],
                        "candidates": [{k: v for k, v in item.items() if k != "commands"} for item in ordered]}
            return candidate["commands"], metadata
    raise ValueError("no proposed recipe was validated; this is not a proof of infeasibility")


def certify_best_learned(plan, output, timeout=300):
    import json
    plan = dict(plan)
    commands, selection = select_learned_strategy(plan)
    _reserve_output(output)
    (Path(output) / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return emit_checked_candidate(plan, commands, output, timeout)


def _reserve_output(output):
    try:
        Path(output).mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("refusing to overwrite an existing artifact directory") from error


def main(argv=None):
    import argparse
    import json
    import math
    from .blind_search import load_task, BlindSearchError
    parser = argparse.ArgumentParser(description="Reuse a discovered Gemmini schedule offline and certify its bytes.")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--strategy", choices=("auto", "batched-a", "resident-a", "resident-a-batched", "serpentine-b"), default="auto")
    parser.add_argument("--batch-columns", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not math.isfinite(args.proof_timeout) or args.proof_timeout <= 0:
        parser.error("--proof-timeout must be finite and positive")
    try:
        plan = load_task(args.task)["plan"]
        if args.strategy == "auto":
            result = certify_best_learned(plan, args.out, args.proof_timeout)
        elif args.strategy == "batched-a":
            result = certify_batched_reuse_a(plan, args.batch_columns, args.out, args.proof_timeout)
        elif args.strategy == "resident-a":
            result = certify_resident_a(plan, args.out, args.proof_timeout)
        elif args.strategy == "resident-a-batched":
            result = certify_resident_a_batched_rows(plan, args.batch_rows, args.out, args.proof_timeout)
        else:
            result = certify_serpentine_b_cache(plan, args.out, args.proof_timeout)
    except (ValueError, OSError, BlindSearchError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
