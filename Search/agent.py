"""
VeriTac Search: agent.py
Brute-force search agent that enumerates tactic sequences.
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Optional

from .interface import validate_schedule, make_matmul_stmt
from .cost_model import score_schedule


# Tactic templates for matmul optimization
TILE_SIZES = [4, 8, 16, 32, 64]
MATMUL_VARS = ["i0", "i1", "i2"]  # i, j, k


def generate_tile_tactics(var: str) -> list[dict]:
    """Generate tile tactic variants for a variable."""
    return [
        {"kind": "tile", "vars": [var], "int_params": [ts], "str_params": []}
        for ts in TILE_SIZES
    ]


def generate_reorder_tactics() -> list[dict]:
    """Generate reorder tactic variants for adjacent loop pairs."""
    pairs = [
        ("i0", "i1"),
        ("i1", "i2"),
        ("i0_outer", "i1_outer"),
        ("i0_outer", "i2"),
        ("i1_outer", "i2"),
    ]
    return [
        {"kind": "reorder", "vars": [v1, v2], "int_params": [], "str_params": []}
        for v1, v2 in pairs
    ]


def generate_annotation_tactics() -> list[dict]:
    """Generate parallel/vectorize tactics."""
    tactics = []
    for var in MATMUL_VARS + ["i0_outer", "i1_outer", "i0_inner", "i1_inner"]:
        tactics.append(
            {"kind": "parallel", "vars": [var], "int_params": [], "str_params": []}
        )
        tactics.append(
            {"kind": "vectorize", "vars": [var], "int_params": [], "str_params": []}
        )
    return tactics


def enumerate_schedules(max_length: int = 3,
                        binary: Optional[Path] = None) -> list[tuple[list[dict], float]]:
    """Enumerate valid tactic sequences up to max_length, scored by cost model.

    Returns list of (tactics, score) sorted by score (lower is better).
    """
    M, K, N = 256, 256, 256
    base_stmt = make_matmul_stmt(M, K, N)

    # Generate candidate single tactics
    single_tactics = []
    for var in MATMUL_VARS:
        single_tactics.extend(generate_tile_tactics(var))
    single_tactics.extend(generate_reorder_tactics())
    single_tactics.extend(generate_annotation_tactics())

    valid_schedules = []

    # Length 1
    for tactic in single_tactics:
        success, result_stmt, msg = validate_schedule(base_stmt, [tactic], binary)
        if success and result_stmt:
            score = score_schedule(result_stmt)
            valid_schedules.append(([tactic], score))

    print(f"Found {len(valid_schedules)} valid length-1 schedules")

    # Length 2: compose valid length-1 results with more tactics
    length1_results = [
        (tactics, result_stmt)
        for tactics in [s[0] for s in valid_schedules[:20]]  # top 20
        for success, result_stmt, _ in [validate_schedule(base_stmt, tactics, binary)]
        if success and result_stmt
    ]

    follow_up_tactics = (
        generate_reorder_tactics() +
        generate_annotation_tactics() +
        [t for var in ["i0_outer", "i1_outer", "i2_outer"]
         for t in generate_tile_tactics(var)]
    )

    for prev_tactics, prev_stmt in length1_results:
        for tactic in follow_up_tactics:
            combined = prev_tactics + [tactic]
            success, result_stmt, msg = validate_schedule(base_stmt, combined, binary)
            if success and result_stmt:
                score = score_schedule(result_stmt)
                valid_schedules.append((combined, score))

    print(f"Found {len(valid_schedules)} total valid schedules (up to length 2)")

    if max_length >= 3:
        # Length 3: extend top length-2 schedules
        length2_sorted = sorted(
            [(t, s) for t, s in valid_schedules if len(t) == 2],
            key=lambda x: x[1]
        )[:10]

        for prev_tactics, _ in length2_sorted:
            for tactic in follow_up_tactics[:20]:  # limit search
                combined = prev_tactics + [tactic]
                success, result_stmt, msg = validate_schedule(
                    base_stmt, combined, binary
                )
                if success and result_stmt:
                    score = score_schedule(result_stmt)
                    valid_schedules.append((combined, score))

        print(f"Found {len(valid_schedules)} total valid schedules (up to length 3)")

    # Sort by score
    valid_schedules.sort(key=lambda x: x[1])
    return valid_schedules


def main():
    """Run the search agent."""
    import argparse
    parser = argparse.ArgumentParser(description="VeriTac brute-force search agent")
    parser.add_argument("--max-length", type=int, default=3,
                        help="Maximum tactic sequence length")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top schedules to report")
    parser.add_argument("--binary", type=str, default=None,
                        help="Path to veritac binary")
    args = parser.parse_args()

    binary = Path(args.binary) if args.binary else None

    print(f"Searching for optimal matmul schedules (max length {args.max_length})...")
    schedules = enumerate_schedules(max_length=args.max_length, binary=binary)

    print(f"\nTop {args.top_k} schedules:")
    for i, (tactics, score) in enumerate(schedules[:args.top_k]):
        print(f"\n  #{i+1} (score: {score:.0f}):")
        for t in tactics:
            params = ""
            if t["int_params"]:
                params = f" {t['int_params']}"
            print(f"    {t['kind']} {t['vars']}{params}")


if __name__ == "__main__":
    main()
