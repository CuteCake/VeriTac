"""Regression tests for the VeriTac soundness fixes and verified tactics.

These exercise the Lean CLI directly. They lock in the two core soundness
guarantees that were fixed during the proof audit:

1. `tile` refuses a tile size that does not divide the loop extent (it would
   otherwise overrun the range and silently produce wrong code).
2. The annotation tactics (`parallel` / `vectorize`) and `unroll` apply and
   produce well-formed JSON, and `unroll` really does replicate the body.
"""

import json
import subprocess
from pathlib import Path

BIN = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"


def run_cli(stmt: dict, tactics: list[dict]) -> dict:
    result = subprocess.run(
        [str(BIN), "--json", json.dumps({"stmt": stmt, "tactics": tactics})],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def simple_loop(bound: int) -> dict:
    """`for i in 0..bound: C[i] = A[i]`."""
    return {
        "tag": "loop", "var": "i0",
        "lo": {"tag": "lit", "val": 0}, "hi": {"tag": "lit", "val": bound},
        "ann": "none",
        "body": {"tag": "bufWrite", "buf": "C",
                 "indices": [{"tag": "var", "name": "i0"}],
                 "val": {"tag": "bufRead", "buf": "A",
                         "indices": [{"tag": "var", "name": "i0"}]}},
    }


def test_tile_dividing_bound_applies():
    out = run_cli(simple_loop(256), [{"kind": "tile", "vars": ["i0"], "int_params": [32]}])
    assert "error" not in out, f"tile should apply: {out}"
    assert out["applied"] == 1
    assert out["stmt"]["var"] == "i0_outer"
    assert out["stmt"]["body"]["var"] == "i0_inner"


def test_tile_non_dividing_bound_rejected():
    """tile by 32 over a 100-length loop must be refused (would overrun)."""
    out = run_cli(simple_loop(100), [{"kind": "tile", "vars": ["i0"], "int_params": [32]}])
    assert "error" in out, "tile over a non-multiple extents should be rejected"
    assert "tile" in out["error"]


def test_unroll_replicates_body():
    out = run_cli(simple_loop(3), [{"kind": "unroll", "vars": ["i0"]}])
    assert "error" not in out, f"unroll should apply: {out}"
    assert out["applied"] == 1
    # The unrolled statement is a right-nested seq of 3 bufWrites, then skip.
    s = out["stmt"]
    count = 0
    while s.get("tag") == "seq":
        count += 1
        s = s["s2"]
    # count == number of seq nodes; with 3 writes there are 3 seq nodes (last ends in skip)
    assert count == 3, f"expected 3 unrolled copies, got {count}"


def test_parallel_and_vectorize_annotate():
    out = run_cli(simple_loop(64), [
        {"kind": "parallel", "vars": ["i0"]},
        {"kind": "vectorize", "vars": ["i0"]},
    ])
    assert "error" not in out, f"annotations should apply: {out}"
    assert out["applied"] == 2
    # vectorize is applied last and requires innermost loop.
    assert out["stmt"]["ann"] == "vectorize"


def test_nondivisible_tile_does_not_corrupt_output():
    """The rejection is a hard error, not a silent fallback to an un-tiled loop
    that the caller might mistake for a *tiled* result."""
    out = run_cli(simple_loop(100), [{"kind": "tile", "vars": ["i0"], "int_params": [32]}])
    assert out.get("stmt") is None
    assert "applied" not in out


if __name__ == "__main__":
    test_tile_dividing_bound_applies(); print("PASS: tile (dividing) applies")
    test_tile_non_dividing_bound_rejected(); print("PASS: tile (non-dividing) rejected")
    test_unroll_replicates_body(); print("PASS: unroll replicates body")
    test_parallel_and_vectorize_annotate(); print("PASS: parallel+vectorize annotate")
    test_nondivisible_tile_does_not_corrupt_output(); print("PASS: non-dividing tile hard-rejected")
    print("All soundness tests passed!")
