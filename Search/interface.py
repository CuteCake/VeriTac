"""
VeriTac Search: interface.py
Interface to the Lean CLI for tactic validation.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional


# Path to the built veritac binary
VERITAC_BIN = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"


def call_lean(stmt_json: dict, tactics: list[dict],
              binary: Optional[Path] = None) -> dict:
    """Call the Lean CLI to apply tactics to a statement.

    Args:
        stmt_json: The LoopNest statement as a JSON dict
        tactics: List of tactic application dicts
        binary: Optional path to the veritac binary

    Returns:
        Result dict with either {"stmt": ...} or {"error": ...}
    """
    bin_path = binary or VERITAC_BIN

    if not bin_path.exists():
        raise FileNotFoundError(
            f"VeriTac binary not found at {bin_path}. Run 'lake build' first."
        )

    input_json = json.dumps({"stmt": stmt_json, "tactics": tactics})

    result = subprocess.run(
        [str(bin_path), "--json", input_json],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        return {"error": f"Process failed: {result.stderr}"}

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON output: {e}"}


def validate_schedule(stmt_json: dict, tactics: list[dict],
                      binary: Optional[Path] = None) -> tuple[bool, Optional[dict], str]:
    """Validate a tactic sequence by applying it via the Lean CLI.

    Returns:
        (success, result_stmt_json, message)
    """
    result = call_lean(stmt_json, tactics, binary)

    if "error" in result:
        return False, None, result["error"]

    return True, result.get("stmt"), f"Applied {result.get('applied', 0)} tactics"


def make_matmul_stmt(M: int, K: int, N: int) -> dict:
    """Create a naive matmul loop nest as JSON.

    Generates: for i in 0..M: for j in 0..N: for k in 0..K: C[i,j] += A[i,k] * B[k,j]
    """
    return {
        "tag": "loop", "var": "i0", "lo": {"tag": "lit", "val": 0},
        "hi": {"tag": "lit", "val": M}, "ann": "none",
        "body": {
            "tag": "loop", "var": "i1", "lo": {"tag": "lit", "val": 0},
            "hi": {"tag": "lit", "val": N}, "ann": "none",
            "body": {
                "tag": "loop", "var": "i2", "lo": {"tag": "lit", "val": 0},
                "hi": {"tag": "lit", "val": K}, "ann": "none",
                "body": {
                    "tag": "bufWrite", "buf": "C",
                    "indices": [
                        {"tag": "var", "name": "i0"},
                        {"tag": "var", "name": "i1"}
                    ],
                    "val": {
                        "tag": "add",
                        "left": {
                            "tag": "bufRead", "buf": "C",
                            "indices": [
                                {"tag": "var", "name": "i0"},
                                {"tag": "var", "name": "i1"}
                            ]
                        },
                        "right": {
                            "tag": "mul",
                            "left": {
                                "tag": "bufRead", "buf": "A",
                                "indices": [
                                    {"tag": "var", "name": "i0"},
                                    {"tag": "var", "name": "i2"}
                                ]
                            },
                            "right": {
                                "tag": "bufRead", "buf": "B",
                                "indices": [
                                    {"tag": "var", "name": "i2"},
                                    {"tag": "var", "name": "i1"}
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
