"""Test the matmul codegen pipeline end-to-end."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from CodeGen.lower import parse_stmt
from CodeGen.emit_c import emit_function


def test_naive_matmul():
    """Test that naive matmul codegen produces correct results."""
    stmt_json = {
        "tag": "loop", "var": "i0", "lo": {"tag": "lit", "val": 0},
        "hi": {"tag": "lit", "val": 4}, "ann": "none",
        "body": {
            "tag": "loop", "var": "i1", "lo": {"tag": "lit", "val": 0},
            "hi": {"tag": "lit", "val": 4}, "ann": "none",
            "body": {
                "tag": "loop", "var": "i2", "lo": {"tag": "lit", "val": 0},
                "hi": {"tag": "lit", "val": 4}, "ann": "none",
                "body": {
                    "tag": "bufWrite", "buf": "C",
                    "indices": [
                        {"tag": "var", "name": "i0"},
                        {"tag": "var", "name": "i1"}
                    ],
                    "val": {
                        "tag": "add",
                        "left": {"tag": "bufRead", "buf": "C", "indices": [
                            {"tag": "var", "name": "i0"},
                            {"tag": "var", "name": "i1"}
                        ]},
                        "right": {
                            "tag": "mul",
                            "left": {"tag": "bufRead", "buf": "A", "indices": [
                                {"tag": "var", "name": "i0"},
                                {"tag": "var", "name": "i2"}
                            ]},
                            "right": {"tag": "bufRead", "buf": "B", "indices": [
                                {"tag": "var", "name": "i2"},
                                {"tag": "var", "name": "i1"}
                            ]}
                        }
                    }
                }
            }
        }
    }

    stmt = parse_stmt(stmt_json)
    SZ = 4000000
    func_code = emit_function("matmul", stmt,
        input_bufs=[("A", SZ), ("B", SZ)],
        output_bufs=[("C", SZ)])

    test_code = func_code + r"""
int main(void) {
    double* A = calloc(4000000, sizeof(double));
    double* B = calloc(4000000, sizeof(double));
    double* C = calloc(4000000, sizeof(double));
    /* 2x2 submatrix, stride=1000 */
    A[0*1000+0]=1; A[0*1000+1]=2; A[1*1000+0]=3; A[1*1000+1]=4;
    B[0*1000+0]=5; B[0*1000+1]=6; B[1*1000+0]=7; B[1*1000+1]=8;
    matmul(A, B, C);
    /* Expected: [[19,22],[43,50]] */
    int ok = (C[0*1000+0]==19) && (C[0*1000+1]==22) &&
             (C[1*1000+0]==43) && (C[1*1000+1]==50);
    printf("C00=%g C01=%g C10=%g C11=%g ok=%d\n",
           C[0*1000+0], C[0*1000+1], C[1*1000+0], C[1*1000+1], ok);
    free(A); free(B); free(C);
    return ok ? 0 : 1;
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "matmul.c")
        binary = os.path.join(tmpdir, "matmul")
        with open(src, "w") as f:
            f.write(test_code)

        result = subprocess.run(
            ["clang", "-O2", "-o", binary, src, "-lm"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Compile failed: {result.stderr}"

        result = subprocess.run([binary], capture_output=True, text=True)
        print(result.stdout.strip())
        assert result.returncode == 0, f"Matmul test failed: {result.stdout}"
        assert "ok=1" in result.stdout


def test_tiled_matmul_via_cli():
    """Test that tiling a matmul via the CLI produces valid JSON."""
    stmt_json = {
        "tag": "loop", "var": "i0", "lo": {"tag": "lit", "val": 0},
        "hi": {"tag": "lit", "val": 256}, "ann": "none",
        "body": {
            "tag": "loop", "var": "i1", "lo": {"tag": "lit", "val": 0},
            "hi": {"tag": "lit", "val": 256}, "ann": "none",
            "body": {
                "tag": "loop", "var": "i2", "lo": {"tag": "lit", "val": 0},
                "hi": {"tag": "lit", "val": 256}, "ann": "none",
                "body": {
                    "tag": "bufWrite", "buf": "C",
                    "indices": [
                        {"tag": "var", "name": "i0"},
                        {"tag": "var", "name": "i1"}
                    ],
                    "val": {
                        "tag": "add",
                        "left": {"tag": "bufRead", "buf": "C", "indices": [
                            {"tag": "var", "name": "i0"},
                            {"tag": "var", "name": "i1"}
                        ]},
                        "right": {
                            "tag": "mul",
                            "left": {"tag": "bufRead", "buf": "A", "indices": [
                                {"tag": "var", "name": "i0"},
                                {"tag": "var", "name": "i2"}
                            ]},
                            "right": {"tag": "bufRead", "buf": "B", "indices": [
                                {"tag": "var", "name": "i2"},
                                {"tag": "var", "name": "i1"}
                            ]}
                        }
                    }
                }
            }
        }
    }

    tactics = [
        {"kind": "tile", "vars": ["i0"], "int_params": [32]},
        {"kind": "tile", "vars": ["i1"], "int_params": [32]},
        {"kind": "parallel", "vars": ["i0_outer"]},
    ]

    input_json = json.dumps({"stmt": stmt_json, "tactics": tactics})
    bin_path = Path(__file__).parent.parent / ".lake" / "build" / "bin" / "veritac"

    result = subprocess.run(
        [str(bin_path), "--json", input_json],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    output = json.loads(result.stdout.strip())
    assert "error" not in output, f"CLI error: {output.get('error')}"
    assert output["applied"] == 3
    assert output["stmt"]["ann"] == "parallel"
    print(f"Applied {output['applied']} tactics, outer loop is parallel")


if __name__ == "__main__":
    test_naive_matmul()
    print("PASS: naive matmul")
    test_tiled_matmul_via_cli()
    print("PASS: tiled matmul via CLI")
    print("All tests passed!")
