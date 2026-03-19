"""
VeriTac CodeGen: runner.py
Compile and run generated C code, compare against numpy.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .lower import parse_stmt
from .emit_c import emit_function, emit_test_harness


def compile_and_run(c_code: str, timeout: float = 30.0) -> tuple[str, float]:
    """Compile C code with clang, run it, return (stdout, elapsed_seconds)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.c")
        bin_path = os.path.join(tmpdir, "kernel")

        with open(src_path, "w") as f:
            f.write(c_code)

        # Compile
        compile_result = subprocess.run(
            ["clang", "-O2", "-fopenmp", "-o", bin_path, src_path, "-lm"],
            capture_output=True, text=True, timeout=timeout
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{compile_result.stderr}")

        # Run
        start = time.perf_counter()
        run_result = subprocess.run(
            [bin_path], capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.perf_counter() - start

        if run_result.returncode != 0:
            raise RuntimeError(f"Execution failed:\n{run_result.stderr}")

        return run_result.stdout, elapsed


def benchmark_matmul(stmt_json: dict, M: int, K: int, N: int,
                     num_runs: int = 5) -> dict:
    """Benchmark a matmul loop nest against numpy.

    Args:
        stmt_json: The LoopNest JSON from Lean
        M, K, N: Matrix dimensions
        num_runs: Number of benchmark iterations
    """
    stmt = parse_stmt(stmt_json)

    func_code = emit_function(
        "matmul", stmt,
        input_bufs=[("A", M * K), ("B", K * N)],
        output_bufs=[("C", M * N)]
    )

    # Generate init code for test harness
    init_lines = []
    init_lines.append(f"    for (int i = 0; i < {M * K}; i++) A[i] = (double)(i % 7);")
    init_lines.append(f"    for (int i = 0; i < {K * N}; i++) B[i] = (double)(i % 5);")
    init_code = "\n".join(init_lines)

    # Add timing loop
    timed_code = func_code + f"""
#include <time.h>

int main(void) {{
    double A[{M * K}];
    double B[{K * N}];
    double C[{M * N}];

    for (int i = 0; i < {M * K}; i++) A[i] = (double)(i % 7);
    for (int i = 0; i < {K * N}; i++) B[i] = (double)(i % 5);

    // Warmup
    matmul(A, B, C);

    // Benchmark
    struct timespec start, end;
    double total = 0.0;
    for (int run = 0; run < {num_runs}; run++) {{
        memset(C, 0, sizeof(C));
        clock_gettime(CLOCK_MONOTONIC, &start);
        matmul(A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total += elapsed;
    }}

    printf("avg_time_ms: %f\\n", (total / {num_runs}) * 1000.0);
    printf("C[0]=%f C[1]=%f C[2]=%f C[3]=%f\\n", C[0], C[1], C[2], C[3]);
    return 0;
}}
"""

    # Run C version
    output, _ = compile_and_run(timed_code)

    # Parse output
    c_time_ms = None
    c_values = []
    for line in output.strip().split("\n"):
        if line.startswith("avg_time_ms:"):
            c_time_ms = float(line.split(":")[1].strip())
        elif line.startswith("C["):
            parts = line.split("=")[1:]
            c_values = [float(p.split()[0]) for p in parts]

    # Numpy reference
    A_np = np.array([i % 7 for i in range(M * K)], dtype=np.float64).reshape(M, K)
    B_np = np.array([i % 5 for i in range(K * N)], dtype=np.float64).reshape(K, N)

    np_start = time.perf_counter()
    for _ in range(num_runs):
        C_np = A_np @ B_np
    np_time_ms = ((time.perf_counter() - np_start) / num_runs) * 1000.0

    # Check correctness
    C_flat = C_np.flatten()
    correct = all(
        abs(c_values[i] - C_flat[i]) < 1e-6
        for i in range(min(len(c_values), 4))
    ) if c_values else False

    return {
        "c_time_ms": c_time_ms,
        "numpy_time_ms": np_time_ms,
        "speedup": np_time_ms / c_time_ms if c_time_ms else 0,
        "correct": correct,
        "c_first_values": c_values[:4],
        "np_first_values": C_flat[:4].tolist(),
    }
