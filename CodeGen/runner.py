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


_OMP_BASE: tuple[str, list[str]] | None = None  # cached: (cc, OpenMP flags+sysroot)


def _find_omp_base() -> tuple[str, list[str]] | None:
    """Return (compiler, OpenMP compile flags) usable on this machine.

    On Apple silicon the Homebrew gcc needs `-isysroot` to link against the
    SDK (plain clang ships without OpenMP), so we probe by compiling *and
    linking* a trivial program. Returns None when no OpenMP compiler exists.
    """
    import shutil
    sysroot = ""
    probe_src = "/tmp/veritac_omp_probe.c"
    with open(probe_src, "w") as f:
        f.write("int main(void){return 0;}\n")
    if shutil.which("xcrun"):
        try:
            sysroot = subprocess.run(
                ["xcrun", "--show-sdk-path"], capture_output=True, text=True
            ).stdout.strip()
        except Exception:
            sysroot = ""

    candidates = []
    if os.environ.get("VERITAC_CC"):
        candidates.append(os.environ["VERITAC_CC"])
    candidates += ["gcc-15", "gcc-14", "gcc-13", "gcc-12", "gcc", "clang"]

    for cc in candidates:
        flags = ["-fopenmp", "-O2"]
        if sysroot:
            # Homebrew gcc on Darwin ships a stale `include-fixed` dir before
            # the SDK; pushing the SDK's usr/include ahead fixes <stdio.h>.
            flags += ["-isysroot", sysroot, "-I", f"{sysroot}/usr/include"]
        try:
            probe = subprocess.run(
                [cc] + flags + ["-o", "/tmp/omp_probe_bin", probe_src],
                capture_output=True, text=True, timeout=20,
            )
            if probe.returncode == 0:
                return cc, flags
        except FileNotFoundError:
            continue
    return None


def compile_and_run(c_code: str, timeout: float = 30.0) -> tuple[str, float]:
    """Compile and run C code, return (stdout, elapsed_seconds).

    Uses an OpenMP-capable compiler if one is available (so `parallel` /
    `vectorize` pragmas take effect); otherwise falls back to a plain compile
    where the pragmas are inert no-ops and the code still runs correctly
    (just without parallelism). The emitted pragmas are only generated when
    `writes_local_to` deems the loop safe (see emit_c.py), so both paths
    preserve semantics.
    """
    global _OMP_BASE
    if _OMP_BASE is None:
        _OMP_BASE = _find_omp_base()

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.c")
        bin_path = os.path.join(tmpdir, "kernel")

        with open(src_path, "w") as f:
            f.write(c_code)

        if _OMP_BASE is not None:
            cc, flags = _OMP_BASE
            cmd = [cc] + flags + ["-o", bin_path, src_path, "-lm"]
        else:
            cc = os.environ.get("VERITAC_CC", "clang")
            cmd = [cc, "-O2", "-o", bin_path, src_path, "-lm"]

        compile_result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{compile_result.stderr}")

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

    # The code generator uses a fixed 1000-stride flat layout (matching Lean's
    # flatIndex), so buffers must be sized for that layout: a d-dimensional
    # buffer needs 1000^d * logical_size_head cells. Using M*K etc. was a bug
    # that made C[i0*1000+i1] overflow (SIGBUS).
    stride = 1000
    def flat_sz(dims: tuple[int, ...]) -> int:
        sz = 1
        for d in dims:
            sz *= d
        return sz * (stride ** (len(dims) - 1))

    # Map from flat C position back to logical (i,j) for numpy comparison.
    A_sz = flat_sz((M, K))
    B_sz = flat_sz((K, N))
    C_sz = flat_sz((M, N))

    func_code = emit_function(
        "matmul", stmt,
        input_bufs=[("A", A_sz), ("B", B_sz)],
        output_bufs=[("C", C_sz)]
    )

    # C uses A[i*1000 + k], B[k*1000 + j], C[i*1000 + j] (single-dim layout).
    # Fill the *used* cells: A at i*1000+k, B at k*1000+j, C zeroed.
    init_lines = []
    init_lines.append("    memset(A, 0, sizeof(A));")
    init_lines.append("    memset(B, 0, sizeof(B));")
    init_lines.append("    memset(C, 0, sizeof(C));")
    init_lines.append(f"    for (int i = 0; i < {M}; i++) for (int k = 0; k < {K}; k++) A[i * 1000 + k] = (double)((i * {K} + k) % 7);")
    init_lines.append(f"    for (int k = 0; k < {K}; k++) for (int j = 0; j < {N}; j++) B[k * 1000 + j] = (double)((k * {N} + j) % 5);")
    init_code = "\n".join(init_lines)

    # Add timing loop
    timed_code = func_code + f"""
#include <time.h>
#include <stdlib.h>

int main(void) {{
    /* The 1000-stride flat layout makes buffers tens of MB; keep them off the
       (small) default thread stack. */
    double *A = (double*)malloc(sizeof(double) * {A_sz});
    double *B = (double*)malloc(sizeof(double) * {B_sz});
    double *C = (double*)malloc(sizeof(double) * {C_sz});
    if (!A || !B || !C) {{ perror("malloc"); return 2; }}

    {init_code}

    // Warmup
    matmul(A, B, C);

    // Benchmark
    struct timespec start, end;
    double total = 0.0;
    for (int run = 0; run < {num_runs}; run++) {{
        memset(C, 0, {C_sz} * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &start);
        matmul(A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total += elapsed;
    }}

    printf("avg_time_ms: %f\\n", (total / {num_runs}) * 1000.0);
    printf("C[0]=%f C[1000]=%f C[{1000 * (M - 1) + 0}]=%f C[{1000 * (M - 1) + (N - 1)}]=%f\\n",
           C[0], C[1000], C[{1000 * (M - 1)}], C[{1000 * (M - 1) + (N - 1)}]);
    free(A); free(B); free(C);
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

    # Numpy reference, built from the same logical (i%7, k%5) values.
    A_np = np.array([((i * K + k) % 7) for i in range(M) for k in range(K)],
                    dtype=np.float64).reshape(M, K)
    B_np = np.array([((k * N + j) % 5) for k in range(K) for j in range(N)],
                    dtype=np.float64).reshape(K, N)

    np_start = time.perf_counter()
    for _ in range(num_runs):
        C_np = A_np @ B_np
    np_time_ms = ((time.perf_counter() - np_start) / num_runs) * 1000.0

    # The C program prints C at flat positions 0, 1000, 1000*(M-1), 1000*(M-1)+(N-1)
    flat_positions = [0, 1000, 1000 * (M - 1), 1000 * (M - 1) + (N - 1)]
    expected = [C_np[p // 1000][p % 1000] if p % 1000 < N else float("nan")
                for p in flat_positions]
    correct = all(
        abs(c_values[i] - expected[i]) < 1e-6
        for i in range(min(len(c_values), len(expected)))
    ) if c_values else False

    return {
        "c_time_ms": c_time_ms,
        "numpy_time_ms": np_time_ms,
        "speedup": np_time_ms / c_time_ms if c_time_ms else 0,
        "correct": correct,
        "c_first_values": c_values[:4],
        "np_first_values": expected,
    }
