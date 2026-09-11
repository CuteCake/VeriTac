# Metal FP32 causal-prefill attention candidate (flash-style)

A standalone, correctness-first Metal candidate kernel and benchmark harness for
causal prefill attention. It is **not** a claimed vendor speedup: it is a
reviewable starting point that implements the exact flash-style streaming
online-softmax schedule frozen for this benchmark, and it is validated against
an independent numpy float64 stable-causal reference.

Scope is intentionally narrow and frozen:

| Property | Frozen value |
|---|---|
| dtype | `float32` only |
| Shape | `[B, H, N, D]`, `B=1`, `H=8`, `N` variable, `D` only `192` or `256` |
| Mapping | `scalar` or `simdgroup` |
| Query tile | scalar: `8` or `16`; simdgroup: `4` or `8` |
| Key tile | `8` or `16` |
| Mask | causal (each query row attends to keys `j <= i`) |
| Scale | `D**-0.5` |
| Tolerances | `atol=1e-4`, `rtol=1e-3` (frozen FP32) |

## Layout

```
benchmarks/attention/metal_candidate/
    metal_candidate.swift   # Metal kernel (embedded MSL) + host runner
    metal_candidate.py      # Python controller / benchmark harness
benchmarks/attention/METAL_CANDIDATE.md
```

The controller talks to the Swift runner through binary files exactly like the
survey helpers: it writes Q/K/V bins and a request JSON, the runner computes the
output, and both sides exchange a response JSON.

Two mappings are implemented in `metal_candidate.swift`, sharing the identical
streaming online-softmax schedule and O(ND) storage (K/V staged in one dynamic
threadgroup allocation of `2*key_tile*D*4` bytes, no N-by-N buffer):

- `scalar`: one thread per query row (`threads_per_threadgroup == query_tile`,
  query tile 8 or 16); each thread owns the full D output accumulator.
- `simdgroup`: one 32-lane Metal SIMD-group per query row
  (`threads_per_threadgroup == query_tile*32`, query tile 4 or 8); each lane
  computes a strided D subset and the dot product is finished with a `simd_sum`
  reduction, with each lane owning `D/32` output accumulators.

## Exact compile & run commands

Compile the Swift runner (once; outputs into a writable repo-local build dir
outside tracked files):

```bash
# from the project root
mkdir -p .lake/metal_candidate_worker
xcrun swiftc -O -framework Metal -framework MetalPerformanceShaders \
    benchmarks/attention/metal_candidate/metal_candidate.swift \
    -o .lake/metal_candidate_worker/metal_candidate
```

The Python controller uses `.lake/metal_candidate_worker` as the default
`--build-dir` and recompiles the runner only when the Swift source is newer than
the binary.

Run a correctness smoke test (SIMD-group mapping, query tile 4/8, key tile
8/16, both dims, plus a non-divisible N):

```bash
PYTHONPATH=. python3 benchmarks/attention/metal_candidate/metal_candidate.py \
    --output .lake/metal_candidate_worker/smoke.json \
    --seqs 128,100 --dims 192,256 \
    --mappings simdgroup --query-tiles 4,8 --key-tiles 8,16 \
    --verify-only
```

Scalar smoke (query tile 8/16):

```bash
PYTHONPATH=. python3 benchmarks/attention/metal_candidate/metal_candidate.py \
    --output .lake/metal_candidate_worker/scalar-smoke.json \
    --seqs 128 --dims 192 \
    --mappings scalar --query-tiles 8,16 --key-tiles 8,16 \
    --verify-only
```

Run a timing sweep:

```bash
PYTHONPATH=. python3 benchmarks/attention/metal_candidate/metal_candidate.py \
    --output .lake/metal_candidate_worker/sweep.json \
    --seqs 128,512,1536 --dims 192,256 \
    --mappings simdgroup --query-tiles 4,8 --key-tiles 8,16 \
    --warmup 10 --samples 30 --inner 5
```

The controller recompiles the runner only when the Swift source is newer than
the binary. It needs only `numpy` (Python) and the Apple system `Metal`
framework (Swift); no MLX, no MPSGraph, no CUDA.

### Controller options

- `--verify-only`: minimal samples (forces `samples=1` unless already smaller),
  and exits non-zero if **any** configuration fails correctness.
- `--seqs`, `--dims`, `--mappings`, `--query-tiles`, `--key-tiles`:
  comma-separated grids. `--mappings` selects `scalar` and/or `simdgroup`.
  `--dims` is restricted to 192/256; `--key-tiles` to 8/16; `--query-tiles` to
  4/8/16. Invalid `(mapping, query_tile)` pairs are **skipped deterministically**
  (recorded per case under `skipped_configs`): scalar requires query tile 8 or
  16, simdgroup requires query tile 4 or 8.
- `--warmup`, `--samples`, `--inner`: timing parameters (`inner` invocations
  per timed sample, `samples` samples; raw per-invocation timings are kept).
- `--output`: output JSON path (written incrementally after every case).
- `--build-dir`: where the compiled Swift binary lives; default is
  `<repo>/.lake/metal_candidate_worker` (writable repo path, not `/tmp`).
- `--mem-cap-bytes`: refuse cases whose float64 reference peak exceeds this.

## Kernel contract (frozen)

- Launch geometry is a 2-D grid: `grid.x = ceil(N/query_tile)` tiles per head,
  `grid.y = B*H = 8` (one plane per head). `threads_per_threadgroup =
  query_tile`, and **one thread owns exactly one query row**. The thread index
  is read directly from the `thread_position_in_threadgroup` attribute (a
  `uint2`, keeping a consistent vector width for all MSL position attributes),
  and the query-row tile is indexed by `threadgroup_position_in_grid`.
  Because every threadgroup maps to a single head plane, the K/V staging is
  uniform across the group even though each thread owns a distinct row.
- Every threadgroup **uniformly loops over every K/V tile** and reaches both
  barriers on every iteration, even when its query row is out of range (the
  trailing partial query tile, where `row_start + tid >= N`). Out-of-range
  threads still participate in the cooperative load and the barriers; their row
  index is **clamped to a safe in-bounds row before any Q device pointer is
  formed**, so no out-of-range device pointer is ever produced (even though it
  is never dereferenced).
- K and V are **cooperatively staged in a single dynamic threadgroup allocation
  of exactly `2*key_tile*D*4` bytes** (K tile in `[0, key_tile*D)`, V tile in
  `[key_tile*D, 2*key_tile*D)`). There is **no N-by-N score/probability buffer**
  anywhere; scores are produced per key tile and consumed immediately by the
  streaming softmax. `2*key_tile*D` is guaranteed divisible by `query_tile` for
  every allowed `(query_tile, key_tile, D)` combination, so the cooperative load
  divides evenly.
- Each valid thread computes causal scaled dot products (`scale * q·k`,
  accumulated in float32) and maintains **streaming online-softmax**
  state `m` (running max), `l` (running sum of `exp`), and an output
  accumulator `acc`, updating them per score and finally writing `acc / l`.
  State is explicitly initialized (`m = -inf`, `l = 0`, `acc = 0`). FP32
  accuracy is **established by the numerical validation** against the float64
  reference below, not claimed a priori.
- **Partial key tiles** (trailing `kt + key_tile > N`) are handled: the load
  guards `c < N` (writing 0 out-of-range) and the compute loop bounds `hi =
  min(kt + key_tile, i + 1)`, so masked keys are never read or counted.
  **Partial query tiles** are handled by the `valid = (i < N)` guard.
- `N = 1536` is a first-class supported case.

### Proof boundary (what this harness demonstrates)

This harness demonstrates **numeric correctness** of the candidate kernel
against an independent reference, and **records honest launch geometry and
timing metadata**. It does **not** prove:

- that the kernel is faster than MLX / MPSGraph / CUDA baselines (no speedup is
  claimed here);
- that a particular tile size is optimal for any device;
- that any specific fused kernel was selected by another backend.

A configuration that computes but fails the numeric contract is reported
`numeric_failed`, never silently replaced by a baseline or another tile. A
runtime/GPU error fails the case with its error.

## What is measured (all units: milliseconds)

- **Command-buffer GPU duration:** per invocation, `MTLCommandBuffer.gpuStartTime`
  / `gpuEndTime` deltas read after `waitUntilCompleted`, excluding input/output
  CPU copies (preallocated shared `MTLBuffer`s reused) and compile/warmup. If
  the timestamps are invalid/incomplete, `gpu_time_available` is `false` and
  `gpu_timing` is `null` (never a fabricated summary).
- **Synchronized API latency:** wall time from before command-buffer allocation
  through `waitUntilCompleted`, per invocation. `samples * inner` raw per
  invocation timings are kept in `wall_timing.samples_ms` (and
  `gpu_timing.samples_ms`).
- The Metal shader is **compiled once** per process via
  `MTLLibrary.makeLibrary(source:)` and reused for warmup + all timed samples.
- A command buffer ending in a Metal error fails the case rather than reporting
  a bogus timing.

## Validation

Every output is checked for full-output finiteness, shape equality, and
`abs(error) <= atol + rtol*abs(ref)` element-wise against the float64 reference
(the reference is computed per head so its peak stays O(N^2)). A configuration
that computes but fails the numeric contract is `numeric_failed`, not `ok`.

### Input seeding contract

Per-case input seed is derived exactly like the vendor survey
(`metal_survey.py`/`cuda_survey.py`) for `float32`:

```
case_seed = seed + (seq*1000003 + dim*104729 + len("float32")*131071) % 2**31
```

Inputs are independent `numpy.default_rng(case_seed)` normal draws (std 0.5)
rounded directly to float32, distinct Q/K/V. Because the candidate and the
survey share this derivation, an identical `--seed` produces byte-identical
q/k/v inputs; the demo (`examples/llm_attention_demo.py`) passes the same seed
to both and cross-checks the recorded `inputs_sha256` hashes. `case_seed` and
the Q/K/V SHA-256 hashes are recorded per case for cross-machine confirmation.

## Measured test evidence

The following was measured on this machine (`Apple M3 Ultra`, Metal 4) with the
compiled runner in `<repo>/.lake/metal_candidate_worker/metal_candidate` and the
frozen FP32 contract (atol=1e-4, rtol=1e-3). Numbers are median GPU durations
(`MTLCommandBuffer.gpuStartTime`/`gpuEndTime` deltas, ms) from
`--verify-only` runs (samples=1); GPU timestamps were reported available for
every case. This is **numeric-correctness and honest-timing evidence only**; no
vendor speedup is claimed.

SIMD-group mapping (`--mappings simdgroup --query-tiles 4,8 --key-tiles 8,16`):

| seq | dim | mapping | qt | kt | status | gpu_med_ms | max_abs_err | norm_max |
|----:|----:|---------|:--:|:--:|:------:|-----------:|------------:|---------:|
| 128 | 192 | simdgroup | 8 | 8 | ok | 0.352 | 1.29e-07 | 5.12e-04 |
| 128 | 192 | simdgroup | 8 | 16 | ok | 0.348 | 1.29e-07 | 5.12e-04 |
| 128 | 192 | simdgroup | 4 | 8 | ok | 0.486 | 1.29e-07 | 5.12e-04 |
| 128 | 192 | simdgroup | 4 | 16 | ok | 0.491 | 1.29e-07 | 5.12e-04 |
| 128 | 256 | simdgroup | 8 | 8 | ok | 0.460 | 1.81e-07 | 4.74e-04 |
| 128 | 256 | simdgroup | 8 | 16 | ok | 0.274 | 1.81e-07 | 4.74e-04 |
| 128 | 256 | simdgroup | 4 | 8 | ok | 0.402 | 1.81e-07 | 4.74e-04 |
| 128 | 256 | simdgroup | 4 | 16 | ok | 0.489 | 1.81e-07 | 4.74e-04 |
| 100 | 192 | simdgroup | 8 | 8 | ok | 0.181 | 1.26e-07 | 5.29e-04 |
| 100 | 192 | simdgroup | 8 | 16 | ok | 0.197 | 1.26e-07 | 5.29e-04 |
| 100 | 192 | simdgroup | 4 | 8 | ok | 0.229 | 1.26e-07 | 5.29e-04 |
| 100 | 192 | simdgroup | 4 | 16 | ok | 0.325 | 1.26e-07 | 5.29e-04 |
| 100 | 256 | simdgroup | 8 | 8 | ok | 0.208 | 1.48e-07 | 5.20e-04 |
| 100 | 256 | simdgroup | 8 | 16 | ok | 0.225 | 1.48e-07 | 5.20e-04 |
| 100 | 256 | simdgroup | 4 | 8 | ok | 0.289 | 1.48e-07 | 5.20e-04 |
| 100 | 256 | simdgroup | 4 | 16 | ok | 0.375 | 1.48e-07 | 5.20e-04 |

Scalar mapping smoke (`--mappings scalar --query-tiles 8,16 --key-tiles 8,16`):

| seq | dim | mapping | qt | kt | status | gpu_med_ms | max_abs_err | norm_max |
|----:|----:|---------|:--:|:--:|:------:|-----------:|------------:|---------:|
| 128 | 192 | scalar | 8 | 8 | ok | 4.786 | 2.27e-07 | 1.26e-03 |
| 128 | 192 | scalar | 8 | 16 | ok | 4.783 | 2.27e-07 | 1.26e-03 |
| 128 | 192 | scalar | 16 | 8 | ok | 4.380 | 2.27e-07 | 1.26e-03 |
| 128 | 192 | scalar | 16 | 16 | ok | 4.386 | 2.27e-07 | 1.26e-03 |

Notes:
- `seq=100` is a **non-divisible N** (not divisible by query tile 8 or key tile
  16); partial query/key tiles validate correctly.
- Deterministic skipping of invalid `(mapping, query_tile)` pairs was verified:
  with `--mappings scalar,simdgroup --query-tiles 4,8,16`, the harness skipped
  exactly `[('scalar', 4), ('simdgroup', 16)]` (recorded under
  `skipped_configs`).
- Every case passed the frozen FP32 numeric contract; no `numeric_failed` or
  runtime failure was observed. Full raw JSON is in
  `metal_candidate/test_simd.json` and `metal_candidate/test_scalar_smoke.json`.

## Device / environment

The runner reports `mtl_device_name`, `registryID`, low-power and unified-memory
flags, the device `maxThreadgroupMemoryLength`, and the launch geometry. After
pipeline creation it also validates and records the compiled-pipeline limits:
`query_tile <= pipeline.maxTotalThreadsPerThreadgroup` and
`dynamic_threadgroup_memory_length + pipeline.staticThreadgroupMemoryLength <=
device.maxThreadgroupMemoryLength`; the compiled values and their total are
reported under the response `pipeline` object. For the allowed
`(key_tile=16, D=256)` case the dynamic staging allocation is
`2*16*256*4 = 32768` bytes, which must be `<= device.maxThreadgroupMemoryLength`
(32768 on common Apple GPUs); the runner validates this before dispatch and
fails explicitly if a device does not allow it.

## Error reporting

Once the `response_path` is known, any pre-dispatch failure (invalid request,
input read/size errors, device or pipeline limit violations, MSL/pipeline
compilation, GPU command-buffer errors) is reported as a structured `status:
"error"` response JSON written to `response_path`, rather than exiting the
process. Only errors occurring before `response_path` can be known (e.g. the
request file itself cannot be read or decoded) fall back to stderr + non-zero
exit.
