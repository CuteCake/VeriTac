# Attention baseline survey and implementation decision

Measured September 9, 2026 (Pacific time; raw timestamps are September 10 UTC).
This is a baseline survey, not a claim that VeriTac already generates an
optimized attention kernel.

## Decision

Build the first hardware-aware attention demonstration on **Metal / M3 Ultra**,
using **FP32 causal prefill with head dimensions 192 and 256**. Retain the
original **CUDA / GB10** demo as the second backend. Share attention semantics,
layout proofs, and structural scheduling concepts between them.

The reason is concrete: the installed MLX forced-fused FP32 kernels for these
dimensions exceed this Mac's threadgroup-memory limit. Generating a smaller,
legal tile is a direct application of a hardware-aware tactic system. It is
also an opportunity to improve attention execution, but a speedup over the
fastest Apple implementation remains a goal to test, not an established result.

Keep a matched FP16 performance track. MLX already has a useful fused FP16 path;
merely selecting that existing path is a dispatch improvement, not a new
VeriTac kernel achievement.

## Machines and method

| | CUDA worker | Local Metal worker |
|---|---|---|
| Hardware | NVIDIA GB10, `sm_121`, 48 SMs | Apple M3 Ultra, 80 GPU cores, 256 GB |
| Runtime | PyTorch 2.14.0+cu130, cuDNN 9.24 | MLX 0.32.2; macOS 26.6.2 MPSGraph |
| Shared/threadgroup memory | 48 KiB default, 99 KiB opt-in per block | 32 KiB per threadgroup |
| Primary comparison | Explicit PyTorch SDPA backend selection | MLX default, MLX forced fused, MPSGraph SDPA |

The initial matrix was `B=1,H=8,N={256,1024},D={64,128,192,256}`, with both
FP32 and FP16: 16 cases on each machine. The confirmation matrix used
`N={1024,2048},D={128,192,256}` in FP16. A further FP32 Metal pass examined
`N={1024,2048},D={192,256}` and included a simple expression baseline.

All cases compute square causal prefill, with contiguous BHND inputs,
`scale=1/sqrt(D)`, and no dropout. Inputs are seeded, distinct Q/K/V normal
draws with standard deviation 0.5, rounded to the tested dtype. The same-case
input hashes match across machines. Each output is compared in full against
an independent stable NumPy FP64 calculation from the rounded inputs.

The frozen numerical thresholds are FP32 `atol=1e-4,rtol=1e-3` and FP16
`atol=2e-3,rtol=2e-2`. These are empirical survey criteria, not formal error
bounds. This survey uses one deterministic random input set per case; broader
seeds, zeros, extreme finite scores and mask-edge tests are required for a
generated kernel.

Initial runs used 10 warmups and 30 samples; confirmation runs used 20 warmups
and 50 samples. CUDA retains eager CUDA-event timing, synchronized call latency,
and graph replay timing (five attention invocations per graph). Metal retains
synchronized call latency, with command-buffer GPU timing additionally available
for MPSGraph. Metal has 150 timed calls per initial case/backend and 100 per
confirmation case/backend. Preparation, reference evaluation, compilation and
warmup are outside timing. Backend order is shuffled deterministically per case.
Runs on the same GPU are serial.

**CUDA graph time and Metal API latency are different metrics.** They must not
be divided to claim one GPU or backend is faster than the other. The Metal
numbers include Python/graph submission, allocation and synchronization costs;
they describe the tested calling paths, not a matched native-kernel comparison.
MPSGraph and MLX also use different output-allocation strategies. GPU profiling
and an equivalent invocation path are required before claiming a kernel-level
win. Inputs are reused, so these are warm-buffer measurements.

## What the measurements show

### Metal FP32: a real hardware-fit problem

Forced MLX fused attention fails for both sequence lengths in the initial
matrix, with these runtime diagnostics:

| Head dimension | Requested threadgroup memory | Available |
|---:|---:|---:|
| 192 | 40,448 bytes | 32,768 bytes |
| 256 | 53,760 bytes | 32,768 bytes |

The failure is recorded; the harness does not silently substitute another
implementation. MLX default and MPSGraph still execute and pass numerical checks.

The dedicated FP32 target pass measured these median **synchronized API
latencies**, in milliseconds:

| N | D | MLX default | MPSGraph | MPSGraph GPU interval |
|---:|---:|---:|---:|---:|
| 1024 | 192 | 0.805 | 0.873 | 0.573 |
| 1024 | 256 | 0.908 | 0.987 | 0.679 |
| 2048 | 192 | 2.247 | 2.422 | 2.123 |
| 2048 | 256 | 2.648 | 2.354 | 2.040 |

MPSGraph won the 1024×256 case in the earlier pass (0.832 ms versus MLX's
0.906 ms), so that close ranking is not stable. Preserve both comparators and
remeasure them beside each candidate. The gap between GPU and API time also
makes short cases poor evidence of kernel-level improvement.

### Metal FP16: default dispatch leaves some performance available

Confirmation medians, again **synchronized API latency in milliseconds**:

| N | D | MLX default | MLX forced fused | MPSGraph |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.418 | 0.422 | 0.440 |
| 1024 | 192 | 0.774 | 0.535 | 0.861 |
| 1024 | 256 | 0.863 | 0.731 | 0.837 |
| 2048 | 128 | 0.888 | 0.857 | 1.084 |
| 2048 | 192 | 1.885 | 1.269 | 2.219 |
| 2048 | 256 | 2.281 | 1.873 | 2.274 |

Forced fusion improves the D=192 cases by approximately 1.45–1.49× over MLX
default; D=256 improves by about 1.18–1.22×. The initial 1024-token pass showed
the same direction. This is existing library functionality and belongs in the
baseline we must beat.

The pinned [MLX 0.32.2 dispatcher](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp)
contains fused implementations for these dimensions but normally routes these
full-attention cases to an unfused path on this target. `force_fused` overrides
that choice; it does not guarantee that the resulting pipeline fits hardware.
This corrects the earlier design's tentative assumption that the fused
implementations might simply be absent. Successful kernel identities were not
captured here; dispatch interpretation is based on the pinned source/API.

### CUDA: strong existing low-precision baselines

The initial matrix produced 60 numerically passing backend results and 20
explicitly unsupported combinations. FP32 cuDNN/Flash were unavailable in the
tested route. For FP16, the forced PyTorch/cuDNN route rejects D=192/256 with
`head_dim should be no more than 128`; Flash handles both. This is a finding
about the tested PyTorch route, not an exhaustive limitation claim about direct
cuDNN Frontend execution plans.

At N=1024, the initial FP16 CUDA graph medians were roughly 39 µs (D=64),
55 µs (D=128), 79 µs (D=192), and 87 µs (D=256) for the fastest supported
paths. The larger confirmation cases and every backend's raw samples are
available in the linked results. CPU profiler operator names confirm that forced
cuDNN, Flash, and efficient requests use their respective attention operations;
auto uses Flash in the measured FP16 cases and efficient attention in FP32.

## Goals and methods

### Goal 1: verified resource feasibility and attention semantics

Deliver a replayable, tactic-generated FP32 Metal attention kernel for
`B=1,H=8,N={1024,2048},D={192,256}`, with:

* No global N×N score/probability allocation.
* Checked threadgroup-memory use, including padding and all simultaneously live
  scratch allocations, within the measured 32 KiB limit.
* Explicit coverage, buffer ownership, synchronization and bounds checks.
* A proved real-arithmetic online-softmax transformation; empirical validation
  for its floating-point realization and the emitted Metal program.

Keep scalar values separate from integer indices. Introduce a checked tactic
subset whose executable acceptance is connected to its Lean soundness theorem.
Do not inherit the existing annotation-only parallel guarantee as a GPU proof.

Start with small query/key tiles, for example query `{8,16}` and key `{8,16}`,
then grow only when the full resource calculation allows it. For D=256 FP32,
staging K and V at key tile 16 already consumes the entire 32 KiB, leaving no
space for other shared scratch. Key tile 8 is therefore a sensible first
candidate if Q or scores also live in threadgroup memory. This is an initial
search hypothesis, not a claim of optimality.

Tactics should control tiling, memory placement, workgroup/subgroup binding,
private accumulator ownership and unrolling. Introduce SIMD-group matrix or
reduction primitives with their own semantics and validation boundary when
scalar execution is insufficient for competitive performance. Barrier placement
is a consequence of staging/lifetime proofs, not a free LLM guess.

### Goal 2: a measured competitive result

Target **at least 1.2× geometric-mean speedup** over the fastest valid Apple
library path on the four FP32 cases, with no repeatable regression greater than
5% on an individual target case. Compare in the same run, include MLX forced
fusion whenever it becomes legal, and retain both API and GPU timing evidence.
These thresholds are goals, not a prediction that the implementation will meet
them. A fused-but-slower result satisfies only the feasibility milestone.

Before claiming success, interleave baseline/candidate runs across at least
three rounds, inspect distributions, test multiple input seeds, and validate
held-out `N=1536` plus irregular tails after guarded tiling is supported. Small
timing differences inside the observed variation do not count as wins. Keep
the complete case matrix and report losses alongside gains.

The FP16 stretch goal is to beat the **forced-fused** comparator, especially
D=192. A switch from default MLX to its existing faster option is recorded as
a dispatch tactic result, separately from generated-kernel speedup.

### Goal 3: the original CUDA demo on the same foundations

Port the accepted kernel IR and applicable tactics to CUDA on GB10. Use FP32
first to validate lowering, then FP16 and supported matrix operations for
competitive comparisons. Reuse the measured Flash/cuDNN baselines and verify
the actual backend per run. CUDA's larger shared-memory budget must come from
its hardware profile rather than a hard-coded Metal schedule.

### Worker/controller workflow

OpenCode writes source code; the controller owns scope, interface contracts,
review, tests, device execution and acceptance. Maximum concurrency is **two
OpenCode requests**. After freezing a small IR/JSON interface, independent Lean
and backend work can occupy the two slots; dependent fixes are sequential.
Workers get bounded file ownership and may not recursively delegate. Compiler
success, worker assurances, or a `verified` label do not replace controller
verification.

This survey used that workflow. Controller review caught and sent back timing
unit errors, non-reproducible seeds, an unsupported PyTorch metadata access,
extra work in a timed baseline, and an MLX dtype mismatch. Runtime smoke checks
then exercised both precisions and the actual vendor backends. No verified
kernel implementation is included in this survey change.

## Artifacts and reproduction

Harness usage: [CUDA](../benchmarks/attention/CUDA.md),
[Metal](../benchmarks/attention/METAL.md). Exact installed packages:
[CUDA environment](../benchmarks/attention/results/cuda-environment.txt),
[Metal environment](../benchmarks/attention/results/metal-environment.txt).

Raw results contain full comparisons, failures, input hashes, versions, and
timing samples:

* [CUDA initial matrix](../benchmarks/attention/results/cuda-survey.json)
* [CUDA confirmation](../benchmarks/attention/results/cuda-confirmation.json)
* [Metal initial matrix](../benchmarks/attention/results/metal-survey.json)
* [Metal FP16 confirmation](../benchmarks/attention/results/metal-confirmation.json)
* [Metal FP32 targets](../benchmarks/attention/results/metal-fp32-targets.json)

Use isolated environments. CUDA was installed from the official PyTorch
`cu130` wheel index on Spark3. Metal used MLX 0.32.2 locally and compiled the
Swift helper against the installed macOS SDK. Cap BLAS threads at four for
reference calculations; do not run two benchmarks concurrently on one GPU.
The desktop was not placed in an exclusive benchmarking mode and no clocks
were locked; Metal tail latency varies appreciably. There is no measured
VeriTac-versus-vendor speedup yet.
