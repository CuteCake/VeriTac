# Attention optimization controller log

## Objective and acceptance gate

Beat the fastest successful vendor implementation on specific, disclosed FP32
causal-prefill shapes on M3 Ultra, then apply the same verified transformations
on GB10 CUDA. Keep B=1, H=8, D=192/256 and the existing numerical contract
(atol=1e-4, rtol=1e-3). No precision reduction or input-dependent shortcuts.

A candidate must pass full-output comparison with the independent float64
reference. Candidate/vendor Q/K/V hashes must match. Repeat potential wins in
at least three rounds, include both backend losses and wins, test multiple
seeds and irregular tails, and retain raw timing samples. Full-pipeline GPU
time includes all passes, merge, and required intermediate operations. Wall
latency is reported separately; different submission overheads are not proof
of a faster GPU kernel. Unsupported vendor backends are not counted as wins.

## Diagnosis of the first SIMD kernel

The original SIMD mapping owns one query row per SIMD group. Each key requires
a dot-product reduction, exponentiation, and a dependent update/rescaling of
all output accumulators. Two threadgroup barriers surround each K/V tile of
8 or 16 keys. Q is referenced from device memory inside the inner loop. The
critical path scales with the causal prefix length, and few query rows share
each staged tile. These observations motivate experiments, not a claimed
hardware-counter profile.

Controller measurements before the new experiments (B1 H8 N1024):

| D | Candidate wall ms | MLX wall ms | MPSGraph wall ms |
|---|---:|---:|---:|
| 192 | 4.8244 | 0.8917 | 0.9713 |
| 256 | 6.4665 | 0.9284 | 0.8339 |

## Transform under investigation

Split each causal key prefix into disjoint ranges. Each range computes a
summary (reference maximum, exponential denominator, weighted numerator).
Rescale summaries to a common maximum, sum their denominators/numerators,
then normalize once. Empty ranges contribute zero and must never evaluate
an undefined infinity subtraction in the implementation.

Lean should prove the finite-sum partition and rescaling identities over
real numbers, as well as structural coverage where provided. These theorems
do not by themselves establish floating-point error, correct lowering, or
race freedom of a particular shader. The controller must review that link
and validate the executable separately.

Secondary experiments: specialize head dimensions, retain Q in registers,
remove unnecessary staging, batch softmax updates, and tile multiple queries
and keys using matrix operations if redundant memory traffic limits splitting.

OpenCode writes implementation code with at most two concurrent requests.
The controller reviews changes, owns acceptance and final benchmark results,
and keeps GPU experiments on each device serial.

## Partition experiment

The first partition implementation uses one threadgroup per query row,
one SIMD-group per partition, two scans of its keys (max, then weighted sum),
and an in-threadgroup merge. Controller review fixed concurrent output writes
by assigning final merge/output ownership to SIMD-group zero, after all groups
reach the shared-memory barrier.

Full-output checks passed for N=64/100/128 and both dimensions, including empty
partitions and irregular tails. At N1024, D192 split4 achieved 1.9537 ms GPU;
D256 split16 achieved 2.6869 ms GPU. Maximum absolute errors were below 1.7e-7.
The weak response to split count suggests throughput/redundant memory traffic
limits this version; this is an inference from timing, not hardware profiling.
Neither dimension beats the vendor. Raw results: `benchmarks/attention/results/metal-partition-*.json`.

Next experiment: FP32 SIMD matrix tiling to reuse K/V across query rows,
compute score blocks, and update softmax summaries once per key tile.

## Original matrix experiment and semantic proof

The first original matrix shader passes numerical checks after controller
corrections to the SIMD matrix transpose argument and causal tile bound.
Its two-pass, scratch-based implementation is slower: GPU medians at N1024
are 7.3288 ms for D192 and 11.0849 ms for D256 (best tested key tile 8).
This is a negative performance result, retained in `metal-matrix-*.json`.

`VeriTac/Attention/Partition.lean` and its examples compile independently.
`merged_equals_direct` connects the executable contiguous-range checker to
normalized real attention semantics. Empty ranges, overlap/gap rejection,
and irregular tails are covered. No new axioms or admitted proofs are used.

The next performance experiment specializes the installed MLX steel template
with smaller FP32 query/key tiles. This is explicitly vendor-derived, with
attribution and source provenance; any speedup is a schedule-specialization
result, not a claim of an independently authored faster attention algorithm.
The original shaders remain separately identified. Compiled-pipeline thread
and static shared-memory limits are checked in addition to resource estimates.

## Confirmed narrow Metal result

The installed MLX steel FP32 specialization with Q16/K8 at B1 H8 N2048 D192
beats the faster vendor calling path in three confirmation rounds. Each round
uses 10 warmups and 30 samples with 3 invocations per sample; seeds are 1234,
4321, and 90210, and candidate/vendor order alternates. Q/K/V hashes match.

| Round | Candidate wall ms | Fastest vendor wall ms | Wall speedup | Candidate GPU ms | MPSGraph GPU ms |
|---|---:|---:|---:|---:|---:|
| 1 | 1.9058 | 2.0823 | 1.093x | 1.7145 | 2.1001 |
| 2 | 1.9154 | 2.0845 | 1.088x | 1.7177 | 2.0992 |
| 3 | 1.9171 | 2.0834 | 1.087x | 1.7214 | 2.1068 |

This is a vendor-derived schedule specialization, not an independently authored
kernel win. GPU timing is compared with MPSGraph only, because the survey does
not expose equivalent MLX GPU timestamps. At N1024 the wall advantage is only
0.6–2.1%, so no robust end-to-end win is claimed there. D256 remains a loss.

Held-out numerical checks pass N=1/7/17/100/257/1536 at D192 with seed777.
At N257, additional checks pass zero inputs, uniform scores, constant values,
large tied scores (Q=K=100), peaked scores (random Q/K multiplied by16), and a
causal impulse at key128. The largest normalized error in that set is 0.3715
(the frozen per-element acceptance threshold is1).

Raw confirmation, held-out, and adversarial records are the `metal-d192-*.json`
files in `benchmarks/attention/results/`. The generated MLX source retains its
copyright notices with the package MIT license alongside it.

Metal API validation passes Q16/K8 at N17/100. Shader validation instrumentation
reports 43,520 bytes for that configuration (double the normal 21,760 bytes),
so the explicit resource guard prevents its launch on a 32,768-byte device.
The smaller Q8/K8 specialization passes both API and GPU shader validation at
N17/100. The retained stderr logs confirm both validation layers were enabled
and contain no validation error. This is not an instrumented validation claim
for the selected Q16/K8 configuration.


## Confirmed CUDA result and controller acceptance

The final CUDA implementation combines a lower launch-bound hint, reversed
logical query-block mapping, explicit async-copy drains, and (for long
sequences) MM1/epilogue scratch aliasing. The alias unlocks Q64/K128 at 86,016
bytes where the original 103,424-byte allocation exceeded GB10's 101,376-byte
opt-in limit. The native vendor three-component TF32 FP32 path is preserved.

Three paired rounds confirm wins at B1H8, N1024/2048/4096/8192, D192/D256.
Config25 is used for N1024, config26 for the longer shapes. Graph speedup ranges
are 1.033–1.054× at N1024, 1.055–1.112× at N2048, and 1.120–1.188× at
N4096/8192. Input hashes match; all 24 complete candidate cases pass eager and
graph numerical checks. Stock clocks and alternating run order expose the
variation rather than hiding it. Full evidence is in
`cuda-final-confirmation-summary.json` and its referenced raw records.

The controller replayed real compiled configurations through the Lean checker
before confirmation. The normalized profile, full catalogue, four .so hashes,
and relevant source/header hashes were identical before and after timing and
sanitizer runs (`cuda-final-binding-validation.json`).

Controller memcheck and racecheck runs are clean for config25 at N1024D192 and
config26 at N2048D256. The corrected final adversarial suite (seed777, N257,
both head dimensions, configs20/25/26) passes36/36, max normalized error0.1855.
The original purported causal impulse was actually uniform-score input;
`cuda-final-adversarial.json` replaces it with one nonzero value key so rows
before that key must stay zero. Old raw records are retained with this caveat.

The first full demo search could choose undrained config16 on timing alone.
That exposed an acceptance gap: successful numerical checks do not erase its
known async-copy racecheck warnings. The controller requires the compiled
explicit-drain contract and disabled V preloading before executing a candidate.
Undrained variants remain visible as Lean-legal planning/diagnostic entries but
are rejected at the execution gate. This policy is separate from Lean's
conditional resource/real-semantics proof.

Structural Lean proofs additionally cover reversal's exactly-once block
coverage, explicit logical output destinations, disjoint scratch lifetimes,
and a meaningful pending-copy drain condition. A vacuous placeholder drain
predicate was rejected during review and replaced before acceptance. No
hardware truth or floating-point compiler correctness is asserted by these
proofs. See `attention_schedule_proofs.md` and the final results report.


Final live demo acceptance: N4096D192, seed2026, selected config26; fresh graph
replay4.5934ms vs fastest vendor5.2714ms (1.148×). Retained23 entries:6 timed,
13 rejected by the execution gate,4 by Lean. Every stored trace replayed through
the final checker. `cuda-final-verified-search.json` is the accepted demo record;
`cuda-rejected-undrained-search.json` preserves the earlier rejected search.
