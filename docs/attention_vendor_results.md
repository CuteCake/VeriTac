# Confirmed attention vendor comparisons

We found repeatable wins on both Apple M3 Ultra and NVIDIA GB10 by specializing
vendor-derived attention kernels. The original independently written Metal
experiments remain in the repository as negative results. These results do not
claim an independently authored replacement for the vendor algorithms.

## Contract and comparison method

All reported wins use causal prefill attention, batch 1, 8 heads, equal Q/K/V
sequence lengths, FP32 inputs/outputs, no dropout, and scale `D ** -0.5`.
Full outputs are compared with an independent NumPy float64 reference at
`atol=1e-4`, `rtol=1e-3`. No input-dependent shortcuts are used.

Each confirmation uses seeds 1234, 4321, and 90210, 10 warmups, 30 samples,
and 3 calls per sample. Candidate/vendor order alternates between rounds.
Q/K/V hashes match in every paired case. Speedups compare each round with its
fastest successful, numerically correct vendor backend. Unsupported backends
are retained in the raw results and never counted as wins. Clocks remain at
stock settings; the ranges below expose variation between runs.

## Metal: a narrow, repeatable win

The MLX 0.32.2 steel FP32 template, specialized to Q16/K8 at **N2048 D192**, uses
21,760 bytes of shared memory and beats the fastest vendor synchronized calling
path in all three rounds:

| Round | Candidate wall ms | Fastest vendor wall ms | Speedup |
|---|---:|---:|---:|
| 1 | 1.9058 | 2.0823 | 1.093× |
| 2 | 1.9154 | 2.0845 | 1.088× |
| 3 | 1.9171 | 2.0834 | 1.087× |

Both MLX fast attention and MPSGraph were measured. Candidate GPU medians were
1.7145–1.7214 ms versus MPSGraph's 2.0992–2.1068 ms. Equivalent MLX GPU timestamps
were unavailable, so the comparison against the fastest vendor is a **wall-call
latency** result. N1024 has only a small wall advantage; D256 loses.

Held-out lengths 1, 7, 17, 100, 257, and 1536 pass. Zero, uniform-score,
constant-value, large tied-score, peaked-score, and causal-value-impulse cases
also pass. API validation passes the winning specialization. Full shader
instrumentation doubles its shared-memory requirement beyond this device's
limit, so its launch is correctly rejected under instrumentation; the smaller
Q8/K8 variant passes both API and shader validation. We do not claim instrumented
shader validation of the winning Q16/K8 specialization.

Raw evidence: [Metal confirmation and search records](../benchmarks/attention/results/metal-d192-verified-search.json),
`metal-d192-{candidate,vendor}-round{1,2,3}.json`, and `metal-d192-*.json` in the
same results directory. Generated shader sources are content-addressed and keep
MLX copyright and license notices.

## CUDA: larger gains on longer sequences

The candidate specializes the installed PyTorch 2.14.0+cu130 CUTLASS memory-efficient
attention kernel for GB10 (`sm_121`, 48 SMs). It preserves the vendor FP32 path's
`OpMultiplyAddFastF32` three-component TF32 emulation; this is not a scalar IEEE
FP32 arithmetic proof or a switch to a lower-precision input contract.

All eight shapes win in all three confirmation rounds. Times are medians of the
three per-round medians; speedup ranges use paired per-round comparisons.

| N | D | Config | Candidate graph ms | Fastest vendor graph ms | Graph speedup range | Event speedup range |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 192 | 25 | 0.3896 | 0.4037 | 1.034–1.054× | 1.042–1.068× |
| 1024 | 256 | 25 | 0.4397 | 0.4555 | 1.033–1.050× | 1.039–1.053× |
| 2048 | 192 | 26 | 1.3076 | 1.4240 | 1.055–1.104× | 1.077–1.113× |
| 2048 | 256 | 26 | 1.5008 | 1.6343 | 1.075–1.112× | 1.089–1.098× |
| 4096 | 192 | 26 | 4.6712 | 5.3591 | 1.138–1.167× | 1.128–1.165× |
| 4096 | 256 | 26 | 5.4248 | 6.1341 | 1.123–1.159× | 1.137–1.166× |
| 8192 | 192 | 26 | 17.9627 | 20.9848 | 1.128–1.188× | 1.141–1.187× |
| 8192 | 256 | 26 | 20.5672 | 23.6837 | 1.120–1.167× | 1.129–1.167× |

The forced vendor backends are `auto`, `efficient`, `cudnn`, `flash`, and `math`.
Auto/efficient are fastest on this FP32 contract; flash and cuDNN report
unsupported. This is not a comparison with their BF16/FP16 kernels.
CUDA graph capture retains a distinct output buffer for every captured call,
matching the vendor harness. Graph results are numerically checked as well.

[Confirmation summary](../benchmarks/attention/results/cuda-final-confirmation-summary.json)
contains every paired result and links the observed catalogue to the
[Lean-checked plans](../benchmarks/attention/results/cuda-controller-checked-plans.json).
Raw files are `cuda-final-{short,long,vendor}-round{1,2,3}.json`.
[Binding validation](../benchmarks/attention/results/cuda-final-binding-validation.json)
confirms all four compiled binaries and relevant source/header hashes remained
unchanged across the confirmation and sanitizer runs.

## Gaps and transformations

1. **Compiler resource pressure.** Lowering the minimum-block launch hint removes
   reported local storage for the selected specializations and permits more
   registers. This is measured compiler output, not an assumed occupancy guarantee.
2. **Unequal causal block work.** Reverse the mapping between physical block
   indices and logical query blocks. All four query-start calculations change
   consistently, preserving masks and logical output destinations. The observed
   gain is empirical; CUDA does not promise a particular physical execution order.
3. **Scratch lifetimes.** FP32 MM1 and epilogue scratch can share a union when V
   preloading is disabled. Explicit async-copy drains precede the existing block
   barriers. Q64/K128 shared memory falls from 103,424 to **86,016 bytes**, fitting
   GB10's 101,376-byte opt-in limit. The larger tile then reuses K/V across more
   queries. Config 26 combines this with reversal; config 25 uses reversal and
   drains with Q32/K128 for short sequences.
4. **Abstract attention tiling.** Stable-softmax summaries over disjoint causal
   key partitions can be rescaled and merged without changing real-number
   attention. This supports the Metal tiling tactics and CUDA catalogue selection.

The slower original Metal partition kernel and original matrix kernel are
retained. CUDA over-budget tiles, unsupported K192 templates, spilling launch
hints, and slower configurations are retained as unsuccessful experiments.
The early `alias-sweep.json` graph timings reused a single output buffer and are
not confirmation evidence; the final graph protocol above corrects that issue.
The early adversarial file's purported impulse was uniform-score input; the
corrected final suite uses one nonzero value key and tests causal exclusion.

## Verification boundary

Lean proves the real-number partition/merge identities, accepted-plan resource
legality given observed hardware/compiler facts, block-permutation structure,
and conditional scratch-lifetime reasoning. Hardware observations, compiler
lowering, floating-point error, and GPU memory behavior are separate trust and
validation boundaries. Source/binary hashes bind the selected catalogue entry
to the tested executable; they are not a compiler-correctness proof.

The final CUDA adversarial suite passes **36/36** cases (seed 777, N257,
D192/D256, configs 20/25/26); maximum normalized error is 0.1855, below the
acceptance threshold 1. Memcheck and racecheck pass with **zero errors and zero
warnings** for config 25 at N1024 D192 and config 26 at N2048 D256. The earlier
undrained kernels' async-copy warnings remain documented as failed checks.
See [adversarial results](../benchmarks/attention/results/cuda-final-adversarial.json)
and `controller-final-{memcheck,racecheck}-cfg{25,26}.log`.

Implementation and proof details: [optimization log](attention_optimization_log.md),
[CUDA specialization](cuda_attention_specialization.md), and
[scheduling proofs](attention_schedule_proofs.md).

## Run the demos

With the prepared Spark3 environment and local `lake build veritac`:

```bash
python3 examples/llm_tiled_attention_demo.py --mock --search
python3 examples/llm_cuda_attention_demo.py --mock --search
python3 examples/llm_cuda_attention_demo.py --mock --seq 8192 --dim 192 \
  --mem-cap-bytes 34359738368
```

The mock path requires no model API. The LLM path receives the exact normalized
hardware profile, compiled resource catalogue, current plan, and Lean feedback.
The controller checks every proposed selection, retains rejected search entries,
replays the winner's own tactic sequence, verifies executable bindings before
and after dispatch, and compares matching vendor inputs. Reports remain in
unique `.lake/{tiled,cuda}_attention_demo/run_*` directories.


## End-to-end demo validation

The final live CUDA search at N4096/D192 (seed2026) retained all23 catalogue
entries: 4 were rejected by Lean's resource check, 13 undrained variants were
rejected by the controller's execution policy, and 6 were benchmarked. It chose
config26, replayed that candidate's own tactic sequence, and measured a fresh
**4.5934ms graph replay versus 5.2714ms** for the fastest vendor (**1.148×**).
All candidate traces replay identically through the final Lean build, and full
compiled config_info fields (including drain/preload flags) match the frozen
catalogue. [Final verified search](../benchmarks/attention/results/cuda-final-verified-search.json)
retains every rejection, accepted candidate, final replay, and raw vendor status.

The Lean library and CLI build, attention-plan tests, hardware-profile tests,
legacy soundness/matmul checks, and both Lean attention test files pass. The
[axiom audit](../benchmarks/attention/results/lean-attention-axiom-audit.txt)
shows only Lean's standard logical axioms for the semantic theorems; the
pending-copy conditional theorem uses no axioms. New attention examples use
kernel `decide`, with no admitted proofs or `native_decide` invocations.
The live runs use mock/deterministic search; the optional LLM API path reuses
the existing client and was not exercised against a live model API.
