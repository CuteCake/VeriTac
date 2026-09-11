# CUDA attention through verified tactics

Status: updated design direction, 2026-09-11. The original CUDA proposal below
is retained as implementation background; the architecture and next milestone
in this opening section supersede its fixed-registry scope and backend-first
milestone order. The [main compiler design](../veritac_design.md) is the shared
architecture reference.

## Goal: a checked path from attention mathematics to instructions

VeriTac should supply the target verification layer itself, including for a new
ASIC with no existing compiler or target-IR validator. The inputs are a workload
specification and formal instruction/hardware semantics. The output should be an
instruction program and a composed Lean proof connecting it to that specification,
conditional on the stated input, numerical, and hardware-model assumptions.

Explore a graph spanning equivalent algorithms, schedules, storage layouts,
parallel programs, and instruction selections. Exact rewrites may share
proof-producing e-classes; lowering refinements and error-bounded transformations
need their own relations and composition rules. They must not be treated as
unconditional equalities. Every accepted edge identifies its source/destination,
semantics version, obligations, and checked proof.

LLM guidance and enumeration both expand this graph. AI-directed development of
new rules and proofs is part of the loop, including the controller/OpenCode work
already performed here. A proposal may instantiate an existing tactic or introduce
a new transformation with a proof obligation. Lean feedback drives repair and
branching; only discharged obligations admit an edge. Vendor compilation and
reference comparisons are useful diagnostics and performance measurements, not
the final semantic authority.

## What the attention experiments establish today

The current Metal/CUDA demos establish abstract real-number partition/merge
identities, conditional resource and plan legality, block-permutation properties,
and conditional scratch-lifetime reasoning. AI-directed vendor-derived kernel
specialization plus enumeration produced repeatable wins on selected FP32 shapes.
The [results report](attention_vendor_results.md) records the complete scope.

The connection between those proofs and the executable shaders/CUDA kernels is
currently established by source review, catalogue/source/binary binding, numerical
tests, and sanitizers. There is no composed proof from attention's mathematical
specification through lowering to the executed instruction program yet. Completing
that connection is the architectural goal; installing an LLM API key in the demo
would change orchestration, not close the proof gap.

## Next milestone: a minimal target with no external correctness oracle

Start with a small dot product or GEMM on a deliberately minimal ISA, using exact
bounded arithmetic and explicit overflow conditions. Define register, arithmetic,
memory/addressing, control, and synchronization semantics. Connect mathematical,
scheduled, and instruction programs with checked refinement proofs. Include a
small interpreter to demonstrate execution; use Lean's composed theorem as the
correctness evidence.

The demonstration must show an AI proposing an invalid step, Lean reporting an
unmet obligation, the AI repairing it, and the final accepted path reaching target
instructions. Retain both failed and repaired attempts. Include enumeration and
at least one newly proposed, checked transformation so the graph can grow beyond
a fixed catalogue. No vendor compiler or reference kernel is required for this
acceptance test. If emitting encoded instructions, prove the encoding relation or
identify it as a remaining boundary before making a machine-code claim.

Then connect attention's partition/online-softmax proofs through layout, ownership,
synchronization, and instruction selection. Real-number algebra must be bridged
to explicit floating-point semantics or a proved approximation contract; numerical
tolerances alone do not discharge that obligation.

## LLM–harness–compiler co-design (planned)

The [main design's co-design plan](../veritac_design.md#llm-harness-compiler-co-design)
connects the model's proposal interface, the exploration harness, and compiler
semantics/proof interfaces. Attention provides concrete tasks for this work:
partition selection, storage-lifetime changes, synchronization, and instruction
mapping should expose their typed edits and localized obligations directly.

The harness will combine enumeration with LLM proposals, retrieve relevant
lemmas, reuse stable context, schedule bounded work, and return structured Lean
feedback for repair. Repeated proof patterns can become new tactics; failures
can reveal missing invariants or poorly chosen IR/action boundaries. Changes to
the theory remain versioned and require rechecking dependent proofs.

Evaluate these interface choices with the minimal-ISA proof path first, then
attention: compare time/cost to a complete verified implementation and kernel
performance under equal search budgets. Include ablations for caching, feedback,
and action granularity. The target numerical contract and Lean proof requirements
stay fixed across those comparisons. This workstream is planned; current
controller/OpenCode experiments provide experience, not a completed co-design
harness.

## Efficient proposal harness (proposed)

Keep the workload contract, target instruction semantics, proof interfaces, and
versioned lemma catalogue in a stable prompt prefix. Append only the selected
frontier, current obligations, relevant failed attempts, and cost feedback. Retrieve
proof context by content hash, and group requests sharing the same target/theory.
Summaries guide the proposer; the checker loads the actual referenced artifacts.
Changes to semantics or numerical policy invalidate affected caches. Measure
prefix-cache hits, latency, cost per accepted edge, and kernel quality; caching
is a proposed efficiency improvement and never part of correctness acceptance.

## Original CUDA implementation proposal

The following sections record the initial 2026-09-09 backend proposal. Historical
repository audit and hardware/environment observations are dated evidence, not a
current inventory. Consult the implementation and final results for current status.

### Initial backend recommendation

Build `examples/llm_attention_demo.py`: an LLM proposes one scheduling tactic per
turn, Lean checks it, CUDA code is generated and measured on Spark3, and the
measurement feeds the next turn. Keep the best validated candidate and save a
replayable trace. Provide `--mock` with the same verification and execution path.

The workload is **causal scaled dot-product prefill attention**. Start with a
three-stage implementation (`QKᵀ`, stable softmax, `PV`), then introduce a proved
online-softmax transformation and block-local fusion. The completed demo should
show a tactic sequence eliminating the quadratic intermediate, followed by
measured tuning of GPU mapping and memory use. A three-stage kernel alone is an
intermediate milestone.

For the initial backend prototype, emit CUDA C++ from structured IR and use
registered tactics as the starting interface. The broader architecture also
accepts AI-proposed transformations and lowerings once their proofs are checked.
An arbitrary source proposal without that proof path remains experimental.
The initial backend proposal starts with FP32 and SIMT execution.
BF16/FP16, tensor cores, asynchronous copies, decode, GQA, backward, and dropout
are subsequent extensions. No speedup over a vendor attention kernel is assumed.

## Initial repository audit (2026-09-09)

The existing GEMM demo provides useful orchestration: configuration, single-step
proposals, JSON calls to Lean, rejection feedback, and statement rendering.
However, its final-only benchmarking cannot guide hardware optimization.

The implementation has several boundaries that the new path must address:

| Current code | Consequence for this design |
|---|---|
| `Schedule/LoopNest.lean`: values and indices are both `Int` | Real attention needs typed scalar operations, ordered reductions, and masking. The generic type on `TExpr` does not make scheduled execution generic. |
| Lean and C flatten indices in base 1000 | This is unsuitable for attention layouts and collides when an inner coordinate reaches 1000. Use explicit shape/stride descriptors and bounds proofs. |
| `execStmt` ignores annotations and allocation scope | Existing annotation equivalence does not prove GPU execution or scratch-memory safety. |
| `Compose/Engine.lean`: `applySchedule_correct` concludes `True` | Acceptance must be connected to transformation theorems before reporting a verified schedule. |
| `reorder` checks bounds; its theorem additionally requires commutation | The CUDA checker must discharge read/write dependence obligations or reject the transformation. |
| Lean `checkIndependence` and Python `writes_local_to` check variable occurrence | Neither proves injective writes or absence of cross-iteration reads. For example, `C[i % 2]` can collide and `C[i] = C[i-1]` has a dependency. |
| `tile`/`fuse` theorems have stronger assumptions than their dispatchers enforce | Check freshness, binder capture, admissible bounds, and divisibility at the accepted transformation boundary. |
| `cache_read` dispatch supplies empty copy shape/indices | It cannot serve as a shared-memory staging primitive; copying and synchronization need their own proof. |
| GEMM benchmark checks four output cells | Attention validation must compare the entire output. |

Reuse the loop-transformation proof techniques, not an assumption that every
current dispatcher is already sound. Add a versioned CUDA path without changing
the existing CPU demo's JSON contract. Initially expose only a restricted tactic
subset whose checker-to-theorem connection is complete.

## Workload contract

Inputs are contiguous `Q, K, V : [B,H,N,D]`; output is `O : [B,H,N,D]`.
For each batch/head/query row `q`:

```text
s[q,k] = sum_d Q[q,d] * K[k,d] / sqrt(D)
P[q,k] = exp(s[q,k] - max_{j <= q} s[q,j])
         / sum_{j <= q} exp(s[q,j] - max_{r <= q} s[q,r])   if k <= q
         0                                                otherwise
O[q,d] = sum_{k <= q} P[q,k] * V[k,d]
```

Require positive dimensions and finite inputs within a declared validation
domain. Every causal row contains at least one key. Represent the mask explicitly;
do not implement it by an arbitrary large negative constant. Padded/inactive
rows and fully masked tiles must bypass the update, avoiding `-inf - -inf`.

Proposed starting cases:

| Purpose | B | H | N | D |
|---|---:|---:|---:|---:|
| Small correctness | 1 | 2 | 32 | 32 |
| Interactive tuning | 1 | 8 | 256 | 64 |
| Primary prefill demo | 1 | 8 | 1024 | 64 |
| Generalization | 1 | 8 | 2048 | 128 |

Add irregular cases such as `N=127,257` once guarded tiling is proved. Before
that milestone, reject unsupported shapes explicitly. Noncausal attention is a
useful validation variant, but arbitrary masks and unequal query/key lengths
are outside the first public contract.

## Hardware context: one profile, two uses

Read-only inspection of `spark-379a.tail64c925.ts.net` found the following. Device
limits were queried through `libcuda.so.1` using attribute identifiers from the
installed CUDA header; compiler information came from `nvcc`.

| Property | Observed value |
|---|---|
| Host architecture / GPU | aarch64 / NVIDIA GB10 |
| Compute capability | 12.1 (`sm_121`) |
| SMs / warp size | 48 / 32 |
| Maximum threads per block / SM | 1024 / 1536 |
| Maximum block dimensions | 1024 × 1024 × 64; also enforce the total-thread limit |
| Maximum grid dimensions | 2147483647 × 65535 × 65535 |
| Shared memory per block, default / opt-in | 49,152 / 101,376 bytes (48 / 99 KiB) |
| Shared memory per SM | 102,400 bytes (100 KiB) |
| Registers per block / SM | 65,536 / 65,536 32-bit registers |
| Maximum resident blocks per SM | 24 |
| L2 cache | 25,165,824 bytes (24 MiB) |
| Reported total device memory | 130,662,940,672 bytes; this is not available allocation capacity |
| Driver / compiler | 580.173.02 / CUDA 13.0, nvcc 13.0.88 |

`/usr/local/cuda/bin/nvcc` lists `sm_121`. Compute Sanitizer and Nsight Compute
exist in that directory, which is absent from the default SSH command path.
The inspected system Python cannot import torch, numpy, or cupy; other virtual
environments were not surveyed. The implementation should use an isolated
environment and explicit toolchain paths. You are allowed to setup a python venv under Documents/codework

NVIDIA also identifies GB10 as compute capability 12.1 in its
[GPU table](https://developer.nvidia.com/cuda/gpus). Do not infer support for a
particular matrix instruction or copy primitive from the name “Blackwell.”

Introduce `Hardware/profile.py` and a small CUDA probe producing versioned JSON:

```json
{
  "schema_version": 1,
  "target": {
    "backend": "cuda", "compute_capability": [12, 1],
    "warp_size": 32, "max_threads_per_block": 1024,
    "shared_mem_per_block_default_bytes": 49152,
    "shared_mem_per_block_optin_bytes": 101376
  },
  "performance_hints": {"sm_count": 48, "l2_bytes": 25165824},
  "toolchain": {"nvcc_version": "13.0.88", "code_target": "sm_121"},
  "provenance": {"source": "cuda_device_query", "host": "spark-379a"}
}
```

This abbreviated example omits dimensional/SM limits shown above. The full
schema includes them, device UUID, driver/runtime versions, observation time,
and feature compile-probe results. Unknown values remain unknown and cannot
authorize a feature. Hash the canonical profile and record it with artifacts.

**Lean consumes capabilities and limits as explicit data.** It proves statements
conditional on that target description; it does not prove the physical GPU
matches a supplied JSON file. Check schema invariants, use unit-bearing names,
and re-probe the actual device before running a saved schedule. Profile drift
requires revalidation, not silent reuse of an old legality result.

**The prompt consumes a rendering of the same profile**, plus performance hints,
current resource estimates, measured timings, and currently available tactics.
The model cannot edit the target, numerical policy, or validation thresholds.
Estimated bandwidth, occupancy, and register pressure are advice, not axioms.

Validate block/grid sizes, exact allocated shared-memory bytes including
alignment and padding, memory budgets, and feature support before compilation.
Separate whole-block legality from warp-collective requirements: only tactics
using full warps require an appropriate multiple of `warp_size`.

After compilation, inspect actual registers, static/dynamic shared memory and
function limits, apply shared-memory opt-in where necessary, and check occupancy
and launch success. Register allocation cannot be known exactly from source IR.
CUDA exposes device properties through
[`cudaDeviceProp`](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/structcudaDeviceProp.html)
and compiled-function resources through
[`cudaFuncAttributes`](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/structcudaFuncAttributes.html).

## IR and proof architecture

Introduce a small version-2 kernel IR alongside the current integer IR. Keep
integer index expressions separate from typed scalar expressions. Scalar nodes
include literals, reads, add/multiply/subtract/divide, max, exp, and select.
Represent ordered reductions and their identities explicitly. Shapes, strides,
buffer roles, allocation scope, and output initialization belong in the IR.
Use stable node IDs so tactics target a particular loop or stage unambiguously.

Use a scalar-semantics parameter for structural proofs: transformations that
preserve each output's operation order should not need arithmetic associativity.
Instantiate a real-valued semantics for attention identities. Keep the dtype,
reduction topology, exp implementation and compiler arithmetic flags in an
explicit numerical policy; do not reinterpret the existing integer division
as floating-point division.

Lower to a restricted GPU IR with block/thread bindings, private storage,
block-shared allocations, load/compute/store phases, uniform barriers, and
explicit collectives. Its semantics must represent storage ownership and
synchronization. An annotation-erasing serial interpreter is insufficient.
Begin with bulk-synchronous blocks and no communication between blocks.

The checked state includes the program, buffer descriptors, schedule, launch
plan, and resource analysis. Acceptance should establish relations of the form:

```text
checkApply(target, policy, before, tactic) = accepted(after)
  -> SemanticRelation(policy, before, after)
  ∧ WellFormed(after)
  ∧ MemorySafeInModel(after)
  ∧ RaceFreeInModel(after)
  ∧ TargetLegal(target, after)
```

Prove that the executable checker implies these predicates, including recursive
application inside a program. Then prove the composition result for the accepted
subset. A theorem name or a JSON `verified: true` field is not evidence by itself.
Serialize diagnostics and canonical IR; proof terms can remain inside Lean.

Use two clearly reported semantic relations:

* **Operation-order preservation:** structural transformations preserve the
  ordered scalar computation under fixed scalar semantics.
* **Real-arithmetic equivalence:** online softmax and reassociated reductions
  preserve the mathematical attention function. Their FP32 realization is
  numerically validated, not claimed bitwise equivalent or formally error-bounded.

Mixed schedules report the weaker, real-arithmetic guarantee. CUDA emission,
nvcc, the driver, actual hardware, and the FP realization remain outside the
initial proof boundary. Testing and sanitizers check that boundary empirically.

## Tactics and the optimization path

| Tactic | Effect | Required evidence |
|---|---|---|
| `tile`, later `tile_guarded` | Split query/key/output loops | Bijection or guarded coverage, bounds and fresh binders; preserve reduction order |
| `fuse_axes`, `reorder` | Restructure independent work | Domain equivalence and full read/write noninterference; reject unknown dependence |
| `bind_block`, `bind_thread` | Map iterations onto CUDA execution | Coverage, unique ownership, no cross-worker dependence, legal dimensions |
| `unroll` | Expand a small literal loop | Same ordered computation, fresh binders, code-size budget |
| `stage_shared` | Cooperatively load a read-only tile | Complete initialized copy, valid local indices, uniform barrier before use and before overwrite |
| `promote_private` | Keep accumulators in thread-private storage | Unique owner, initialization and final writeback equivalence |
| `online_softmax` | Replace materialized normalization with streaming state | Real-valued prefix invariant, positive denominator, causal mask handling |
| `fuse_attention` | Place a query tile's stages in one block | Producer/consumer locality, scratch lifetimes, phase synchronization, same output |
| later `reduce_warp` | Choose an explicit reduction tree | Lane coverage, collective participation and masks; real equivalence plus numeric validation |

The existing `fuse` merges nested iteration axes; it does not fuse attention
stages. Keep those concepts distinct in the registry and prompt. Do not expose
raw `barrier` insertion as a free scheduling knob: staging/fusion tactics produce
the required synchronization together with their evidence.

Start with a legal CUDA baseline for the three-stage graph. For the fused
version, assign one block to `(batch, head, query_tile)`, iterate key tiles in
order, stage K/V cooperatively, and maintain row maxima, normalizers and output
accumulators locally. Blocks write disjoint output tiles. Preserve serial
reductions first; introduce cooperative reductions only with their own contract.

For scores in the processed valid-key prefix, maintain:

```text
m = max(score)
l = sum exp(score - m)
u = sum exp(score - m) * V
O = u / l
```

For a new nonempty tile with local maximum `m_t`, set `m' = max(m,m_t)`, rescale
old `l,u` by `exp(m-m')`, and add the new tile's contributions relative to `m'`.
Use an explicit empty-state constructor for initialization and skip tiles with
no valid keys. Prove the prefix invariant by induction, then prove equivalence
to materialized softmax. This follows the IO-aware streaming approach described
in the [FlashAttention paper](https://arxiv.org/abs/2205.14135).

Initial tuning choices: query tiles `{8,16,32}`, key tiles `{16,32,64}`, and
blocks of `{64,128,256}` threads, filtered by coverage and resource checks.
These are proposed search bounds, not measured winners. For example, staging
FP32 K and V with `Bk=64,D=128` alone uses `2*64*128*4 = 65,536` bytes, already
above the default shared-memory limit; scores, Q, padding and other scratch
must still fit. Register spills and occupancy may make a legal tile slower.

## Search, execution and trace

Keep the controller and Lean verification local; use Spark3 as a CUDA worker.
Transfer only candidate source, descriptors, deterministic test inputs and
fixed runner requests into a dedicated run directory. Compile on aarch64 and
return structured results. LLM credentials stay with the controller. Running
the entire pipeline directly on Spark3 can be added as another worker backend.

Each turn:

1. Present workload, policy, target profile, current IR, allowed tactic schema,
   best timings, recent failures, and resource use.
2. Parse one tactic and apply it transactionally through Lean. On failure retain
   the parent and return a structured diagnostic with node and failed predicate.
3. If the intermediate state is launchable, compile, validate all output values,
   and benchmark. Some steps leave a legal partial schedule awaiting binding;
   label these `verified_partial` and never report an invented timing.
4. Save the result. Maintain separate current and best validated states so a
   temporarily slower transformation can enable fusion. Permit returning to a
   saved parent through a controller action; it is not a compiler tactic.
5. On `done`, replay the best schedule from the original IR, revalidate it and
   report the confirmed measurement. A failed candidate never becomes the winner.

Preserve a bounded history: the current GEMM loop reconstructs messages each
turn, so appending failures alone does not preserve them across turns. Cache
compilation by IR/profile/compiler/flags hash; deduplicate repeated proposals.
Apply fixed time, step, source-size, allocation and compile budgets.

Save `hardware.json`, workload/policy, original/final IR, schedule, trace JSONL,
CUDA source, compiler flags and logs, resource reports, validation results and
raw timing samples. Proposed trace states include `lean_rejected`,
`verified_partial`, `compile_failed`, `resource_rejected`, `numeric_failed`,
`runtime_failed`, and `measured`. Mock mode exercises this same path; a separate
`--verify-only` mode supports development without a GPU.

## Validation and success criteria

Compare the complete output against an independent stable FP64 reference on
small cases and an independent PyTorch attention reference for larger cases.
Use a tested aarch64 CUDA PyTorch environment for comparisons; a small scalar
reference keeps basic correctness checks independent of its availability.
Report an unavailable library comparator explicitly rather than treating it as
passed. Keep vendor SDPA performance separate from the same-dtype naive CUDA
baseline; record backend selection, dtype, TF32 settings and mask semantics.

Use seeded random inputs plus zeros, repeated scores, large finite score
differences, first/last causal rows, and padding boundaries. Require finite
outputs and per-element `abs(error) <= atol + rtol*abs(reference)`. A proposed
initial FP32 policy is `atol=1e-4, rtol=1e-3`, frozen before search and checked on
held-out inputs; it is an empirical acceptance rule, not a numerical theorem.
Record max absolute error, relative error with an explicit denominator floor,
and RMS error. Never tune tolerance to rescue a candidate.

Run memory, shared-memory race, initialization and synchronization checks on
representative schedules and the final winner. These are the distinct tools in
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html);
passing racecheck does not prove all global-memory races absent.

Measure with CUDA events on one stream after warmup. Time the entire attention
pipeline, including intermediate kernels, with allocated buffers and input
transfers outside that interval. Report transfer-inclusive latency separately.
Batch short launches, collect medians and spread, interleave baseline/winner
remeasurements, and record contention and device state when available. Initial
defaults: 10 warmups and 30 timing samples, increased only if noise warrants it.

Success means a replayable mock and LLM-driven demo, checked transformations and
target legality for every accepted step, full-output validation and sanitizer
passes, and a repeatable speed improvement over the baseline on the primary
case. Report the vendor gap honestly. No numeric speedup target is justified
until the baseline runs.

## Original backend milestones (historical planning sequence)

| Milestone | Concrete deliverable / exit condition |
|---|---|
| 1. Trusted acceptance boundary | Versioned kernel IR and target schema; checked structural tactic subset with soundness/composition theorems; negative cases for collisions, dependencies, capture, unsupported bounds and invalid launches |
| 2. First CUDA demo | Hardware probe, three-stage FP32 attention, emitter/worker, full-output reference tests, timing, mock trace; explicit supported-shape contract |
| 3. Streaming attention | Proved online-softmax invariant and block-local fusion/staging; no global N×N intermediate; mask/tail proofs and synchronization checks |
| 4. Measured LLM optimization | Shared proposer utilities, profile-derived prompt and registry, compile/resource/error feedback, best-candidate replay, confirmed Spark3 benchmark report |
| 5. Optional performance extension | Warp reductions, BF16/FP16 or supported tensor-core operations, each with explicit semantics and validation policy |

Suggested module ownership:

```text
VeriTac/Hardware/Target.lean           target data and legality predicates
VeriTac/Kernel/{IR,Semantics,Layout}.lean
VeriTac/Attention/{Spec,OnlineSoftmax}.lean
VeriTac/GPU/{IR,Semantics,Legality}.lean
VeriTac/Tactic/Cuda/*.lean             checked structural/mapping/staging tactics
VeriTac/Compose/Checked.lean           sound application and composition
Main.lean                            versioned request routing + tactic registry
Hardware/{profile.py,probe_cuda.cu}
CodeGen/{emit_cuda.py,cuda_runner.py}  deterministic emission and worker protocol
Search/{llm_client.py,attention_agent.py,interface.py}
examples/llm_attention_demo.py
tests/test_attention.py
tests/test_cuda_legality.py
```

The main scope choice for review is to finish the FP32 causal prefill path,
including streaming fusion, before adding low-precision tensor-core execution.
That gives the tactics system a substantive CUDA demonstration while making
its mathematical, execution-model, and empirical guarantees visible.

## Vendor comparisons and a Metal target

The local machine reports an Apple M3 Ultra, 80 GPU cores and 256 GB memory.
Metal is therefore a practical local target. No Metal or vendor attention
benchmarks have been run as part of this design review.

Use three baseline roles: an independent numerical reference; the simple
generated implementation to measure tactic gains; and a production library to
measure competitiveness. The vendor implementation is a comparator, not the
semantic specification or an automatically verified starting program.

| Target | Vendor comparator | Additional practical comparator |
|---|---|---|
| GB10 / CUDA | cuDNN SDPA, if a valid execution plan exists for the installed version, device, shape and dtype | PyTorch SDPA with its selected backend recorded; force supported backends separately |
| M3 Ultra / Metal | MPSGraph `scaledDotProductAttention` | Apple MLX `mx.fast.scaled_dot_product_attention` |

The [cuDNN attention documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
describes fused SDPA with low-precision input support. Its Blackwell support
table names B200/B300; that is insufficient evidence for a particular GB10
execution plan. Probe support explicitly and report unavailable configurations.
Do not silently benchmark a fallback and label it cuDNN fused attention.

Apple exposes [MPSGraph SDPA](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:))
and describes its fusion in [WWDC24](https://developer.apple.com/videos/play/wwdc2024/10218/).
[MLX fast attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
is another relevant Apple-maintained baseline, with inspectable source and
explicit causal-mask support. Use both on Metal; beating just one can reflect
library dispatch differences.

The inspected [MLX Metal dispatcher](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp)
has separate short-query and full-attention paths. Its full path supports a
selected set of head dimensions (64, 80, 96, 128 in the inspected source),
suggesting larger head dimensions as candidate specialization opportunities.
This is a hypothesis to validate against a pinned installed version, not a
performance result or a claim about MPSGraph's internals.

Benchmark two tracks:

* **Common fused cases:** causal prefill with `D=64,128` and several sequence
  lengths. These test competitiveness against the libraries' established paths.
* **Coverage opportunities:** realistic `D=192,256` and irregular sequence lengths.
  Record whether the library fused the computation. A speedup over an unfused
  fallback is useful but must be identified as a coverage improvement.

Use the same mathematical operator, inputs, dtype, layout, mask convention,
output requirements and error policy. Keep an FP32 correctness track and a
matched FP16 performance track where supported, with explicit accumulation
behavior. FP32 SIMT versus FP16 matrix hardware is not an apples-to-apples
speedup comparison. Moving FP16 earlier does not remove the need for a numerical
contract and tests; matrix-instruction tactics remain a separate proof task.

Compare GPU execution and synchronized API latency separately. Warm compiled
graphs/pipelines and materialize lazy inputs before timing; force MLX output
evaluation and synchronization. Record library versions, selected execution
paths, and setup/workspace costs. Give every implementation equivalent caching
and command-submission treatment. Report the full predefined case matrix and
win/loss distribution, not just the winning shape. Compare each generated kernel
with baselines on its own device; cross-device latency also reflects hardware.

For portability, express shared scheduling concepts as workgroups, threads,
subgroups, workgroup memory, private memory and synchronization phases. CUDA
lowers these to blocks/warps/shared memory; Metal lowers them to
threadgroups/SIMD groups/threadgroup memory. Reuse attention identities, layout,
coverage and ownership proofs. Backend-specific memory semantics, collectives,
matrix instructions and emission obligations still need separate treatment.

A Metal profile queries device features and memory limits plus compiled-pipeline
`threadExecutionWidth`, `maxTotalThreadsPerThreadgroup` and static threadgroup
memory. Apple documents these in its
[Metal limits](https://developer.apple.com/metal/limits/). CUDA register/occupancy
fields cannot be assumed available or assigned invented Metal equivalents.
Unknown fields remain explicit. Add `emit_metal.py`, a native Metal runner and
Metal legality checks only once this target is selected for implementation.

Recommended next milestone: a baseline survey on both devices, then choose one
backend and workload family for the first competitive demo. Metal specialization
is promising because the local development loop is convenient and dispatch
coverage is inspectable. There is not yet evidence that it is generally easier
to beat Apple's optimized kernels than NVIDIA's. The survey should decide that
choice while preserving the original CUDA objective.
