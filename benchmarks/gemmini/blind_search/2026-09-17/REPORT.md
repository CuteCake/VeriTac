# Live blind Gemmini optimization, 2026-09-17

The flat-command experiment produced byte-certified kernels for **5 of 6
registered tasks**. Four improved on VeriTac's registration-time generators;
one rediscovered an existing strategy; one exhausted its generation budget.
The accepted improvements reduce **modeled input DMA bytes by 20–41.7%**.
These are not hardware latency measurements.

The separately registered same-task compact-interface phase also completed:
**3 of 6 tasks** received byte certificates. It did not improve on the flat
phase's best costs and had more generation timeouts. Across both phases, all
eight final certified artifacts passed binding audits and independent local
execution checks; they cover five distinct tasks, not eight distinct tasks.

## What was fixed

- Computation: row-major, full-tile DIM16 int8 GEMM with int32 output, the
  existing overflow bound, and unchanged input/target contracts.
- Acceptance: the existing sequential Gemmini semantics, RV64 byte decoder,
  `checkExecutable_sound`, and concrete Lean kernel certificates. No Lean
  checker or formal semantics was changed for these proposals.
- Requested model: `dgxspark-glm/glm-5.3-flash`, through the user's configured
  private endpoint and OpenCode 1.18.31. Model weight/version hashes were not
  independently attested.
- Budget: four fresh model calls per task, 900 seconds per call, at most 4096
  expanded commands, and 600 seconds for final certificate checking. At most
  three tasks run concurrently.
- Exposure: task, instruction semantics/schema, and the model's own preceding
  proposals and factual validation feedback. No optimized reference kernel,
  generator template, or prescribed repair was supplied.

The tasks were deliberately selected to exercise different storage/aspect-ratio
conditions, not randomly sampled. Held-out means withheld implementation
examples in this experiment, not absence from model training data. Tool calls
were denied in fresh temporary directories outside the repository; the observed
global/ancestor instruction-file audit and resolved configuration are recorded.
This is not a claim of an OS-hermetic sandbox.

See [the protocol](PROTOCOL.md), [task registration](preregistration.json),
[formal acceptance fingerprint](acceptance_fingerprint.json),
[environment audit](environment_audit.json), and
[flat controller snapshot](source_snapshot.json).

## Flat-command results

SP/ACC are declared row capacities. Old best is the lowest-cost validated
baseline or feasible batched-B configuration available at registration.

| Task | M × N × K | SP/ACC | Old-best input bytes | Certified input bytes | Reduction vs old best | Outcome |
|---|---|---|---:|---:|---:|---|
| rows_tight | 80 × 16 × 32 | 32/32 | 4096 | 4096 | 0% | Existing B reuse rediscovered |
| columns_tight | 16 × 80 × 32 | 32/32 | 5120 | 4096 | 20% | A reuse across output-column pairs |
| square_tight | 48 × 48 × 32 | 32/32 | 7680 | — | — | Four generation timeouts |
| columns_cache | 16 × 80 × 32 | 80/16 | 5120 | 3072 | 40% | Resident A, streamed B |
| mixed_cache | 32 × 48 × 48 | 64/32 | 6912 | 5376 | 22.2% | Resident A plus B/A streaming-slot reuse |
| single_acc | 48 × 32 × 32 | 48/16 | 6144 | 3584 | 41.7% | Cached B and reuse at reversed traversal boundaries |

There were 24 model rounds: 16 native acceptances, one semantic rejection,
and seven generation timeouts. A native acceptance is a search filter; only the
five final kernels with successful Lean kernel certificates count as certified.
All five also passed independent local RV64 operand interpretation and six
complete-output command-model cases. See [the binding audit](flat_audit.json),
[summary](flat_summary.json), [all outcomes](suite_outcomes.json), and
[mechanism review](flat_mechanism_review.json).

The square task's failure does not establish mathematical infeasibility: the
model returned no completed candidate within four 900-second call limits.

## Compact-interface comparison

This second phase used the same model, tasks, checker and per-task budgets,
but accepted bounded loops/expressions and preserved original compact source in
feedback. It is an exploratory comparison of representation **and** feedback,
with one trajectory per task/phase, not a randomized causal estimate.

| Task | Flat certified input bytes | Compact certified input bytes | Compact outcome |
|---|---:|---:|---|
| rows_tight | 4096 | — | Three timeouts, one semantic rejection |
| columns_tight | 4096 | 4096 | Certified; no cost improvement over flat |
| square_tight | — | — | Four timeouts |
| columns_cache | 3072 | 3072 | Two rejections, then certified |
| mixed_cache | 5376 | — | Three timeouts, one semantic rejection |
| single_acc | 3584 | 4096 | Certified; higher cost than flat |

Compact had 24 rounds: 10 native acceptances, 4 semantic rejections, and 10
model timeouts. Three final byte certificates passed the independent RV64 and
six local full-output checks. See [registration](compact_preregistration.json),
[controller snapshot](compact_source_snapshot.json), [all outcomes](compact_suite_outcomes.json),
[summary](compact_summary.json), and [binding audit](compact_audit.json).

Reported usage was 80,022 input and 120,274 output tokens (200,296 total), with
10 timed-out calls of unknown usage. Summed model-call wall time was 13,844.032
seconds. The smaller reported token sum does **not** establish a total cost
saving: more calls timed out without usage. Shorter visible JSON also did not
reliably predict endpoint-reported output token counts. Certificate checks for
the three winners took approximately 19.3–22.0 seconds each.

## What the model actually discovered

**Self-repair without a prescribed fix.** On `columns_tight`, the first proposal
overwrote live partial sums while keeping addresses within capacity. The
symbolic output obligation rejected it. The model then held two output-column
tiles live, shared A across them, and stored both before recycling the
accumulators. Round 2 was certified; neither its program nor its repair was
written by the controller.

**Using more scratchpad instead of more accumulators.** On `columns_cache`, the
model kept the two A reduction tiles resident and streamed B through a third
scratchpad tile. This used one accumulator tile and 48 of the declared 80
scratchpad rows. The 3072-byte input count equals the total A+B input size for
this task. That observation is an input-traffic bound, not a latency optimum.

**Combining storage lifetime with retained weights.** On `mixed_cache`, round
2 used 5888 input bytes. Round 3 reduced this to 5376: it retained all three A
tiles for the first output-row block, loaded B into one streaming slot, computed
the first row block (latching B), then overwrote that scratchpad slot with the
second row block's A and computed using retained B. The exact emitted bytes
received a certificate. The combination of resident A, retained B, and reuse of
B's scratchpad slot for A was absent from the registration-time generators.

**Keeping useful data across a traversal turn.** On `single_acc`, the model
cached B for a column and reversed output-row traversal for the next column.
It recycled B storage after its last use to preserve A for the boundary row.
The winner's actual count is **14 loads = 3584 input bytes, 49 commands**.
Its rationale incorrectly claimed 2560 bytes and 45 commands. All scores and
reported gains use the actual command stream; model claims of minimum cost or
global optimality are not accepted as evidence. Its fourth-round proposal was
worse, and the controller retained the certified third-round winner.

## Concrete rejection witnesses and safe fallback

Two rejected model programs now have independently kernel-checked concrete
counterexamples: [flat columns round 1](rejection_witnesses/flat_columns_tight_round1_v3/result.json)
and [compact rows round 2](rejection_witnesses/compact_rows_tight_round2/result.json).
Both use sparse 0/1 inputs and prove the decoded bytes return 1 at a cell whose
specified value is 0. The proof also checks input domains, plan/layout validity,
cell bounds, completed execution, and the reported values. These witnesses were
created after the proposals and were not fed into either registered phase.
The initial flat witness proof attempt failed elaboration and is preserved as
`kernel_proved: false`; only the successful final versions support this claim.

The checker is conservative: integer reduction-order changes can be correct yet
fail its ordered symbolic comparison. The witness helper therefore compares
commutative polynomial coefficients and executes candidate sparse inputs; it
does not label every rejection a numerical bug. This is a bounded search and
absence of a witness does not prove correctness.

For `square_tight`, the offline selector independently certified the existing
batched-B recipe at 7680 input bytes in approximately 33 seconds, with independent
RV64 and six local complete-output checks. This is a useful deployable fallback
within the declared model, **not a successful blind proposal**. See
[offline fallback](offline_fallback/square_tight/selection.json).

## Cost and evidence boundaries

The flat phase reports 125,320 input tokens and 174,484 output tokens
(299,804 total) for completed responses with usage. Seven timed-out calls have
unknown token usage, so these sums undercount total consumption. The summed
model-call wall time is 13,320.567 seconds; tasks ran concurrently, so that is
neither elapsed experiment time nor GPU execution time. Provider-reported cost
zero is not evidence of zero compute cost. Reported output tokens must not be
equated with visible program text or interpreted as a complete accounting of
internal model processing.

The five final kernel checks took approximately 19.5–31.6 seconds each on the
development host with an already built Lean environment. These numbers concern
proof checking, not accelerator runtime.

The audit checks that the original model reply yields the delivered commands,
that the delivered binary equals the byte list embedded in the theorem, that
the task and metrics agree, and that the recorded correctness theorem uses only
`propext`, `Classical.choice`, and `Quot.sound`. It also reruns the native byte
predicate. The recorded kernel checks supply the separate formal evidence;
native replay or artifact hashes alone do not replace them.

New upstream Spike replay has not been performed. Automatic approval review
blocked uploading/executing these artifacts on Spark3 because authorization for
that destination/payload was not explicit; a user authorization question is
pending. Local interpreter evidence is labeled separately. Sequential-to-
asynchronous hardware correspondence, caller/loader assumptions, physical
conformance, floating-point semantics, and measured hardware speed remain open.

## Functionality added

- A live, bounded proposal/feedback controller with immutable task/run
  configuration, source fingerprints, strict JSON/command parsing, tool-denied
  sessions, separate failure classes, and final byte certification.
- Advisory diagnostics for instruction/resource/initialization failures and
  symbolic output mismatch. They do not authorize acceptance or prescribe a
  repair.
- A bounded loop/expression adapter that does not execute model-generated Python.
  Expanded programs use the same byte checker. The separately registered compact
  experiment measures this interface and its source-preserving feedback.
- Optional sparse numerical counterexamples with kernel-checked rejected-byte
  execution, separate from acceptance and from the registered model feedback.
- Four reusable schedule families generalized **after** the corresponding
  discoveries were certified: batched A reuse, resident A, resident A with a
  shared B/A streaming slot, and serpentine B caching.
- An offline `auto` selector over known recipes. It ranks actual traffic/counts,
  validates candidates in that order, and kernel-certifies the chosen body,
  without another model call. It does not claim global optimality.

These generalizations are controller work, not additional blind-model results,
and are excluded from the registration-time controls. New concrete shapes have
their own certificates and local execution evidence under [generalization/](generalization/).
The Python generators remain untrusted; each accepted instance still requires
validation. The formal ISA/arithmetic coverage has not expanded.

See [usage and implementation details](../../../../docs/gemmini_blind_search.md).


## What this milestone establishes, and what remains

Within the declared exact-integer sequential Gemmini subset, a live proposer
received no optimized reference implementation, produced optimizations absent
from the registered generators, repaired a rejected proposal using factual
feedback, and delivered bytes checked against a fixed specification for all
inputs in its domain. The formal acceptance implementation stayed unchanged.
This is a restricted instance of the intended milestone, not universal AI kernel
verification or a CUDA/FP32 implementation.

Next experiments should prioritize (1) a sound arithmetic normalization theorem
that permits valid integer reduction reorderings, with deliberately equivalent
and inequivalent mutations; (2) upstream execution/conformance and independently
measured runtime, keeping traffic separate from latency; and (3) larger held-out
shape/resource sets and repeated fixed-budget trials before making claims about
interface or model reliability. Floating-point reassociation requires a new,
explicit arithmetic contract; the present exact-integer proof cannot be reused
as that justification.

The [final validation record](validation.json) and [test log](regression_tests.log)
record 266 local regression tests run successfully with one
environment-dependent skip. Recorded build artifacts and formal sources retain
their registration-time fingerprints. No commit or remote publication was made.
