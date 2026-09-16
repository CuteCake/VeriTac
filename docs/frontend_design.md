# A meta-frontend for verified specialization

Design proposal, 2026-09-16. This describes the next architecture; the existing
Gemmini JSON checker and Lean APIs remain the implemented interfaces.

The reusable frontend is a method for constructing a precise interface for a
particular computation and target. Its common structure establishes meaning,
scope, proof obligations, and evidence. Each specialization chooses the language
and representations that make its computation and hardware understandable.

Specialize around a **model family, numerical contract, target execution model,
and intended deliverable**. A platform name alone is insufficient: inference
and training, dense and sparse operations, or batch and streaming execution can
need different interfaces on the same device. Conversely, several devices can
share an interface when their semantic contracts support it.

## 1. The questions every specialization must answer

These are obligations to explain, rather than mandatory fields in one large
schema. Each answer must identify its scope and its formal representation, or
explicitly identify an unresolved boundary. Irrelevant concepts can be omitted
with an explanation; missing concepts must not silently acquire default meaning.

| Concern | Required answer | What can vary |
|---|---|---|
| Observable behavior | What does the computation produce, and what counts as correct? | Tensor functions, state transitions, streams, distributions, or protocols |
| Input domain and state | Which shapes, values, layouts, aliasing relationships, initial states, and external interactions are allowed? | Fixed kernels, dynamic graphs, persistent caches, sparse metadata, streams |
| Numerical meaning | Which input, product, accumulation, cast, rounding, overflow, and exceptional-value semantics apply? What error is permitted relative to which reference? | Exact integers, floating point, quantization, approximation, probabilistic guarantees |
| Execution and resources | What can execute, where does data live, who owns it, and when are effects visible? What establishes completion? | Sequential instructions, concurrent threads, asynchronous DMA, dataflow engines |
| Allowed change | What may the optimizer alter while preserving the contract? | Algorithms, fusion, layouts, tiling, precision within an accepted budget, instruction selection |
| Acceptance evidence | Which relation connects each representation to the next, and how is it checked? | Equivalence, simulation/refinement, bounded error, concrete translation validation |
| Delivery and measurement | What artifact is delivered, what must its environment supply, and what performance objective is measured? | Raw bodies, libraries, linked executables, deployed models; latency, throughput, memory, energy |

Stateful and nondeterministic computations need observable traces or relations,
not necessarily a pure function. Concurrent correctness must account for the
allowed executions, with explicit progress/fairness assumptions when completion
depends on them. A successful execution does not establish correctness of all
allowed executions.

## 2. Separate the sources of meaning

A specialization brings together four independently versioned descriptions:

1. **Computation contract.** The authoritative mathematical behavior, input
   domain, numerical relation, and observables. Record whether weights are fixed
   artifacts or quantified inputs, and whether state persists between calls.
2. **Target contract.** Formal execution semantics plus separately identified
   capabilities and environment assumptions. Resource probes and ISA documents
   provide provenance; a formal model states the assumptions used by the proof.
3. **Representation and proof adapters.** The input language, its interpretation,
   candidate representations, proof-producing transformations or validators,
   and executable-output boundary. Several adapters may serve one contract.
4. **Optimization request.** Objectives, measurement protocol, search budget,
   and preferences among already admissible implementations.

This separation prevents three common mistakes. A performance estimate cannot
establish instruction behavior. A larger capacity declaration cannot establish
that a device has that memory. An optimizer cannot change the reference result,
input domain, or error tolerance to make its proposal pass.

Classify each constraint as a required property, a declared environment
assumption, a derivable fact, or a search preference. For example, peak scratch
usage is derived from a candidate; available scratch capacity comes from the
target description; fitting the former into the latter is a checked obligation.
Unknown required capabilities remain unresolved. Alternate contracts can be
explored as explicit branches, with acceptance reported under the new contract.

## 3. What is common, and what is specialized

The common orchestration layer needs a small artifact envelope: specialization
and contract identities, payload format/version and artifact references, the
claim being made, its assumptions, and associated obligations/evidence. For a
transformation it also identifies the actual source and destination artifacts.
Identity must cover the semantics and dependencies used to interpret a payload;
hashes establish identity, not semantic validity.

There is no common requirement that payloads be tensor graphs, loop nests, or
instruction lists. A specialization defines its own parser, typed views,
candidate edits, diagnostics, and output format. It may expose several views
of the same artifact, with checked connections where correctness depends on
their agreement. Cross-specialization edges require an explicit bridge relation;
similar names or shapes do not establish interoperability.

The shared lifecycle is:

1. Establish and identify the contract and intended delivery boundary.
2. Interpret the input, reporting the exact accepted scope and any unsupported
   constructs. Parsing, semantic validation, and proof acceptance are distinct.
3. Expose representations and unresolved obligations suitable for this task.
4. Propose changes; check the relevant claims against the fixed contract.
5. Compose accepted relations and return the artifact, theorem, assumptions,
   unresolved boundaries, and separately labeled measurements.

Keep status per claim. An artifact may be parsed, resource-legal, proved correct
at an abstract level, byte-certified, or independently executed. These are
different facts, not interchangeable meanings of a single `verified` flag.
Native checks can guide search quickly; final acceptance must satisfy the
specialization's stated proof policy, as the Gemmini kernel replay does today.

Importing an external model also creates an obligation: the internal contract
must mean what the source model means, including framework/operator versions.
Until that connection is proved or checked by a sound validator, name the
importer as a trust boundary. Tests can support that boundary without closing it.

## 4. Design the human and AI interfaces together

The human-facing interface should expose decisions about the computation,
accuracy, deployment, and optimization objective. It need not expose instruction
addresses or proof tactics unless the user is working at that level.

The AI-facing interface should expose the structures useful for a specific
change. A Gemmini reuse edit needs tile indices, B residency, partial-sum
lifetimes, and capacities. An asynchronous attention edit needs ownership,
pending transfers, synchronization, masking, and numerical obligations. Neither
interface needs to imitate the other.

Provide a formal contract reference alongside any readable summary. Show allowed
edits, relevant semantics and lemmas, the candidate's current claims, and
actionable failure details. A rejection should identify the failed obligation,
the affected artifact location, and available evidence; include a counterexample
only when one was actually obtained. Distinguish unsupported representation,
false claim, missing proof, and checker timeout. A timeout is not a disproof.

The AI may propose a better representation or a new specialization. Such changes
need versioned interpretation and checked bridges before existing proofs can
be reused. A prompt explains the contract but cannot authorize acceptance.

## 5. Two concrete specializations

| Decision | Gemmini exact GEMM, current foundation | Stateful GPU attention, proposed example |
|---|---|---|
| Computation input | Dimensions and the existing Lean GEMM definition | Attention operation graph, masks, position rules, KV-cache state and updates |
| Numerical contract | Signed int8 inputs, int32 output, explicit overflow bound | Storage/product/accumulator formats, reduction and transcendental semantics, declared error relation |
| Execution view | Full-tile loads, preload, compute, accumulator storage, fence | Thread groups, fragments, shared/global memory, asynchronous copies and barriers |
| Useful optimizer edit | Batch output rows to reuse B within accumulator capacity | Change staging, fusion, layout, or partitioning with ownership and numerical obligations |
| Proof route | Existing symbolic validation and concrete byte certificates | Target-specific refinements and numerical proofs still required; external compilation is an explicit boundary |

For Gemmini, the immediate specialization should wrap the existing proof path.
Split its currently bundled plan into computation, target, and candidate choices
at the interface level, with an adapter to the existing request format. The
schedule tag is a candidate choice, not the definition of GEMM or the ISA.
Make fixed numerical and runtime assumptions visible in the contract rather
than leaving users to infer them from the checker implementation.

For attention, begin by specifying the distinct obligations, without claiming
the existing real-number and launch-legality proofs establish executable
floating-point correctness. Its different needs are a design test for the
meta-frontend before committing to a shared implementation framework.

## 6. Specialization process and exit conditions

Use the [kickoff prompt](frontend_specialization_prompt.md) to produce a design
dossier: decisions on the common concerns, precise native representations,
an obligation/assumption map, one valid example, rejection examples, a proof
path to the requested delivery boundary, and the first implementation slice.
The dossier is reviewable design input, not an automatically trusted backend.

First implement a Gemmini adapter without weakening its current acceptance
path. Then specify a contrasting attention interface and revise the common
structure if it forces target-specific facts into generic fields. Extract
shared code only where these two designs demonstrate a stable common need.

Evaluate the frontend by whether it makes the intended computation and trust
boundary clear, preserves meaning through import and specialization, supports
useful AI edits and repairs, and produces replayable evidence. Track time to
bring up a new specialization and proof effort as well as optimization quality.
The number of models fitting a universal schema is not an acceptance metric.

Reject silent domain narrowing, changed target assumptions, stale certificates,
and lost numerical obligations. Demonstrate that unsupported semantics and
proof timeouts remain explicit, and that the final acceptance refers to the
actual delivered artifact. Approximation and refinement claims must compose
under their hypotheses; error budgets cannot simply be added without a theorem.
