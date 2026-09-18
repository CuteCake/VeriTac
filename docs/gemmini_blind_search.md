# Live blind Gemmini search

`examples/gemmini_blind_search.py` runs fresh OpenCode proposals against a fixed
GEMM task and the existing Lean byte validator. Implementation workers and
experimental proposers have different roles: implementation workers may read
the repository; experimental proposers receive only instruction semantics, the
task, and their own earlier proposals and validation feedback.

The initial registered experiment is in
[`benchmarks/gemmini/blind_search/2026-09-17`](../benchmarks/gemmini/blind_search/2026-09-17/PROTOCOL.md).
Its tasks intentionally exercise different capacities and aspect ratios. They
are held out from the experiment's supplied implementation examples, not claimed
to be absent from model training data.

## Run one task

```bash
lake build veritac gemmini_check gemmini_program_check VeriTac
python3 examples/gemmini_blind_search.py run \
  --task benchmarks/gemmini/blind_search/2026-09-17/tasks/rows_tight.json \
  --model dgxspark-glm/glm-5.3-flash --rounds 4 \
  --model-timeout 900 --proof-timeout 600 --max-commands 4096
```

Without `--out`, this creates a unique directory under `.lake/blind_search/`.
An explicit output directory must not already exist. `prepare` creates an
inspection snapshot without making model calls; `run` creates its own fresh
experiment. `report --experiment /absolute/path/to/run` displays the outcome
and checks its recorded source fingerprints against current files.

A task is exactly `{ "id": "identifier", "plan": { ... } }`, with the existing
Gemmini plan fields: `m`, `n`, `k`, `dim`, `scratchpad_rows`, `accumulator_rows`,
and `schedule`. Use the `baseline` seed label to avoid imposing the existing
full-B-reuse template's plan requirement. This label does not generate the
candidate: the model submits the concrete commands that are checked.

The configured OpenCode model must already be available. Every experimental
call starts a fresh session in a temporary directory outside the repository,
with tools disabled, permissions denied, plugins disabled, and sharing disabled.
The controller checks the resolved configuration and rejects any tool event.
This is a tool-execution boundary, not an OS-level hermetic sandbox: OpenCode
still reads its global provider configuration. The report records the limitation
and the observed isolation probe; provider credentials are not archived.

## Acceptance and feedback

The proposer supplies a single JSON object with `commands` and `rationale`.
Duplicate keys, extra contract fields, ambiguous multiple objects, non-finite
numbers, and Boolean values substituted for integer fields are rejected.
The optimizer cannot change the task to make a program pass.

The objective is modeled input DMA bytes, then command count. Each candidate is
encoded and checked by the native Lean byte checker. A native acceptance is a
search filter; the selected winner must separately pass Lean kernel checking
of its actual emitted bytes. Proof failures remain failures, even after native
acceptance. Candidate identity and metrics are checked again before certification.

Feedback includes the model's actual previous commands, status, metrics, and
checker reason. Advisory diagnostics can locate a resource or initialization
failure or identify a symbolic output mismatch. They never accept programs,
invent numerical counterexamples, or suggest a specific repair. Diagnostics
track input provenance and return `inconclusive` when their work budget is
exceeded. The byte checker remains authoritative even when advisory execution
finds no failure.

The controller distinguishes malformed proposals, semantic rejection, model
timeouts, checker timeouts, process failures, isolation violations, and proof
failure. It preserves prompts, raw response events, proposals, receipts, token
usage when available, timings, and certificates. Synthetic orchestration tests
are explicitly marked `synthetic_test`; they are not live optimization results.
Reported provider cost zero must not be interpreted as zero compute cost.

The task, formal sources, built checker, built Lean library, toolchain metadata,
and acceptance-related controller modules are bound for each run. Drift aborts
the experiment. Later source changes may make a historical report's *current
environment* integrity check fail; replaying its proof requires the recorded
dependencies. A hash alone is not a proof.

## Controls and execution evidence

`specializations/gemmini_gemm/evaluation.py` compares a finished proposal against
the baseline and all feasible existing batched-B settings under the same plan.
These controls are never exposed in proposal feedback. A different command hash
does not establish a new optimization; inspect actual residency, traversal, and
A/B transfer counts before making a novelty claim.

The experiment's `check_local_execution.py RUN_DIRECTORY` checks actual RV64
operand packets with an independent interpreter and complete GEMM outputs on
six deterministic input cases. It is local regression evidence, not upstream
Spike or hardware conformance. The separately prepared `replay_spike.py` sends
only the certified body, request, and runner to the existing private Spark
toolchain when that transfer/execution is authorized. It retains large logs
remotely and retrieves the small report. Neither path measures silicon latency.

## Compact schedule adapter

`specializations/gemmini_gemm/schedule_ir.py` provides a separate bounded
expansion API for larger schedules. The initial registered flat-JSON experiment
does **not** use this interface, so its results do not establish a benefit from
compact syntax.

Use `--proposal-format compact` on the live-search command to select this
interface. A [separately registered second phase](../benchmarks/gemmini/blind_search/2026-09-17/compact_preregistration.json)
uses the same six tasks, model, round/time budgets, and formal acceptance.
Its records live in `compact_runs/`, separate from the flat phase's `runs/`.
The comparison covers the compact representation and its source-preserving
feedback; it is an exploratory paired run, not a statistical causal estimate.

Loop nodes contain exactly `for`, `start`, `stop`, `step`, and `body`; command
fields can use `{ "expr": "i * 16" }`. Constants and loop indices are immutable
within scope. The interpreter supports bounded integer arithmetic, comparisons,
and conditional expressions. It does not execute Python, call functions, read
files, or permit attribute access. Source size, nesting, iterations, integer
magnitude, and expanded command count are bounded.

Expansion is untrusted. The resulting flat commands still need strict command
parsing, native byte validation, and a final kernel certificate. A compact
schedule that expands to the wrong indexing is rejected by the same existing
checker. No compiler-correctness theorem is claimed for this adapter.

## Reusing a discovered schedule

After the live `columns_tight` round-2 candidate obtained its byte certificate,
the controller generalized its A-reuse pattern in
`specializations/gemmini_gemm/learned_strategies.py`. `gen_batched_reuse_a`
handles multiple output-row tiles and a partial last column batch;
`certify_batched_reuse_a` returns acceptance only through the existing concrete
byte certificate path. Oversized batches are rejected rather than silently
changing the task's resources.

This is a post-discovery controller generalization, not a template supplied to
the experimental proposer. It is excluded from the registered old-generator
controls. A reuse is not always better than B reuse: choose among checked
programs for the actual workload. The generalized Python generator itself is
untrusted and has no total-correctness theorem; each accepted concrete result
has the existing validator's soundness guarantee and its own certificate.

The certified `columns_cache` round-1 program later supplied a second pattern:
`gen_resident_a` keeps an output-row block's A tiles in scratchpad across all
column tiles, using `K+16` scratchpad rows and one accumulator tile.
`certify_resident_a` provides the same concrete acceptance path. The resource
estimate is descriptive; the checker still examines every actual access.

These strategies can be reused without another model call:

```bash
python3 -m specializations.gemmini_gemm.learned_strategies \
  --task benchmarks/gemmini/blind_search/2026-09-17/tasks/columns_tight.json \
  --strategy batched-a --batch-columns 2 --out .lake/learned-a-new
python3 -m specializations.gemmini_gemm.learned_strategies \
  --task benchmarks/gemmini/blind_search/2026-09-17/tasks/columns_cache.json \
  --strategy resident-a --out .lake/resident-a-new
```

Both output directories must be new. The commands perform fresh byte
certification; a successful earlier discovery does not bypass validation.

Two further certified discoveries were generalized after the flat phase:

- `resident-a-batched` keeps the first row block's A resident and alternates a
  streaming slot between B and the other row blocks' A, using latched B weights.
- `serpentine-b` caches B by column, reverses row traversal at column boundaries,
  and rotates dead storage to preserve the last row's A. Its implementation was
  derived from the winning command stream; that model's own traffic/count
  explanation was inaccurate and was not used as the cost authority.

`--strategy auto` (the offline CLI default) ranks the baseline, old batched-B,
and discovered recipes by actual input traffic and command count, then validates
in that order and kernel-certifies the selected concrete body. It records the
selection in `selection.json`; untested recipes have a null native verdict.
This reuses discoveries without model calls and does not claim a globally
optimal schedule or replace open-ended model search.


## Proving a rejected candidate is numerically wrong

`rejection_witness.find_sparse_counterexample(plan, commands)` optionally
searches for a concrete 0/1 input after an output mismatch. It normalizes
quadratic product coefficients before trying sparse assignments, then executes
the command model to verify the reported differing cell. This is separate from
acceptance and was not supplied as feedback in either registered phase.

`prove_sparse_counterexample(plan, commands, new_output_directory)` emits the
rejected bytes and a Lean theorem evaluating those bytes on the concrete input.
The kernel checks that execution drains, writes the selected in-bounds cell,
and produces the reported value different from GEMM. Input-domain theorems and
an axiom whitelist are checked too. `kernel_proved: true` certifies a
counterexample, never an accepted kernel. Failed proof attempts retain their
logs and remain false.

This is a bounded witness search, not a complete decision procedure. The current
acceptance checker is conservative: a change in reduction order can be correct
for these exact integers yet fail its ordered symbolic comparison. Such a
rejection must not be described as a numerical bug without a counterexample.
