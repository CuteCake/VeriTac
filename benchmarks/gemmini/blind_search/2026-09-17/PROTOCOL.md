# Blind proposal experiment, 2026-09-17

This experiment asks whether a live model can produce useful Gemmini schedules
without receiving an optimized reference kernel or a prescribed reuse strategy.
The six task contracts were recorded in `preregistration.json` before proposals.
They deliberately exercise different matrix aspect ratios and storage limits;
they are not a random sample of workloads. “Held out” refers to this experiment's
prompt exposure, not to an unverifiable claim about model training data.

## Fixed acceptance boundary

The mathematical task is exact row-major int8 GEMM with int32 outputs, full
16-element tiles, and the existing overflow bound. The existing sequential
Gemmini semantics and raw-byte checker remain unchanged. Each task fixes shape,
scratchpad capacity, accumulator capacity, and numerical domain. The schedule
tag is a compatibility seed label, not permission to change the computation.

The proposer receives the task, complete instruction semantics/schema, the
optimization objective, and its own previous proposals and feedback. It receives
no baseline commands, optimized templates, prior repair transcripts, or control
scores. A fresh, tool-disabled OpenCode session is used outside the repository.
Any inability to establish that isolation is reported as a limitation, rather
than silently calling the result blind.

Four rounds per task are allowed. Native Lean validation supplies fast feedback;
it does not establish final kernel certification. The selected candidate must
have a successful Lean kernel proof of its actual emitted bytes. All attempts,
including malformed output, timeouts and rejected programs, remain in the
record. The controller does not write repairs to model proposals. Implementation
workers are distinct from isolated experimental proposers.

## Objective and controls

Rank admissible candidates by total modeled input DMA bytes, then command count.
This is a traffic objective, not a hardware latency measurement. Score the final
candidate against the existing baseline and every feasible existing batched-B
generator setting, under the same task contract. These controls are computed
after proposals and are never included in model feedback.

A changed command hash or instruction order is not evidence of a new mechanism.
Review actual A/B load counts, residency lifetimes, traversal, and scratchpad
use. Report rediscovery of an existing strategy as rediscovery. Report any
improvement over existing generators separately from improvement over baseline.

## Independent execution and reporting

Where certificates succeed, execute those exact raw bodies on the existing
upstream Gemmini/Spike toolchain. Check byte identity, committed opcodes, all
outputs, and command/traffic counts on zeros, alternating extrema, and
deterministic pseudorandom inputs. A simulator with greater physical capacity
does not establish performance of the smaller declared target; the formal
checker establishes the program's declared resource bound.

Record model identifier, raw response events, available token usage, wall time,
round outcomes, proof-check time, task/source hashes, control comparisons, and
execution reports. Missing token accounting is unknown, not zero. Both failed
tasks and infrastructure failures remain in aggregate denominators. Report
hardware latency as unmeasured. The sequential-model-to-asynchronous-hardware
correspondence remains a separate boundary.
