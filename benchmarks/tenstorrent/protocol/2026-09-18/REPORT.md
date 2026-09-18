# Tenstorrent protocol milestone: local simulator, proof, and AI optimization

The third milestone is complete for a restricted, genuine asynchronous
Metalium protocol: **four live AI-generated programs received Lean protocol
certificates and ran successfully on the official Blackhole simulator installed
on the local Mac Studio**. Four serial controls ran through the same execution
checks. Across eight programs and six input cases each, all **48 full-output
byte comparisons passed**.

This is a protocol-IR result. It is **not** a proof of the emitted C++ or ELF,
not a floating-point/GEMM result, and not a hardware speedup measurement.

## Fixed scope and experiment

- Blackhole; one Tensix core; one data-movement RISC-V issuer and one NoC.
- Disjoint immutable input DRAM, output DRAM, and one-page circular-buffer slots.
- Exact page permutation/replication. Each page in this pilot is 512 bytes.
- Reserve, asynchronous read, read barrier, publish, acquire, asynchronous write,
  write barrier, release. Every pending transfer can complete in any order.
- Objective: explicit read/write barrier count, then instruction count, then
  peak live slots. Transfer bytes are reported separately.
- Four tasks, three fresh OpenCode calls per task, at most three parallel calls,
  900-second generation limit, 600-second final proof limit.
- Requested model: `dgxspark-glm/glm-5.3-flash` on the user-authorized private gx10
  endpoint. The model's weights were not independently attested.
- Exposure: fixed task and operation semantics, objective, own earlier proposals
  and factual native-checker feedback. No optimized implementation or control
  recipe was supplied; tool execution was disabled in fresh temporary working
  directories. This is not an OS-hermetic isolation claim.

The [scope](SCOPE.md), [registration](preregistration.json), and
[source snapshot](source_snapshot/) were fixed before the model trials.
Registration was at 2026-09-18 10:14:27 UTC; the last task finished at 10:20:54 UTC.
The formal implementation, compiled protocol library and search-controller
fingerprints remained unchanged throughout the registered experiment.

## Results

Every task selected its first-round proposal. All 12 rounds passed the native
filter; four selected programs subsequently passed independent Lean kernel
checks. There were no model rejections or timeouts in this small pilot, so it
does not demonstrate live self-repair. Rejection behavior was tested separately
with deliberate bad programs and certificate mutations.

| Task | Output source pages | Slots | Serial barriers | AI barriers | Serial input bytes | AI input bytes | AI instructions |
|---|---|---:|---:|---:|---:|---:|---:|
| pair | 0, 1 | 2 | 4 | 2 | 1024 | 1024 | 14 |
| permutation | 2, 0, 3, 1 | 2 | 8 | 4 | 2048 | 2048 | 28 |
| replication | 0, 0, 0, 0 | 1 | 8 | 2 | 2048 | 512 | 11 |
| reuse_permutation | 1, 0, 1, 0 | 2 | 8 | 2 | 2048 | 1024 | 16 |

The model independently submitted barrier batching and source-page reuse,
reducing explicit barriers by **50–75% against the serial control**. The two
reuse tasks also reduce input traffic. **Every winner matches, rather than
beats, the preregistered controller-written batched/reuse control.** Those
controls were created for development/preflight and withheld from the proposer.
This is a successful optimization-and-verification pilot, not evidence of novel
optimization beyond existing recipes or global optimality. The tasks are small
and selected deliberately; no broad model-success-rate claim follows.

See [all task outcomes](suite_outcomes.json), [fixed controls](controls/),
[model runs](runs/), and the [binding audit](audit.json).

## What Lean proves

The untrusted producer supplies a finite set of states. The Lean checker
independently checks initial-state membership, each local safety condition,
every modeled successor's membership, and correct terminal states. A general
induction theorem then covers **all reachable event interleavings**, not merely
a trace sampled by Python or the simulator.

Each accepted graph also checks that the rank
`2 * remaining_instructions + pending_transfers` strictly decreases on every
edge. The generic specialization proves event-path well-foundedness. Idle time
is not an event: physical completion still assumes eventual DMA completion and
issuer scheduling. This is not multi-thread/multi-core deadlock verification.

The output theorem interprets each proven source origin as an arbitrary input
page payload under the declared immutable-input, exact-copy semantics. It does
not rely on distinct test values or a reference optimized kernel. It does not
prove that actual Metalium primitives, partial physical DMA writes, firmware or
compiler lowering refine those abstract semantics.

The four selected model certificates took **3.713–4.110 seconds** each on the
local Lean host with built dependencies. Their audited axioms are `propext` and
`Quot.sound`; no `sorryAx` or native-evaluation axiom is admitted. Four batched
preflight programs and four serial controls also received protocol certificates,
recorded separately from the four AI results.

The first development proof attempt under `preflight/pair/` failed and is
preserved as `accepted: false`. The successful development certificates are in
[preflight_v2/](preflight_v2/). Failed attempts are not counted as proof evidence.

## Local official-simulator execution

The simulator was installed on the **Mac Studio itself**, in Linux ARM64 Docker
container `veritac-tt-sim`, persistent volume `veritac-tt-sim-data`. Spark3 was not
used. The official RISC-V integer example first passed with result 21; then the
generated single-DM-kernel Metalium programs were compiled and executed.

The frozen protocol is encoded as a runtime instruction stream consumed by a
small fixed data-movement kernel. The host program initializes disjoint DRAM
buffers and CBs; the device uses actual Metalium NoC and CB operations. This is
not execution of the Python protocol model. The emitted source, instruction
stream and original certified proposal are tied together by the artifact audit.

Each program was checked on a default LCG input plus zero, all-ones,
alternating-bit, LCG-seed-0 and LCG-seed-917 inputs. The C++ host compares all
bytes and saves `output.bin`; Python independently compares those saved bytes
with its reference. Checksums are additional provenance, not the correctness
oracle. Successful output, logs and generated sources are under
[runtime/pilot-v1/](runtime/pilot-v1/), summarized in
[pilot-v1_runtime_summary.json](pilot-v1_runtime_summary.json).

The actual JIT RISC-V device ELF files, source association and build logs are
retained under [environment/device_code/](environment/device_code/).
The formal certificate does **not** extend to those ELF files merely because
hashes bind them to an execution.

Environment:

- Metalium base commit `ad232e1cd799adf53841774805bc73fc5605b634`.
- Official Blackhole ARM64 `ttsim v1.10.8`; installed binary SHA-256 verified
  against the release digest:
  `0c3a5af653e7feec018f6d1ce3bba116967a951c150fcaa03ad6521e8ec88065`.
- Ubuntu 24.04 ARM64 base image pinned by digest; native `g++-14` host build.
- Two recorded local compatibility patches, touching three host-side files:
  AVX2-profiler include portability via existing SIMDe, and suppression of
  OpenMPI legacy C++ bindings when linking the C MPI interface.
- MPI enabled: this pinned revision's MPI-disabled single-host startup called
  an unsupported `all_reduce`. Firmware uses the normal JIT path.
- Streaming-profiler functionality was not validated; no DMA, arithmetic or
  protocol-checking logic was patched.

[Installed artifact hashes](environment/installed.json),
[exact source diff](environment/installed_metal.patch), and
[package/submodule versions](environment/packages_and_submodules.txt) are saved.
The reusable setup script composes the commands exercised during this install;
it has not been tested by a second full clean installation. Apt mirrors are
not pinned snapshots. This is a reproducible procedure with recorded versions,
not a claim of bit-identical hermetic rebuilding.

## Validation and costs

An independent worker implemented a differently represented reference model
(DFS and issue-number-tagged pending sets). Replaying **3,000 random and 4,000
structured trials** found zero status disagreements; 1,444 accepted structured
programs also matched the canonical replay and adapter accounting. These are
finite tests, not the formal soundness argument.

The [review notes](review/controller_notes.md) preserve limitations and correct
an overstatement in the worker's original review: random mutation sampling did
not activate a weakened destination-conflict guard. A directed test does detect
it (original/reference `unsafe`, weakened mutant `accepted`). The redundant
publish guard mutant is separately labeled. See [review evidence](review/).

Lean integration tests also kernel-check rejection of certificates missing the
initial node, omitting a successor, using an unsafe program, or changing the
output specification. Runtime development fixes were tested against real C++
compilation and JIT execution, in addition to unit tests.

The live proposal phase reports **53,941 input + 19,182 output = 73,123 tokens**
across 12 calls, all with usage. Summed model-call wall time is **1,019.221
seconds**, not elapsed time or accelerator compute time. Provider cost zero is
not a claim of zero resource consumption. These numbers exclude implementation
and review worker calls. Implementation/review concurrency and proposal
concurrency never exceeded three OpenCode workers/calls at once.

The final repository regression ran **339 tests successfully, with one environment-dependent skip**. See the [validation manifest](validation.json) and [test log](regression_tests.log). The earlier Gemmini acceptance-source/binary fingerprints remain unchanged. No commit, push, or external message was made.

## Next boundary

This completes the initial protocol-optimization milestone. The next useful
extension is separate reader/writer issuers with genuine producer/consumer
blocking, followed by a sound correspondence between the checked protocol and
emitted implementation. Floating-point compute and actual-card performance
need their own contracts and evidence. The present demo can support a concrete
hardware discussion, but should be presented with these boundaries intact.

See [usage and installation instructions](../../../../docs/tenstorrent_protocol.md).
