# Gemmini GEMM optimization

This backend targets signed int8 GEMM with int32 output on Gemmini's
weight-stationary dataflow. The first optimization reorders tile execution to
reuse each B tile across multiple output-row tiles. Its extra live output tiles
require more accumulator storage; the Lean plan checker must reject the
optimization when that storage is unavailable.

The supported initial domain is positive M, N, K divisible by the array width.
The conservative arithmetic contract is `K * 16384 <= 2147483647` for inputs
in `[-128, 127]`, with no bias, activation, or output quantization. Exact
full-width accumulator readback is essential. The prototype's modeled target
capacity is an explicit input, not an automatically verified hardware fact.

The current emitter supports the pinned `DIM=16` int8/int32 configuration and
capacity restrictions within its limits. Other array widths require an emitter
extension even though the Lean mathematical checker is parameterized by width.
Buffer element counts must also fit the emitter's signed-int indexing range.

## Checked executable subset

The repository has a shape-parameterized acceptance path for full-tile
GEMM on the fixed `DIM=16` int8/int32 configuration. Acceptance is no longer a
four-template whitelist. `Symbolic.check` executes the concrete command program
over symbolic input references and ordered product lists, then checks every
output against GEMM. `Symbolic.check_sound` relates that execution to the numeric
instruction semantics for all bounded int8 inputs.

The shared instruction engine tracks addressing, strides, initialized storage,
retained B weights, int32 partial sums, stores, and fences. The independently
proved `K32Baseline` and `K32Reuse` programs cover 32×16×32, including accumulation
of nonzero partial sums across two K tiles.

`Lowering.baselineProgram` and `batchedReuseProgram` propose programs for
parameterized shapes. `lowerBaseline`, `lowerBReuse`, and `reuseOperand` return
checked programs or failure. `checkedLowering_sound` and `checkedRewrite_sound`
prove accepted results correct; they do not claim every legal proposal succeeds.
The ordered-product checker is conservative and can reject otherwise equivalent
reduction reorderings.

`checkExecutable_sound` in [Executable.lean](../VeriTac/Gemmini/Executable.lean)
connects the submitted bytes to the row-major GEMM specification for **every
int8 input value** in the declared buffers. Its conclusion includes successful
execution, complete output coverage, and fenced completion. The chain is:

```
submitted RV64 bytes
  → operational register execution and Gemmini operand trace
  → decoded loads/preloads/computes/stores
  → modeled scratchpad, retained weights, accumulator, and output state
  → exact integer GEMM
```

The untrusted Python emitter constructs ordinary RV64 instructions to materialize
operands in x5/x6, then emits CUSTOM_3 words, a fence, and a return. It does not
use a C compiler or assembler to construct this kernel body. The checker rejects
unknown instructions, uninitialized register reads, changed operands, missing
fences, truncated/trailing bytes, overlapping buffers, and unsupported programs.
The packet decoder reconstructs commands from the bytes; execution is not defined
using the request's claimed program or the GEMM specification.

The [generalized proof artifacts](../benchmarks/gemmini/results/generalized_2026-09-16.json)
include 32×16×32 baseline and reuse, 16×32×48 reuse, 32³ reuse, and 64³ reuse.
Each includes the actual raw `.bin`, command list, and generated Lean certificate.
Separate kernel reductions check symbolic acceptance, byte execution, and
command decoding; the final theorem composes these links. Reductions use
`decide` (new certificates use `decide +kernel`), with no `native_decide`, admitted proofs, or new axioms.
The axiom audit reports only `propext`, `Classical.choice`, and `Quot.sound`.
Manifests bind the bytes, proof source, dependencies, and toolchain.

Concrete certificate checking took about 10–17 seconds for the smaller examples
and 135 seconds for 64³ on the development host. These are proof-checking times,
not accelerator latency. Replay uses `lake env lean -s 65536 <certificate.lean>`;
the larger stack is needed for kernel reduction.

This is a restricted kernel body, not a verified ELF loader or production model
executable. The caller must map initialized A/B and writable C at the artifact's
fixed, nonoverlapping addresses, keep the code immutable and separate from those
buffers, provide a valid return address, enable the accelerator, synchronize the
instruction cache, and provide exclusive access during execution. Only x5, x6,
and x31 are clobbered. The model assumes full-tile aligned storage, exact identity
scales, and sequential completion of commands; correspondence to asynchronous
hardware and physical silicon remains a separate conformance boundary. The
existing C emitter is still an unverified compatibility adapter.

## Raw-byte Spike validation and actual AI repair (2026-09-16)

Eight exact certified bodies ran on upstream Gemmini/Spike: the original four
K=16 cases, both 32×16×32 schedules, 16×32×48 reuse, and a model-proposed
48×16×32 batched-reuse repair. Each ran zeros, alternating int8 extrema, and
deterministic random inputs, with every output compared to scalar GEMM.

The [raw runner](../benchmarks/gemmini/run_raw_spike.py) embeds the `.bin` with
`.incbin`; only host setup is compiled as C. It checks section placement,
ELF and runtime byte equality, every committed kernel PC/opcode, exact dynamic
command and DMA counts, and complete output files. Reports record toolchain,
configuration and log hashes. Large commit logs remain on the recorded remote
host. See the [combined evidence index](../benchmarks/gemmini/results/completion_2026-09-16.json).

On the Linux host with the pinned upstream toolchain installed:

```bash
python3 benchmarks/gemmini/run_raw_spike.py \
  --root /home/cake/.cache/veritac-gemmini \
  --request benchmarks/gemmini/checked_general/32x16x32_baseline/request.json \
  --binary benchmarks/gemmini/checked_general/32x16x32_baseline/baseline.bin \
  --output /tmp/veritac-raw-new-run
```

The output directory must be new. This runner assumes an already certified
request/binary pair; use the saved Lean certificate to replay the proof first.


The [recorded OpenCode interaction](../benchmarks/gemmini/ai_repair/2026-09-16/trace.json)
uses the same 48×16×32 workload, 32 scratchpad rows, and 32 accumulator rows
throughout. In a guided capacity challenge, the model proposed keeping three
output-row tiles live and explicitly noted the likely capacity risk. Lean
rejected that actual proposal. Given the rejection in the same session, the
model proposed two-row batches plus a one-row tail. Both endpoint byte proofs
and a separate rewrite-equivalence proof passed, then the repaired bytes passed
Spike. B loads fall from six to four per call (Spike observed twelve over three
calls). This is traffic reduction, not a hardware speedup claim.

Replay the authentic archived proposals and regenerate the accepted edge:

```bash
python3 examples/gemmini_ai_repair_demo.py
# Faster native decisions, without regenerating kernel certificates:
python3 examples/gemmini_ai_repair_demo.py --check-only
```

Replay does not make a fresh model call. The visible transcript, original
proposal, diagnostics, repair, exact bytes, and proof logs are archived together.

## Run the local demo

```bash
lake build gemmini_check gemmini_program_check VeriTac.Gemmini.Executable
python3 examples/gemmini_gemm_demo.py --m 32 --n 16 --k 16
# Same 32×16×16 workload, restricted to one output tile in the accumulator:
python3 examples/gemmini_gemm_demo.py --m 32 --n 16 --k 16 --accumulator-rows 16
PYTHONPATH=. python3 -m unittest tests.test_gemmini tests.test_gemmini_checker tests.test_gemmini_spike_trace tests.test_gemmini_program tests.test_gemmini_encoding tests.test_gemmini_machine_bytes tests.test_gemmini_acceptance_audit
lake env lean -s 65536 tests/lean/GemminiProgram.lean
lake env lean -s 65536 tests/lean/GemminiGeneral.lean
```

The first run selects `reuse_b`; the restricted run rejects that candidate and
selects `baseline`. Each run writes a unique directory under
`.lake/gemmini_demo/`, containing the candidate plans, Lean diagnostics,
command streams and hashes, exact interpreter checks, generated C, raw kernel
bytes, kernel-checked certificates, and a fresh winner replay. Both candidates use the same declared target capacities.
Selection minimizes modeled DMA bytes, then instruction count, then required
accumulator rows. The fallback in this enumerating demo is scripted. The separate AI repair
archive above records an actual model interaction.

OpenCode workers implemented the initial backend, Lean checker, and regression
tests. Controller review and independent upstream compilation/execution found
and repaired tile-offset, preload/compute, capacity-addressing, and C macro
issues. Correctness tests remain mandatory in addition to plan acceptance.

## Recorded C/Spike results (2026-09-15)

All 16 emitted kernels passed full-output scalar comparisons under upstream
Gemmini/Spike, and their dynamic command counts matched the generated schedule.
Coverage includes both schedules, the four shapes below, and 32-cubed cases
with all-zero, all-127, all-minus-128, and minus-128-times-127 inputs.
The generated program initializes output pages before execution to avoid
counting proxy-kernel page-fault retries as additional Gemmini commands.

| M × N × K | Baseline B loads | Reuse B loads | Baseline input bytes | Reuse input bytes | Input reduction |
|---|---:|---:|---:|---:|---:|
| 32 × 32 × 32 | 8 | 4 | 4096 | 3072 | 25% |
| 64 × 64 × 64 | 64 | 16 | 32768 | 20480 | 37.5% |
| 48 × 32 × 64 | 24 | 8 | 12288 | 8192 | 33.3% |
| 16 × 48 × 32 | 6 | 6 | 3072 | 3072 | 0% |

For 64 cubed, output traffic is unchanged at 16384 bytes, so total modeled
input-plus-output traffic falls from 49152 to 36864 bytes (25%). These are
transfer and command counts, **not measured hardware latency speedups**.

A second, rebuilt Gemmini functional model with 32 scratchpad rows and 16
accumulator rows also passed 64-cubed baseline execution after Lean rejected
the B-reuse candidate. This validates adaptation to a modified simulator
configuration, not a second physical chip. The exact winner from the final
user-facing demo was also compiled and executed successfully.

The committed [results and artifact hashes](../benchmarks/gemmini/results/spike_2026-09-15.json)
record the 16 suite cases, constrained-model run, and final demo winner.
Validation also includes 36 Python tests, `lake build veritac gemmini_check VeriTac`,
and the axiom audit described below.

## Execution on upstream Gemmini

`benchmarks/gemmini/setup_spike.sh` prepares a private toolchain directory on
Ubuntu 24.04. It downloads and extracts packages without changing system
packages, pins upstream source revisions, builds Spike and libgemmini, builds
the RISC-V proxy kernel, and runs an upstream transfer smoke test. It needs
existing native build tools, git, apt-get, dpkg-dev, and network access.

```bash
bash benchmarks/gemmini/setup_spike.sh "$HOME/.cache/veritac-gemmini"
```

The tested configuration is Linux/aarch64 on Spark3. The same package layout
may work on Ubuntu/amd64 but has not been validated here. The source commits
are pinned; downloaded Ubuntu package versions depend on the apt repository,
and their exact hashes are saved in `deps/packages.sha256`.

`benchmarks/gemmini/run_spike.py` compiles a generated C program using the
upstream Gemmini headers, runs it with the upstream Gemmini Spike extension,
and saves a report, artifact hashes, and logs. It requires a fresh output
directory for each run. The default success marker is a standalone
`VERITAC_GEMMINI_PASS` line after full-output reference checks.

```bash
python3 benchmarks/gemmini/run_spike.py \
  --root "$HOME/.cache/veritac-gemmini" \
  --source /absolute/path/to/generated.c \
  --output /absolute/path/to/new-run-directory
```

The runner checks that the compiler's Gemmini parameter header and the
simulator source parameter header agree. Executable and extension hashes bind
the run to specific artifacts; they do not prove those binaries were compiled
from the recorded source. Hardware conformance remains a separate boundary.

To reproduce the constrained model after the normal setup, use a new directory:

```bash
python3 benchmarks/gemmini/make_constrained_simulator.py \
  --source-root "$HOME/.cache/veritac-gemmini" \
  --output "$HOME/.cache/veritac-gemmini-small"
```

Run `run_spike.py` with that directory as `--root` and the emitted `baseline.c`
from a demo using `--accumulator-rows 16`. The helper copies and modifies the
simulator and compiler parameter headers identically, rebuilds the extension,
and shares the original toolchain read-only. Its default capacities are the
32/16-row configuration above. This helper was exercised on Spark3.

## What the evidence means

- `VeriTac/GemminiExact.lean` proves `tiledDot_eq_dot_of_dvd` (the recursively
  executed tile reductions equal the mathematical dot product) and
  `prefix_fits_int32` (every scalar prefix stays in signed int32 under the
  stated contract). Their axiom audit reports only `propext`,
  `Classical.choice`, and `Quot.sound`; there are no admitted proofs.
- `VeriTac/GemminiCorrectness.lean` connects those results to the actual plan
  checker in `checked_reduction_correct`: acceptance implies the blocked
  reduction equality and every prefix's int32 bounds for int8 inputs.
- The new `checkExecutable_sound` theorem covers the restricted raw kernel bodies
  above. The historical C/Spike results below remain execution evidence for the
  broader emitter; they do not certify upstream C macros, the C compiler, Spike,
  or physical hardware.
- The portable interpreter and full-output scalar comparisons are executable
  validation. Concrete raw-body certificates provide the separate formal evidence
  for supported programs; the historical C-produced binaries remain unproved.
- Spike execution tests actual generated RISC-V programs containing Gemmini
  custom instructions. Its trace supplies executed instruction counts and
  scratchpad input-transfer bytes for this int8 subset.
- Spike is a functional simulator. Its wall time and cycle CSR values must
  not be reported as Gemmini hardware latency or speedup. Cycle-accurate RTL
  simulation or silicon is needed for that claim.

## Upstream references

- [Gemmini architecture and ISA](https://github.com/ucb-bar/gemmini)
- [Gemmini C headers and tests](https://github.com/ucb-bar/gemmini-rocc-tests/tree/7c540b3adf1b86ad93d07f893abe3a73489b568e)
- [Gemmini Spike extension](https://github.com/ucb-bar/libgemmini/tree/5b1254a7dc72f757442280e7f28ef30542c78533)
- [Spike](https://github.com/riscv-software-src/riscv-isa-sim/tree/1e05ddac3a6c351bfc0aeed0cf3a68940e7200ab)
- [RISC-V proxy kernel](https://github.com/riscv-software-src/riscv-pk/tree/9c61d29846d8521d9487a57739330f9682d5b542)

To replay a saved concrete certificate directly:

```bash
lake build VeriTac.Gemmini.Executable
lake env lean benchmarks/gemmini/checked/32x16x16_reuse_b/reuse_b_certificate.lean
```

The direct-byte artifacts have kernel proofs and independent RV64 execution
regressions. They have not yet been run under upstream Spike; the recorded Spike
suite above exercised the C adapter's generated programs.
