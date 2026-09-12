# Apple Neural Engine feasibility

Follow-up: the [M3 loader investigation](ane_loader_investigation.md) exported
an E5 descriptor and cache reference, and demonstrated fresh-process reuse
with exact numerical output. It still recovered no HWX executable and found
no demonstrated route for loading independently generated M3 code.

This spike asks whether the local ANE can become an execution target for
VeriTac's proof-carrying compiler. The first workload is a static FP16
linear layer (1×1 convolution) followed by ReLU. Runtime execution and
container inspection are evidence-gathering steps; neither establishes
instruction-level correctness.

## Local observations

The host is an Apple M3 Ultra Mac Studio (`Mac15,14`) with 256 GiB of memory,
running macOS 26.6.2 (25G83). The installed ANECompiler framework reports
9.509.0. IORegistry exposes two ANE nodes, each reporting 16 cores and
architecture `h15g`, CPU subtype 6, board type 224, ANE version 176 and minor
version 16. The driver class name `H11ANEIn` does not identify the silicon
generation.

The private in-memory model descriptor, compiler, loader and evaluation
selectors are present, and both initial workloads compiled, loaded and
evaluated successfully through this route. Power-rail sampling is unavailable in
this session because `sudo -n powermetrics` requires administrator
authentication. Runtime and artifact evidence must be reported separately
from independent hardware power measurements.

## Bounded experiment

Use one input and output of shape `[1,512,1,64]`, stored as contiguous FP16
values with offset `channel * 64 + spatial`. Start with identity weights,
then test `W[o,o] = 0.5` and `W[o,(o+1) mod 512] = -0.25` on mixed-sign,
dyadic inputs. Compare every output against an independent float64 reference
with explicit tolerances and finite-value checks. The second case detects
weight-orientation and ReLU mistakes hidden by an identity-only test.

Compile attempts, including failures, are limited to eight for this spike
and run serially. Record compilation, loading, input/output transfers and
dispatch separately. Retain exact MIL, weight blobs, tensor data, source and
binary hashes, runtime metadata, failures and compiler-produced artifacts.
Preserve the framework's exact model-identifier directory layout. The
isolated `TMPDIR` policy passed its path check in the sandbox, but the compiler
returned failure there. The successful normal launch used `system-model`:
exclusively create the model's directory under Foundation's actual temporary
root, refuse an existing directory, and copy only that directory into the run
record before releasing the model. Never inventory the temporary parent.

The runtime reference is [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29),
pinned at `d91c9845c0784dec7753048954fc6d0e8411fe29` (MIT). Its
[M3 Ultra report](https://github.com/maderix/ANE/issues/42) motivates the
initial 512-channel case; it is not a universal shape constraint.

## Proof boundary and decision criteria

The private MIL route still uses Apple's compiler. This is compatible with
an untrusted producer only if VeriTac can independently validate the
resulting executable against its own semantics. A proof about MIL or a
matrix identity alone does not certify the compiler-produced HWX binary.

The artifact inspector must distinguish container structure from task,
register, memory and arithmetic semantics. Unknown commands remain raw
offset/type/length/hash records. The
[ANE program-format research](https://github.com/sbryngelson/ane-guide/blob/main/part-7-toolchain/23-program-format.md)
is a starting point, not authority to apply M1 descriptor offsets to M3.
Reported signing constraints also need to be distinguished from locally
tested behavior. No security changes or loader patches are part of this
experiment.

Proceed to a narrow Lean instruction model only when an actual emitted
program supplies enough substantiated semantics to describe its reads,
writes, arithmetic and execution order. Otherwise retain ANE as an
experimental execution backend, identify the missing descriptor semantics,
and continue the vendor-independent demonstration on the minimal modeled
ISA. Do not fill this gap with an invented ANE instruction set.

## Execution evidence — 11 September 2026

Both `[1,512,1,64]` cases passed all 32,768 output comparisons with
`atol=rtol=0`. The controller independently decoded the saved FP16 weight
blob and input, performed a dense float64 matrix multiplication, applied
ReLU, and compared the complete output. Identity produced 16,278 nonzero
outputs; two-tap produced 16,335. Every expected result is exactly
representable in FP16 for the chosen dyadic inputs.

| Weight pattern | Setup + compile | Load | Median dispatch | Mean dispatch |
|---|---:|---:|---:|---:|
| Identity | 118.061 ms | 46.809 ms | 152.438 µs | 195.288 µs |
| Two-tap | 22.813 ms | 45.892 ms | 131.958 µs | 137.733 µs |

Each case used two warmups and ten timed synchronous evaluations. Dispatch
includes the private API's host/IPC overhead; it is not an isolated hardware
kernel time. Setup + compile includes descriptor construction, staging files
and IOSurface setup. These are feasibility measurements, not a vendor
comparison or evidence that two-tap is intrinsically faster than identity.

See [independent verification and hashes](../benchmarks/ane/results/run-20260911-4/controller_verification.json),
[raw run index](../benchmarks/ane/results/run-20260911-4/index.json), and
[reproduction instructions](../benchmarks/ane/README.md).

Seven bridge attempts were made, including setup diagnostics. Four reached
`compileWithQoS`: one sandboxed failure, two successful evaluated models,
and one successful compile-only metadata probe. The eight-attempt bound was
not exhausted. Earlier failed attempts and matching source snapshots remain
in `benchmarks/ane/results/run-20260911-{1,2,3}`. The first index has a known
serialization bug; its per-case `results.json` and `merged.json` are the
usable records.

## Artifact findings and decision

The two successful runs preserved four files each: `model.mil`, `net.plist`,
`data`, and `weights/weight.bin`. Despite its filename, `net.plist` contains
the same MIL bytes; these files are staging inputs. **No compiled HWX binary
was recovered.** The [artifact report](../benchmarks/ane/results/run-20260911-4/artifact_inspection.json)
classifies all eight files as unknown formats for its Mach-O/HWX/metadata
parser. A successful inspection process is not a successful executable decode.

The compile-only [runtime metadata probe](../benchmarks/ane/results/run-20260911-4/runtime-metadata/compiledpaths.txt)
returned a model URL pointing back to staging, no source URL or cache URL
identifier, and vendor-supplied tensor attributes. Those attributes report
Float16, width 64, channels 512, row/plane stride 128 bytes and batch stride
65,536 bytes, consistent with the tested layout. They are observations from
the compiler, not independently trusted instruction semantics. The diagnostic
staging directory was not retained after the helper exited.

The generic inspector passes 29 host tests covering malformed containers,
endianness, read bounds, zero-fill sections, symlink escapes and metadata.
Its synthetic HWX tests do not establish M3 instruction decoding.

**Decision: usable experimental execution backend; not yet a
vendor-independent proof target.** The next ANE milestone is a supported,
reproducible export of the exact compiled program, followed by an M3-specific
model of a minimal compute and memory descriptor subset. Until then, no Lean
theorem about this experiment can bridge MIL to the physical ANE executable.
We deliberately added no pretend instruction model or disconnected transform
proof. The minimal modeled ISA remains the primary route for demonstrating
the complete vendor-independent proof chain; ANE can provide a useful
execution and artifact-recovery research track alongside it.
