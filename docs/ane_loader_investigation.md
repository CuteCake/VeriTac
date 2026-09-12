# M3 ANE compiler-independence investigation

The question is whether VeriTac can control an ANE program's implementation
and execute it without Apple's optimizing compiler choosing the final
instructions. Calling a private compiler API, reopening a serialized graph,
or setting a runtime object's program handle does not establish that capability.

This investigation follows the [numerical feasibility spike](ane_feasibility.md).
It does not change DwarfStar or add a Lean hardware model.

## Evidence and scope

The local machine is the M3 Ultra (`Mac15,14`) previously measured, running
macOS 26.6.2 with ANECompiler 9.509.0. OpenCode implemented the probes under
controller review, with no more than two concurrent worker requests.

References are pinned to:

- [ANEForge](https://github.com/sbryngelson/ANEForge/tree/67c1d4861c562b0168d60c7a5677840e2cad44ea),
  especially its recovered E5 C signatures and dispatch implementation.
- [ANE architecture guide](https://github.com/sbryngelson/ane-guide/tree/9eb1afe95cdc766d7bec4d1d02d4918ef6268ece),
  especially the software-stack and entitlement-boundary chapters.

The guide reports M1 program signature and trust checks that reject
self-produced executables. ANEForge separately documents limitations on
reusing compiled bundles across processes. These reports motivate the tests;
they are not measurements of this M3 host.

## Local runtime surface

The [runtime inventory](../benchmarks/ane_loader/results/inventory.json)
records class methods, instance methods, properties and daemon protocol
methods from the loaded frameworks. Both frameworks loaded successfully.

The inspected daemon protocols expose compile, load, cache queries, unload,
chaining and telemetry, plus private real-time-task and echo operations.
They expose no separately named signing operation. This observation is
limited to the inspected Objective-C surface; it is not proof that no
internal signing functionality exists.

Five allowlisted, read-only getters were queried after inspecting their
signatures. `precompiledModelChecksDisabled` returned false. The named VM
boot argument is `ane_vm_allowPrecompiledBinary`; the testing precompiled
path points to an H13 framework asset. Neither is a demonstrated production
M3 custom-program interface. No settings, boot arguments, entitlements or
code-signing identities were changed.

## E5 export and preparation experiment

The E5 compiler configuration exposes `cache_bundle_location`. The probe
sets this to a new, caller-owned directory, selects ANE device mask 4 and
the graph segmenter, and journals every call. It never scans private caches.
The staged input is the same FP16 `[1,512,1,64]` identity linear + ReLU used
in the earlier successful numerical run.

The experiment separates library creation/function enumeration from
precompiled-operation preparation. The latter may contact the ANE service;
neither is a numerical evaluation. A fresh-process reopen makes no explicit
compiler API calls, but that alone cannot rule out internal compilation.

## Measured results

All 23 E5 symbols needed by the initial compile/reopen probe resolved. Two
E5 compiler calls were made in this investigation. The execution controls
performed exactly one inference each; no timing sweep was run.

| Experiment | Observation | What it establishes |
|---|---|---|
| Compile and construct an operation | All calls returned 0; function `main` exposed | The E5 route works for the selected graph |
| Fresh-process bundle reopen and operation construction | All calls returned 0 | Metadata/operation reuse works on this host |
| Explicit `program_function_load_for_execution` diagnostic | Returned 2 | An ambiguous API failure, not evidence of a signature rejection |
| Compile, bind buffers, encode and execute in one process | All 32,768 outputs matched exactly | Positive numerical execution control |
| Fresh process: reopen the unmodified copied bundle, bind and execute | All 32,768 outputs matched exactly; no explicit compiler calls | Saved bundle reuse works through the tested runtime path |
| Fresh process: same copied bundle path, only `.anehash` changed to an invalid reference | `op_create` returned 13; no execution call reached | The valid external program reference is required by this path |

Both successful executions produced 16,278 nonzero outputs and matched the
controller's independent float64 ReLU reference. Output buffers were filled
with NaNs before execution to catch unwritten elements. Allocation sizes were
queried before memory access. The negative control changed no instruction
bytes and is **not** a test of an altered HWX signature.

The saved bundle contained:

- `H15D.e5`: 2,480 bytes. Readable strings identify
  `main__Op0_AneInference`, the compiler version, and a relative `.anehash` path.
- `main/main_ane/model.anehash`: 129 bytes, two hexadecimal hashes separated
  by an underscore.

No HWX file or HWX magic was found. The same two files were inventoried and
copied while the library and operation were still alive, so their limited
contents are not an artifact of cleanup. This is an exported runtime
descriptor with a program reference, not a recovered hardware executable.
The E5 file itself was not fully decoded into a formal semantics.

The controls demonstrate cross-process reuse on this M3 stack, despite the
different behavior reported in the pinned ANEForge reference. They do not
establish offline portability, survival after daemon-cache eviction, or the
absence of internal service work. We only assert that our fresh-process
control made no compiler API calls.

Raw journals, process records, bundle snapshots, outputs, source snapshots
and [independent verification](../benchmarks/ane_e5_probe/results/controller_verification.json)
are retained in `benchmarks/ane_e5_probe/results/`. `cache-reference-control.bundle`
contains the final, deliberately invalid reference; the unmodified source
remains in `compile-01/bundle_snapshot/`. The probe implementation is in
[ane_e5_probe](../benchmarks/ane_e5_probe/README.md).

## Decision for VeriTac

**No practical route for executing independently generated M3 ANE code was
demonstrated.** We can submit a graph to Apple's compiler and later execute a
saved reference to the resulting program. We have not obtained the hardware
program bytes or an ordinary-user interface for signing/loading our own
program. The inspected Objective-C surface and runtime observations are
consistent with a restricted loading path, but they do not prove every
possible interface is unavailable.

Consequently, an ANE frontend or DwarfStar offload path would still delegate
the hardware implementation to Apple. It could be useful application work,
but would not demonstrate VeriTac's vendor-independent lowering objective.
Further ANE work toward that objective should first establish access to a
real M3 executable **and** a permissible loading path for independently
produced code. Building a large Lean ANE backend before that would risk
producing a compiler whose output cannot be executed.

This investigation did not disable platform protections, change signing
identities, claim restricted entitlements, patch system binaries, inspect
unrelated model caches, or modify DwarfStar. A proof of program correctness
would not itself grant permission to Apple's loader.
