# ANE Runtime Capability Inventory Probe (read-only)

A small, **read-only** Objective-C probe that inventories the *exposed* API
surface of Apple's private ANE runtime frameworks on this machine
(M3 Ultra, Mac15,14, macOS 26.6.2). It answers a discovery-only question for
the root controller: **what entry points exist for (a) obtaining an
already-compiled ANE executable, (b) loading externally produced programs,
and (c) signing separately from compilation** — and reports precompiled-model
runtime knobs that distinguish production from VM/test support.

## Scope and safety

- `dlopen`s `AppleNeuralEngine.framework` and `ANECompiler.framework`,
  reporting **actual dlopen success/error** separately from bundle presence.
- Uses `objc_*` runtime APIs (`class_copyMethodList`, `class_copyPropertyList`,
  `objc_copyClassList`, `objc_copyProtocolList`,
  `protocol_copyMethodDescriptionList`) and reads method/property/protocol
  names and type encodings.
- **No model compile/load/eval, no cache enumeration, no `sudo`, no security
  changes, no binary patches.** No flags, boot-args, or code-sign identity are
  set, and no `_ANEVirtualClient` methods are invoked.

### Authorized read-only getters (the only selectors invoked)

Only these **zero-argument read-only class getters** are called, and only
after verifying the discovered type encoding matches the expected return type
(`B` for BOOL, `@` for NSString), each wrapped in `@try/@catch`:

| Class | Selector | Return |
|---|---|---|
| `_ANEDeviceInfo` | `precompiledModelChecksDisabled` | BOOL |
| `_ANEStrings` | `vm_allowPrecompiledBinaryBootArg` | NSString |
| `_ANEStrings` | `testing_external_precompiledModelPath` | NSString |
| `_ANEStrings` | `compilerServiceAccessEntitlement` | NSString |
| `_ANEStrings` | `secondaryANECompilerServiceAccessEntitlement` | NSString |

Results are stored under `runtime_observations` (with status, matched
encoding, and value or exception). On this production M3 Ultra they indicate
precompiled-model checks are **enabled** (`precompiledModelChecksDisabled =
false`) — useful for distinguishing production vs VM/test precompiled support.

## Output

Structured, deterministic JSON is written to `results/inventory.json`
(sorted keys/arrays, **no timestamp**, framework `dlopen` status + `CFBundle`
versions, class → method/property tables, protocol → required/optional
instance/class method tables). Relevant private classes for the stated
objective include `_ANEInMemoryModel`, `_ANEModel`, `_ANEClient`,
`_ANECompiler`/compiler types and cache/path storage types — surfaced here for
the controller to review before choosing any later API call. Note that
`ANECompiler.framework` is C++ and exposes no ObjC classes to the runtime, so
only its framework version is captured.

JSON/write failures cause a non-zero exit instead of a false success.
No object values, pointers, or environment secrets are emitted.

## Files

```
src/ane_inventory.m   introspection probe (single file, <200 lines)
Makefile              host build + run
results/inventory.json owned structured output (generated)
```

## Build & run

```bash
make -C benchmarks/ane_loader run   # from repository root
```

Host build+run only. Requires Xcode command-line tools.
