# E5 Compiled-Bundle Export / Reopen Probe

Measured outcome: [M3 loader investigation](../../docs/ane_loader_investigation.md).
The bundle exports E5 metadata and an ANE cache reference, not HWX bytes.
Fresh-process execution of the unchanged bundle passed exact numerical
checks; changing only its cache reference failed operation creation.

`execute_control.py` is a fixed-workload controller check for the saved
identity + ReLU graph, not a general model runner. For example, from the
repository root, with a fresh output directory:

```sh
python3 benchmarks/ane_e5_probe/execute_control.py \
  --mode compile --path benchmarks/ane_e5_probe/fixtures/linear_relu/model.mil \
  --input benchmarks/ane/results/run-20260911-4/case-identity-c512s64/input.fp16.bin \
  --output /tmp/ane-e5-execution-new --deadline 55
```

Use a parent process timeout (the recorded runs used 60 seconds), since a
deadline checked between C calls cannot interrupt a blocked call. `--mode
reopen --path <saved outer .bundle directory>` makes no explicit compiler
calls. The negative-control bundle under `results/cache-reference-control.bundle`
is deliberately invalid; use the untouched `compile-01/bundle_snapshot`
for a valid reopen. Runtime access failed in the sandbox during buffer
allocation; the recorded execution controls used the normal launch context.

A bounded, inspectable probe that exercises Apple's private **Espresso
`e5rt_*`** C ABI (via Python `ctypes`, `dlsym` only) to (a) **compile** a MIL
into an ANE E5 bundle and (b) independently **reopen** that bundle and list its
functions. The exact ABI is taken from `reference/e5rt_api.h` (pinned ANEForge
`67c1d4861c562b0168d60c7a5677840e2cad44ea`, MIT — see `LICENSE.md`).

The probe never calls `e5rt_error_code_get_string` (it has a C++ `std::string`
ABI), never scans private caches, never signs, and never uses `sudo`.

## Files

```
ane_e5_probe.py          probe CLI (ctypes binding + compile/reopen/capabilities)
reference/e5rt_api.h     pinned ANEForge recovered C-API signatures (authoritative ABI)
reference/ane_e5rt_dispatch.mm  pinned ANEForge dispatch (compile_and_build_op provenance)
LICENSE.md               MIT attribution for the pinned ANEForge reference
tests/test_abi.py        pure validation tests (no hardware)
```

## Subcommands

```
capabilities --output DIR          list which e5rt_* symbols resolve (may run)
compile --mil PATH --output DIR [--prepare]    compile MIL -> E5 bundle + list functions
reopen   --bundle PATH --output DIR [--prepare] open a bundle + list functions (no compiler)
```

- `--prepare` (default **off**): additionally retain program function `main`,
  build precompiled-op create options (`operation name=main`,
  `allocate intermediate buffers=1`) and create the precompiled operation.
  This is **hardware preparation only** — no execute, no stream, no input
  binding.
- All subcommands take `--deadline SECONDS` (default 300).

## What `compile` does

Config (`cache_bundle_location = DIR/cache`), compiler via
`create_with_config` (out-FIRST, config-SECOND), options
(`device mask = 4` ANE, `force_recompilation = 1`, segmenter `graph`), one
`compile`, then records library function names (capped at 64, NULL names
rejected), then releases all handles in reverse.

While the library/operation handles are still alive, the compiled `cache/` is
copied to `DIR/bundle_snapshot/` (symlinks copied, not followed) and both the
live cache and the snapshot are SHA-256 inventoried. `DIR/bundle_snapshot/` is
the preserved compiled-bundle evidence for a later independent `reopen`.

## What `reopen` does

`program_library_create(out, path)` (out-FIRST) + list functions. No
compiler-API calls. With `--prepare`, additionally builds the precompiled
operation from the reopened bundle's `main`.

## Safety & accounting

- Every e5rt call checks `rc` and non-NULL out; every `release` takes
  `void **` via `byref` and is attempted even if a prior release fails.
- A JSON **stage journal** (`journal.json`) is flushed before/after every call
  so a crash inside a private API call is attributable to the exact stage.
  `status` is only `completed` after cleanup finishes.
- `final.json` records function names, alive/after inventories, the snapshot,
  and (on failure) the scoped error + own-dir inventory. Failed artifacts are
  preserved.
- Existing output dirs are rejected atomically (`makedirs exist_ok=False`);
  only own `DIR` and the explicit bundle are hashed (regular files only).

## Proof boundary

This probe establishes **actual E5 bundle export** (the compiler wrote a
bundle to `DIR/cache`, snapshotted while the library is alive) and
**independent reopen evidence** (a fresh `program_library_create` + function
list against that bundle / snapshot). It does **not** claim programmability
from library creation: `--prepare` only creates a precompiled operation, and
nothing is executed, timed, or driven for correctness.

## Verification procedure (root-run)

Reference workload:
`benchmarks/ane/results/run-20260911-4/case-identity-c512s64/{model.mil,weight.bin}`;
stage `weight.bin` at `weights/weight.bin` beside the mil before compiling.

1. **Same-process positive**: `compile --mil <model.mil> --output <newdir> [--prepare]`
   → expect `function_names: ["main"]`, `status: completed`, and
   `bundle_snapshot_inventory` populated.
2. **Fresh-process reopen positive**: `reopen --bundle <newdir>/bundle_snapshot --output <dir2>`
   → expect `function_names: ["main"]` again (independent open of the exported bundle).
3. **Fresh-process negative control**: `reopen --bundle <nonexistent> --output <dir3>`
   → expect clean failure (`status: failed`), preserved artifacts, and no crash.

Root orchestrates the parent timeout process. These steps must NOT be run
until root review.

## Tests (pure, no hardware)

```
PYTHONPATH=. python3 -m unittest benchmarks/ane_e5_probe/tests/test_abi.py -v
```

Validates the ctypes ABI against `reference/e5rt_api.h` (incl.
out-FIRST/out-LAST and `void **` releases), plus a fake-functions compile /
prepare / reverse-cleanup flow, caps, NULL-name rejection, atomic outdir
rejection, symlink-safe inventory, and failure JSON.
