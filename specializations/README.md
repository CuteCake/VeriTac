# Backend specializations

Runtime implementations live here, grouped by computation and target. There is
no universal backend class or common kernel IR. Each package retains its native
program representation, checker protocol, and execution tools.

```text
specializations/
  catalog.py                 Metadata discovery; no target imports or execution
  cpu_gemm/                  Loop JSON → C/OpenMP and host execution
  gemmini_gemm/              Commands, raw encoding, certificates, checked rewrites
    runtime/                 Spike setup and execution tools
    blind_search.py          Live tool-denied proposals and byte certification
    diagnostics.py           Advisory, bounded rejection explanations
    evaluation.py            Post-proposal comparison with existing generators
    schedule_ir.py           Untrusted bounded loop/expression expansion
  tenstorrent_protocol/      Exact-copy async protocol IR, graph certificates, live search
    runtime/                 Metalium emission and local ARM64 ttsim setup/replay
  metal_attention/           Metal driver and Swift sources
    partitioned/             Partitioned and vendor-derived attention variants
  cuda_attention/
    kernels/                 CUDA drivers, kernels, vendor headers and licenses
```

Each package has a `specialization.json` describing its scope, native entry
points, implementation files, formal sources, proof coverage, boundaries, and
evidence locations. These files are navigation metadata, not certificates or
an implemented formal frontend contract. Existing validators remain authoritative.

```bash
python3 -m specializations list
python3 -m specializations show gemmini_gemm
python3 examples/gemmini_gemm_demo.py
python3 specializations/gemmini_gemm/runtime/run_raw_spike.py --help
```

Listing packages uses only the Python standard library; it does not import
NumPy, PyTorch, MLX, or initialize devices. The catalog does not execute commands
from descriptions. Native entry points keep their own argument schemas.

## Code ownership and shared layers

- `VeriTac/IR`, `Schedule`, `Tactic`, and `Compose` remain the reusable Lean
  foundations. `VeriTac/Attention` contains mathematical and plan-specific work;
  `VeriTac/Gemmini` contains the current Gemmini instruction proof path.
- Lean definitions, namespaces, CLI binaries, and file paths are unchanged in
  this refactor. Saved proof dependencies can therefore be replayed without
  migrating the formal library. Package descriptions locate the relevant modules.
- `Search/` and `Hardware/` retain search and hardware-profile helpers. Some
  orchestration remains in `examples/`; this refactor does not claim a common
  search engine or full model importer.
- `benchmarks/` owns surveys, recorded results, raw outputs, and proof evidence.
  Active kernel implementations and runtime tools have moved here. Historical
  artifacts are preserved verbatim, including old paths and source hashes.
- `tests/` exercises canonical implementations and old entry-point compatibility.

## Compatibility and deployment

`CodeGen.lower`, `emit_c`, `runner`, and `gemmini*` are compatibility modules.
Imports resolve to the canonical module object so class identity, cached state,
and monkeypatches work across old and new imports. New callers should import
`specializations.cpu_gemm` or `specializations.gemmini_gemm` directly.

Legacy benchmark driver and kernel paths are relative symlinks to the canonical
sources. Metal result JSON stays in its original benchmark directory. CUDA and
partitioned Metal sources retain adjacent assets and licenses. On a remote host,
copy the canonical package subtree, not just the compatibility symlink; the
standalone runtime drivers do not require installing the full repository.
This compatibility layout assumes a checkout with symlink support, as on the
currently supported macOS/Linux development hosts.

The next step is the actual computation/target contract adapter described in
[the meta-frontend design](../docs/frontend_design.md). Package discovery alone
does not implement that contract or increase any backend's proof coverage.

Gemmini also has a [live blind-search controller](../docs/gemmini_blind_search.md).
It fixes each task contract, uses the existing byte checker for proposals, and
requires a kernel certificate for the final winner. Its compact schedule adapter
is a separate untrusted expansion layer; neither feature changes the formal ISA
or arithmetic coverage.

Tenstorrent begins at a separate [restricted asynchronous protocol layer](../docs/tenstorrent_protocol.md).
The checker certifies a finite state graph and its all-input origin semantics;
the official Blackhole simulator executes generated Metalium programs on the
local Mac Studio via Linux ARM64 Docker. This evidence does not extend the
Gemmini byte theorem to Tenstorrent or prove the emitted C++/ELF.
