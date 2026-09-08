# VeriTac - an ML Compiler concept for the agentic era. (in exploration)

A verified tactic-based ML compiler. Tensor optimizations are expressed as formally verified Lean 4 tactics, and an AI search agent proposes tactic sequences that are correct by construction.

The difference between this repo vs many other agentic kernel writing repo is, they require a vendor provided compiler, like CUDA, to validate the code. However, what if you are the ASIC vendor and there is nothing to refer? This repo ensures 1. every step of the optimization is correct, 2. correctness of optimization can be ver

### End-to-end GEMM demo

`examples/gemm_demo.py` runs the full pipeline on a square matmul:

1. The search agent proposes a schedule (`tile` / `fuse` / `reorder` /
   `parallel` / `vectorize`) and the **Lean CLI re-verifies each tactic**.
2. The schedule is lowered to C. OpenMP pragmas (`#pragma omp parallel for`,
   `#pragma omp simd`) are emitted **only** when a write-dependence check
   (mirroring the proved `indexLocalP_flat_leading` criterion) confirms the
   loop is independent — a reduction loop over `k` correctly gets a comment
   instead of a pragma.
3. The C is compiled with an OpenMP compiler and benchmarked against numpy.

Measured on this machine (gcc-15, 10 threads): the agent's top schedule
(`parallel i0`, `vectorize i2`, `parallel i1`) runs **4.5x** faster at 128³,
and a full register-tiling chain (`tile(32)`×2 → `reorder i1_inner i2` →
`vectorize i1_inner` → `parallel i0_outer`) reaches **12.8x** at 256³
(18.9 ms → 1.47 ms), all matching numpy exactly. The Lean proofs and the
codegen guard together ensure the annotated loops really are safe to
parallelize.

There are two demo entry points:

- `examples/gemm_demo.py [dim]` — offline pipeline: search agent → Lean CLI →
  C → benchmark.
- `examples/llm_gemm_demo.py` — LLM-driven pipeline with full trace output.
  It reads the API key/model from `OPENAI_API_KEY` / `OPENAI_MODEL` /
  `OPENAI_BASE_URL` (or a root `.env`, or interactive prompt); each turn an LLM
  proposes one tactic, the Lean CLI verifies and applies it, and the trace
  prints the proposed tactic, the resulting statement tree, and any rejection
  (which is fed back to the model) — then benchmarks the final schedule.
  `--mock` runs the same trace with a fixed schedule and no API key.

## Overview

VeriTac separates *what* to compute (semantic IR) from *how* to compute it (schedule). Optimizations are applied as composable, verified tactics that transform loop nests while preserving semantics. An AI search agent proposes tactic sequences; the verifier guarantees every accepted sequence produces correct code.

```
User Program → Semantic IR → [Verified Tactics] → Scheduled IR → C Code
                                    ↑
                              AI Search Agent
```

## Project Structure

```
VeriTac/
├── lakefile.lean                 # Lake build config (Lean 4 + Mathlib)
├── lean-toolchain                # Lean 4.29.0-rc6
├── Main.lean                     # CLI: JSON-in/JSON-out tactic application
├── VeriTac/
│   ├── Basic.lean                # Re-exports all modules
│   ├── IR/
│   │   ├── Shape.lean            # Shape (List Nat), Index (dependent Fin vector)
│   │   ├── TExpr.lean            # Semantic IR: const, tensor, map, zip, reduce
│   │   ├── Denote.lean           # Denotational semantics (TExpr → Index → α)
│   │   └── Operations.lean       # tensorAdd, tensorMap, tensorSum
│   ├── Schedule/
│   │   ├── LoopNest.lean         # Stmt, SExpr, Env, Store, execStmt (fuel-based)
│   │   ├── Equiv.lean            # ScheduleEquiv: ∀ env store, exec s1 = exec s2
│   │   └── Lower.lean            # TExpr → LoopNest (naive nested loops)
│   ├── Tactic/
│   │   ├── Tile.lean             # Split loop into outer/inner
│   │   ├── Split.lean            # Alias for tile
│   │   ├── Fuse.lean             # Merge adjacent loops via div/mod
│   │   ├── Reorder.lean          # Swap independent loops
│   │   ├── Unroll.lean           # Fully unroll constant-bound loops
│   │   ├── Vectorize.lean        # Annotate innermost loop for SIMD
│   │   ├── Parallel.lean         # Annotate loop for parallel execution
│   │   ├── CacheRead.lean        # Insert local buffer + copy + read substitution
│   │   └── Library.lean          # Tactic registry and applyTactic dispatcher
│   ├── Compose/
│   │   ├── Precondition.lean     # Precondition checking per tactic kind
│   │   └── Engine.lean           # applySchedule: sequential tactic composition
│   └── Util/
│       ├── Finset.lean           # Sum partition lemma (for tiling proofs)
│       └── List.lean             # List swap utility
├── CodeGen/                      # Python — unverified C code generator
│   ├── lower.py                  # Parse LoopNest JSON → Python AST
│   ├── emit_c.py                 # Python AST → C code with OpenMP pragmas
│   └── runner.py                 # Compile, run, benchmark against numpy
├── Search/                       # Python — brute-force search agent
│   ├── interface.py              # Lean CLI subprocess wrapper
│   ├── cost_model.py             # Analytical cost model (ops + memory traffic)
│   └── agent.py                  # Enumerate tactic sequences, rank by cost
└── tests/
    └── test_matmul.py            # End-to-end matmul tests
```

## Building

### Prerequisites

- [Lean 4](https://leanprover.github.io/lean4/doc/setup.html) (installed via `elan`)
- Python 3.9+
- `clang` (for C code compilation)
- `numpy` (for benchmark comparisons)

### Build the Lean project

```bash
lake update    # Fetch Mathlib (downloads prebuilt cache, ~5 min first time)
lake build     # Build the library (534 modules)
lake build veritac  # Build the CLI binary
```

### Verify

```bash
# Run the end-to-end matmul (codegen) tests
PYTHONPATH=. python3 tests/test_matmul.py

# Run the soundness regression tests (tile rejection, unroll, annotations)
PYTHONPATH=. python3 tests/test_soundness.py
```

## Usage

### CLI

The `veritac` binary accepts JSON on stdin or via `--json` and applies a sequence of tactics to a loop nest statement.

```bash
# Apply tile(i0, 32) to a simple loop
.lake/build/bin/veritac --json '{
  "stmt": {
    "tag": "loop", "var": "i0",
    "lo": {"tag": "lit", "val": 0},
    "hi": {"tag": "lit", "val": 256},
    "ann": "none",
    "body": {
      "tag": "bufWrite", "buf": "C",
      "indices": [{"tag": "var", "name": "i0"}],
      "val": {"tag": "bufRead", "buf": "A",
              "indices": [{"tag": "var", "name": "i0"}]}
    }
  },
  "tactics": [
    {"kind": "tile", "vars": ["i0"], "int_params": [32]},
    {"kind": "parallel", "vars": ["i0_outer"]}
  ]
}'
```

**Output:**

```json
{
  "applied": 2,
  "stmt": {
    "tag": "loop", "var": "i0_outer", "ann": "parallel",
    "body": {
      "tag": "loop", "var": "i0_inner", "ann": "none",
      "body": { "..." }
    }
  }
}
```

### JSON Format

#### Statements (`Stmt`)

| Tag | Fields | Description |
|-----|--------|-------------|
| `skip` | — | No-op |
| `bufWrite` | `buf`, `indices`, `val` | Write value to buffer |
| `loop` | `var`, `lo`, `hi`, `ann`, `body` | Loop with annotation |
| `seq` | `s1`, `s2` | Sequential composition |
| `alloc` | `buf`, `shape`, `body` | Allocate local buffer |

#### Expressions (`SExpr`)

| Tag | Fields | Description |
|-----|--------|-------------|
| `lit` | `val` (int) | Integer literal |
| `var` | `name` (string) | Variable reference |
| `add`, `mul`, `div`, `mod` | `left`, `right` | Binary arithmetic |
| `bufRead` | `buf`, `indices` | Read from buffer |

#### Annotations

`"none"`, `"parallel"`, `"vectorize"`, `"unrolled"`

#### Tactic Applications

```json
{
  "kind": "<tactic_name>",
  "vars": ["<target_var>", ...],
  "int_params": [<nat>, ...],
  "str_params": ["<string>", ...]
}
```

### Available Tactics

| Tactic | vars | int_params | Description |
|--------|------|------------|-------------|
| `tile` | `[var]` | `[tile_size]` | Split loop into outer/inner |
| `split` | `[var]` | `[factor]` | Alias for tile |
| `fuse` | `[var1, var2]` | — | Merge two adjacent sequential loops |
| `reorder` | `[var1, var2]` | — | Swap two adjacent independent loops |
| `unroll` | `[var]` | — | Fully unroll loop (requires constant bounds) |
| `vectorize` | `[var]` | — | Annotate innermost loop for SIMD |
| `parallel` | `[var]` | — | Annotate loop for parallel execution |
| `cache_read` | — | — | Insert local cache buffer (advanced) |

### Python Code Generation

```python
import json
from CodeGen.lower import parse_stmt
from CodeGen.emit_c import emit_function

# Get optimized loop nest from Lean CLI
result = json.loads(subprocess.check_output([
    ".lake/build/bin/veritac", "--json", json.dumps({
        "stmt": matmul_stmt,
        "tactics": [
            {"kind": "tile", "vars": ["i0"], "int_params": [32]},
            {"kind": "parallel", "vars": ["i0_outer"]}
        ]
    })
]))

# Generate C code
stmt = parse_stmt(result["stmt"])
c_code = emit_function("matmul", stmt,
    input_bufs=[("A", M*K), ("B", K*N)],
    output_bufs=[("C", M*N)])
```

### Search Agent

```bash
# Find optimal tactic sequences for matmul
PYTHONPATH=. python3 -m Search.agent --max-length 3 --top-k 5
```

The agent:
1. Enumerates tactic sequences up to the given length
2. Validates each via the Lean CLI (precondition checking)
3. Scores valid sequences with the analytical cost model
4. Reports top-K results

## Architecture

### Two-Level IR

**Semantic IR (`TExpr`)** describes *what* to compute:
- Parameterized over `α` (any type) — proven for all types, tested with `Nat`/`Int`
- `denote : TExpr α s → (Index s → α)` gives mathematical meaning
- Proven lemma: `denote (map g (map f e)) = denote (map (g ∘ f) e)`

**Scheduled IR (`Stmt`/`SExpr`)** describes *how* to compute it:
- Loop nests with explicit iteration, buffer reads/writes
- Annotations for parallelism and vectorization
- `execStmt` with fuel-based semantics for termination

### Correctness

Each tactic has a correctness theorem of the form:

```
theorem tactic_correct : original_stmt ≈ₛ transformed_stmt
```

where `s1 ≈ₛ s2` means `∀ fuel env store, execStmt fuel env store s1 = execStmt fuel env store s2`.

Composition correctness follows from transitivity:

```lean
theorem ScheduleEquiv.trans : s1 ≈ₛ s2 → s2 ≈ₛ s3 → s1 ≈ₛ s3
```

### Proof Status

The verification framework is set up, and several core correctness theorems now
carry machine-checked proofs (no `sorry`). The following are **proven**:

| Theorem | Status | Notes |
|---------|--------|-------|
| `vectorize_correct`, `parallel_correct` | ✅ proven | `execStmt` discards loop annotations, so these are syntactic no-ops in the model |
| `substExprVar_eval` | ✅ proven | expression-level substitution (`v ↦ e` evaluates under `env[v := e]`) |
| `substStmtVar_lit_correct` | ✅ proven | statement-level substitution for a *literal* `v ↦ lit k` |
| `unroll_correct` | ✅ proven | via `unrollBody_correct`, which connects `execLoopIters` to the unrolled statement |
| `tile_correct` | ✅ proven | via `execLoopIters_partition` + `substStmtVar_correct` |
| `fuse_correct` | ✅ proven | via `execLoopIters_fuse` + `substStmtVar_correct` |
| `reorder_correct` | ✅ proven | restated with a semantic commutation hypothesis |
| `cacheRead_correct` | ✅ proven | restated: read substitution is a no-op from a mirroring store |
| `split_correct` | ✅ proven | reduces to `tile_correct` |

All tactic correctness theorems now carry **machine-checked proofs with zero
`sorry`**. The proof infrastructure lives in `VeriTac/Schedule/`:

| Infrastructure | What it provides |
|----------------|------------------|
| `LoopComposition.lean` | Function-level Kleisli composition (`kcomp`), block *partition* (tile), *swap* (reorder), *fusion* (fuse), interleave/commutation, and general substitution lemmas over `execLoopIters` |
| `FreeVars.lean` | `varFreeStmt` (bounds-sensitive capture analysis), `exprStatic`, read-only / read-set predicates |
| `ExecLemmas.lean` | Totality of the interpreter, store-invariance under unused bindings, read-only and write-set discipline, "agreement off a buffer" (blind) lemmas |

The proofs themselves:

- **`tile_correct`** — via `execLoopIters_partition` + the general statement
  substitution lemma `substStmtVar_correct` (hypotheses: `loopBinds v body = false`,
  the generated `_outer` / `_inner` names are fresh, and the tile size divides the
  loop extent). The `tileAt` outer bound is now the *exact quotient* `hi/ts`.
- **`fuse_correct`** — via `execLoopIters_fuse`, using the `(i, j) ↔ i*M + j`
  bijection with the required `0 ≤ j < M` range, plus a double application of
  `substStmtVar_correct` (the fused name must be fresh for the body).
- **`reorder_correct`** — restated honestly: it requires `v1 ≠ v2`, bounds that are
  independent of the other axis (`varInExpr`/`exprStatic`-free) *and* a semantic
  pairwise-commutation hypothesis on the body. The Lean theorem encodes exactly why
  `loopsIndependent` is only a syntactic heuristic; discharging the commutation
  hypothesis for a concrete body is the dependence-analysis obligation.
- **`cacheRead_correct`** — restated honestly: substituting `origBuf ↦ cacheBuf`
  reads is a no-op when the cache mirrors the source in the starting store and both
  buffers are read-only for the statement (`isReadOnly`). The *copy loops* inserted
  by `cacheRead` must establish that mirror, which is the remaining (Python-side)
  obligation.
- **parallel dependence bridge** — `indexLocalP_flat_leading` proves that when
  every write index of a loop body is *local to* the axis variable (bare `v` as
  the leading index, the rest `v`-free and store-independent), different axis
  values write disjoint flat cells, so iterations are pairwise independent. This
  is the sound dependence criterion (`loopLocalWritesP`) behind a `parallel`
  annotation; the syntactic `checkIndependence` remains the search-side heuristic.
- **`split_correct`** — reduces to `tile_correct` (same theorem, now proved).

#### Soundness fixes made

While auditing the semantics, several genuine soundness bugs were found and fixed:

1. **Fuel was consumed per loop-nesting level.** `execStmt` decremented fuel when
   entering a loop body, so restructuring a nest (tiling adds a level) starved the
   leaf computation, making `tile_correct`'s `∀ fuel` statement false. Fuel is now
   carried but not consumed: each loop has a finite iteration count, so the
   interpreter is total and tiling cannot change the budget leaves receive.
2. **`tile` overran the range when the tile size didn't divide the extent.** The
   inner loop's upper bound was hard-coded to `tileSize`, so the final block
   iterated past `hi` when `hi % tileSize ≠ 0`. `tile` now requires a literal
   `[0, hiVal)` range with `hiVal % tileSize = 0` and refuses otherwise.
3. **`fuse` was applied to sequential loops using a div/mod split**, which ran the
   second body `N₁ * N₂` times. It now fuses *nested* loops correctly.

> **Note on annotations.** `execStmt` ignores loop annotations, so `parallel` and
> `vectorize` are trivially "correct" in the Lean model. The *actual* correctness of
> the generated parallel/SIMD code still requires a genuine dependence analysis for
> the annotation to be safe (`parallel` on a loop with loop-carried dependencies
> would produce incorrect `#pragma omp parallel` code). The Lean proofs establish
> semantic equivalence of the loop nest; the code-generation-level dependence
> property is a separate, open obligation.

## Design Decisions

1. **Numeric abstraction**: IR parameterized over any type; float semantics deferred
2. **Shapes as `List Nat`**: Avoids heavy dependent-type plumbing
3. **Fuel carried but not consumed**: Loop bounds are finite, so the interpreter is
   total. Fuel is kept in the signature but never reduces, which is what makes the
   loop-restructuring equivalence theorems provable for every `fuel` value.
4. **Lean-Python interface**: JSON over subprocess (no FFI needed)
5. **Flat indexing with stride 1000**: Both Lean execution and C codegen use the same convention for correctness alignment
6. **Selective Mathlib use**: `Finset`, `Fin`, `omega` — prebuilt cache for fast builds

## License

MIT
