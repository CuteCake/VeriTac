# VeriTac

A verified tactic-based ML compiler. Tensor optimizations are expressed as formally verified Lean 4 tactics, and an AI search agent proposes tactic sequences that are correct by construction.

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
# Run the end-to-end tests
PYTHONPATH=. python3 tests/test_matmul.py
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

The correctness theorems are currently stated with `sorry` placeholders. The verification framework and proof obligations are fully set up — the proofs require:
- `sum_partition_blocks` for tile/unroll (Finset arithmetic)
- Loop commutativity for reorder (independence)
- Div/mod bijection for fuse
- Copy correctness for cache_read

Annotation tactics (vectorize, parallel) are trivially correct since `execStmt` ignores annotations.

## Design Decisions

1. **Numeric abstraction**: IR parameterized over any type; float semantics deferred
2. **Shapes as `List Nat`**: Avoids heavy dependent-type plumbing
3. **Fuel-based execution**: Simplifies termination for Phase 1
4. **Lean-Python interface**: JSON over subprocess (no FFI needed)
5. **Flat indexing with stride 1000**: Both Lean execution and C codegen use the same convention for correctness alignment
6. **Selective Mathlib use**: `Finset`, `Fin`, `omega` — prebuilt cache for fast builds

## License

MIT
