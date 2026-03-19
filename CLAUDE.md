# VeriTac — Developer Guide

## Build & Test

```bash
lake build              # Build Lean library (534 modules)
lake build veritac      # Build CLI binary → .lake/build/bin/veritac
PYTHONPATH=. python3 tests/test_matmul.py   # End-to-end tests
```

## Architecture

Two-level IR:
- **TExpr** (VeriTac/IR/): semantic "what to compute", parameterized over any type α
- **Stmt/SExpr** (VeriTac/Schedule/): scheduled "how to compute", loop nests with annotations

Tactics (VeriTac/Tactic/) transform Stmt → Stmt. Each has a `sorry`-ed correctness theorem.
Compose/Engine.lean chains tactics with precondition checking.
Main.lean is the JSON CLI that bridges Lean and Python.

## Conventions

- `autoImplicit = false` — all variables must be explicitly bound
- Lean toolchain pinned via `lean-toolchain` (updated by Mathlib compatibility)
- Flat buffer indexing uses stride 1000 (both Lean `flatIndex` and C codegen)
- Python files require `PYTHONPATH=.` from project root

## Key files

- `VeriTac/Schedule/LoopNest.lean` — core IR types (Stmt, SExpr, execStmt)
- `VeriTac/Tactic/Tile.lean` — substExprVar/substStmtVar used by tile, fuse, unroll
- `VeriTac/Tactic/Reorder.lean` — varInExpr/varInStmt used by parallel, vectorize
- `Main.lean` — JSON parse/serialize, CLI entry point
- `CodeGen/emit_c.py` — C code generation with OpenMP pragmas
- `Search/agent.py` — brute-force tactic enumeration
