# VeriTac — Developer Guide

## Build & Test

```bash
lake build              # Build the Lean library
lake build veritac gemmini_check gemmini_program_check
PYTHONPATH=. python3 -m unittest discover -s tests
PYTHONPATH=. python3 tests/test_cuda_attention_demo.py
PYTHONPATH=. python3 tests/test_tiled_attention_plan_demo.py
PYTHONPATH=. python3 tests/test_matmul.py   # End-to-end tests
```

## Architecture

Two-level IR:
- **TExpr** (VeriTac/IR/): semantic "what to compute", parameterized over any type α
- **Stmt/SExpr** (VeriTac/Schedule/): scheduled "how to compute", loop nests with annotations

Tactics (VeriTac/Tactic/) transform Stmt → Stmt with individual correctness
theorems and explicit preconditions. Generic composition and emitted C are not
yet connected by a complete semantic proof.
Compose/Engine.lean chains tactics with precondition checking.
Main.lean handles loop and attention JSON requests. GemminiMain.lean checks
resource plans; GemminiProgramMain.lean checks concrete command/byte requests.
Runtime code lives under specializations/; formal module paths remain stable.
Package metadata describes proof scope and does not itself authorize acceptance.

## Conventions

- `autoImplicit = false` — all variables must be explicitly bound
- Lean toolchain pinned via `lean-toolchain` (updated by Mathlib compatibility)
- The generic loop/C path uses stride 1000 for flat indexing; Gemmini has its own row-major addressing contract.
- Python files require `PYTHONPATH=.` from project root

## Key files

- `VeriTac/Schedule/LoopNest.lean` — core IR types (Stmt, SExpr, execStmt)
- `VeriTac/Tactic/Tile.lean` — substExprVar/substStmtVar used by tile, fuse, unroll
- `VeriTac/Tactic/Reorder.lean` — varInExpr/varInStmt used by parallel, vectorize
- `Main.lean` — JSON parse/serialize, CLI entry point
- `specializations/cpu_gemm/emit_c.py` — C code generation with OpenMP pragmas
- `specializations/README.md` — backend ownership, entry points, and compatibility
- `specializations/*/specialization.json` — native implementations and proof scopes
- `CodeGen/` — compatibility imports; add new backend code under `specializations/`
- `Search/agent.py` — brute-force tactic enumeration
