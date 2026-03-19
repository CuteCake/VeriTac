# Verified Tactic-Based ML Compiler: Design Plan

## 1. Executive Summary

This document proposes **VeriTac**, a next-generation ML compiler that replaces traditional compiler passes with a formally verified **tactic library** and an **AI-driven search agent**. The core idea: correctness is encoded into pre-verified, composable transformation primitives (tactics), while performance optimization is delegated entirely to AI. The AI cannot produce incorrect code — it can only produce slow code.

---

## 2. Prior Work & Landscape

### 2.1 Current ML Compilers and Their Limitations

**TVM / Meta Schedule (Apache, 2018–present)**
TVM pioneered the separation of computation description from scheduling. Its schedule primitives — `split`, `reorder`, `fuse`, `vectorize`, `cache_read`, `compute_at`, `unroll`, `bind` — are the closest existing analog to our tactic library. TVM's Meta Schedule (3rd generation) added a design-space DSL that lets developers describe spaces of possible schedules and uses auto-tuning to search them. However, TVM's primitives are *unverified* — correctness is checked by testing, and certain compositions produce wrong code, especially around boundary conditions when tile sizes don't evenly divide loop bounds.

**MLIR (Google/LLVM, 2019–present)**
MLIR provides a flexible multi-level IR framework with an ecosystem of dialects (linalg, affine, scf, gpu, etc.) and passes. Its power lies in extensibility, but passes are trusted by convention — no formal framework prevents a pass from silently changing semantics. The proliferation of custom dialects and passes makes end-to-end correctness increasingly hard to maintain.

**Triton (OpenAI, 2019–present)**
Triton offers a Python-like DSL for writing GPU kernels with automatic tile-level optimizations. Its compiler handles memory coalescing, shared memory allocation, and thread synchronization, but the optimizations are opaque and unverified. Recent work (GEAK, TritonRL, TritonForge, AutoTriton) shows the rapid growth of LLM-based kernel generation and optimization for Triton, validating the trend of AI-driven optimization.

**Halide (MIT/Google, 2013–present)**
Halide pioneered the strict separation of algorithm and schedule, which directly inspires our architecture. The algorithm defines *what* to compute; the schedule defines *how*. Halide's term rewriting system (TRS) has over 1000 handwritten rules, and a 2020 OOPSLA paper (Newcomb et al.) used Z3 and Coq to verify many of these rules — finding bugs in the process. Z3 automatically verified the majority of rules, but 123 rules involving division and modulo required manual Coq proofs.

### 2.2 Verified Compilation

**CompCert (INRIA, 2005–present)**
The gold standard for verified compilation. CompCert compiles a large subset of C to ARM/PowerPC/RISC-V/x86 with a machine-checked Coq proof that the generated assembly preserves the semantics of the source. Key lessons: (a) it's feasible to verify a realistic compiler, (b) the proof effort is enormous (~100k lines of Coq), (c) the verified compiler's performance is competitive with GCC -O1 but not -O2/-O3. CompCert proves that each pass preserves a simulation relation between source and target semantics.

**Verified Polyhedral Code Generation (Leroy et al., POPL 2021)**
Extended CompCert-style verification to polyhedral code generation, proving correct the lowering from a polyhedral representation (schedules, polyhedra) to loop nests. The formalization uses Coq and the VPL library for certified polyhedral operations. This is directly relevant — it shows that loop transformation correctness *can* be mechanized, but at significant proof engineering cost.

**Verified Functional Tensor Language Compiler (Liu et al., PLDI 2024)**
Verified a compiler from a functional tensor language (ATL) to imperative loop nests, with separate control of compute and storage ordering. Notably, this exercise *revealed a soundness bug* in the original published compilation algorithm, demonstrating the value of formal verification. The work is done in Coq and achieves performance comparable to Halide.

**Halide Translation Validation (Clément & Cohen, OOPSLA 2022)**
An end-to-end translation validation approach for Halide that checks each compilation output a posteriori, rather than verifying the compiler once and for all. This is an alternative to our approach — instead of pre-verifying tactics, you verify each specific compilation. The tradeoff: translation validation catches bugs per-compilation but doesn't guarantee all future compilations are correct.

### 2.3 Equality Saturation and E-Graphs

**egg (Willsey et al., POPL 2021)**
A fast, extensible e-graph library for equality saturation. E-graphs compactly represent exponentially many equivalent programs, and equality saturation applies all rewrite rules simultaneously, avoiding the phase ordering problem. The egg library has been applied to floating-point accuracy (Herbie), CAD simplification (Szalinski), and tensor graph optimization (TENSAT).

**TENSAT (Yang et al., MLSys 2021)**
Applied equality saturation to tensor computation graph optimization, achieving up to 16% speedup over TASO with 48x less optimization time. Extended e-graphs to support multi-pattern rewrite rules needed for DL graph substitutions. Demonstrates that principled rewrite-based approaches outperform heuristic-based graph rewriting.

**Guided Equality Saturation (Koehler et al., POPL 2024)**
Introduced guided equality saturation with a prototype *Lean 4 tactic* that bridges egg and Lean, enabling equational proofs via equality saturation within the Lean theorem prover. This is a direct predecessor to our approach — it shows that e-graph-based reasoning can be integrated with a proof assistant.

### 2.4 AI for Kernel Optimization (2025 State of the Art)

The landscape has exploded in 2025:

- **GEAK (AMD, Aug 2025)**: Agentic AI system for Triton kernel generation on AMD GPUs, achieving up to 2.59x speedup over reference kernels using LLM-based generation + reflection + optimization.
- **TritonRL (Oct 2025)**: RL-trained 8B LLM specialized for Triton, using hierarchical verifiable rewards to prevent reward hacking.
- **TritonForge (Dec 2025)**: Profiling-guided LLM framework that reads NVIDIA Nsight Compute metrics and iteratively optimizes Triton kernels.
- **Liger Kernel (LinkedIn)**: Production Triton kernels for LLM training, demonstrating 20% throughput increase and 60% memory reduction.
- **LOOPerSet (Oct 2025)**: A 28M-sample dataset for training ML-based polyhedral cost models, with every transformation verified for correctness via dependency analysis.

### 2.5 Lean 4 and TensorLib

Lean 4 provides both a dependently-typed proof assistant and an efficient programming language. Lean's metaprogramming system allows custom tactic creation — users can define domain-specific automation that produces machine-checked proofs. Lean's Mathlib provides formalized real analysis, linear algebra, and number theory. The official `leanprover/TensorLib` project is building a verified tensor library in Lean. HepLean has demonstrated formalized tensor index notation with automated rewriting tactics.

---

## 3. Architecture Overview

```
                        ┌─────────────────────┐
                        │   User Program       │
                        │  (PyTorch / JAX)     │
                        └──────────┬──────────┘
                                   │ trace/export
                                   ▼
                        ┌─────────────────────┐
                        │   Semantic IR (SIR)  │
                        │  Typed tensor ops    │
                        │  with denotational   │
                        │  semantics           │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼───────────────────────┐
          │                        │                       │
          ▼                        ▼                       ▼
┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│  Verified Tactic │   │  AI Search Agent   │   │  Hardware Cost   │
│  Library (Lean4) │◄──│  (LLM / RL)        │──►│  Model           │
│                  │   │                    │   │                  │
│  ~30 primitives  │   │  Proposes tactic   │   │  Estimates perf  │
│  Each with proof │   │  sequences         │   │  of candidates   │
└──────────────────┘   └────────────────────┘   └──────────────────┘
          │                        │
          │   All compositions     │
          │   are correct by       │
          │   construction         │
          ▼                        ▼
                        ┌─────────────────────┐
                        │  Execution IR (EIR)  │
                        │  Scheduled, tiled,   │
                        │  hardware-specific   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Target Code         │
                        │  (CUDA/PTX, Metal,   │
                        │   CPU SIMD, etc.)    │
                        └─────────────────────┘
```

---

## 4. The Tactic Library

### 4.1 Design Principles

1. **Small, orthogonal set.** We target ~25-35 tactics covering loop transformations, memory transformations, parallelism transformations, and numerical transformations. Each tactic does one thing.

2. **Parameterized.** Each tactic takes parameters (tile sizes, axis indices, memory scopes). The proof covers all valid parameter values, with preconditions defining "valid."

3. **Composable by construction.** The output of any tactic is a valid input for any other tactic (subject to preconditions). Correctness of compositions follows from the correctness of each step — just like in Lean's tactic proofs.

4. **Preconditions as types.** Invalid tactic applications are rejected at composition time, not at runtime. The precondition system acts as a type system over the transformation space.

5. **Proved in Lean 4.** Each tactic has a machine-checked proof in Lean 4 that it preserves the denotational semantics of the IR. We use Lean rather than Coq because (a) Lean 4 is both a proof assistant and an efficient programming language, (b) the metaprogramming system enables custom domain-specific tactics, and (c) the growing Mathlib and TensorLib ecosystem provides foundations we can build on.

### 4.2 The Semantic IR

The Semantic IR represents tensor computations as typed expressions with denotational semantics. Every node in the IR has a *meaning function* mapping it to a mathematical function over indices.

```lean
-- Core IR types
inductive Shape where
  | scalar : Shape
  | tensor : List Nat → Shape

inductive DType where
  | float32 | float16 | int32 | int8 | ...

-- An IR expression with its shape and dtype
inductive TExpr : Shape → DType → Type where
  | const    : (val : α) → TExpr .scalar α
  | tensor   : (data : Index s → α) → TExpr (.tensor s) α
  | map      : (f : α → β) → TExpr s α → TExpr s β
  | zip      : (f : α → β → γ) → TExpr s α → TExpr s β → TExpr s γ
  | reduce   : (f : α → α → α) → (init : α) → (axis : Fin n) →
                TExpr (.tensor (dims)) α → TExpr (.tensor (remove axis dims)) α
  | reshape  : (new_shape : List Nat) → (proof : product old = product new_shape) →
                TExpr (.tensor old) α → TExpr (.tensor new_shape) α
  | gather   : (indices : TExpr idx_shape Nat) → (axis : Fin n) →
                TExpr (.tensor dims) α → TExpr (.tensor new_dims) α
  | ...

-- Denotational semantics: maps TExpr to a mathematical function
def denote : TExpr s α → (Index s → α)
  | .const v         => fun _ => v
  | .tensor f        => f
  | .map g e         => fun idx => g (denote e idx)
  | .zip g e1 e2     => fun idx => g (denote e1 idx) (denote e2 idx)
  | .reduce f init ax e => fun idx =>
      Finset.fold f init (Finset.range (dims.get ax))
        (fun k => denote e (insert_index ax k idx))
  | ...
```

Common operations are defined as compositions of primitives:

```lean
def matmul (A : TExpr [m, k] Float) (B : TExpr [k, n] Float) : TExpr [m, n] Float :=
  reduce (· + ·) 0.0 (axis := 2) (zip (· * ·) (expand_dims A 2 n) (expand_dims B 0 m))

def conv2d (input : TExpr [b, ci, h, w] Float)
           (kernel : TExpr [co, ci, kh, kw] Float) : TExpr [b, co, oh, ow] Float :=
  -- defined via im2col + matmul, or directly as nested reductions
  ...

def softmax (x : TExpr [n] Float) : TExpr [n] Float :=
  let shifted := map (· - reduce max (-∞) 0 x) x
  let exps := map Real.exp shifted
  let sum := reduce (· + ·) 0.0 0 exps
  map (· / sum) exps
```

### 4.3 Tactic Categories and Specifications

#### Category 1: Loop Transformations

These operate on the loop nests implied by the IR's tensor dimensions.

| Tactic | Parameters | Preconditions | Proof Obligation |
|--------|-----------|---------------|------------------|
| `tile` | axis, tile_size | tile_size > 0 | Sum decomposition: splitting iteration into blocks preserves total |
| `split` | axis, factor | factor > 0 | Same as tile (tile is split on one axis) |
| `fuse` | axis1, axis2 | Axes are adjacent, independent | Product of ranges equals fused range |
| `reorder` | perm : List Fin | Valid permutation | Commutativity: reordering independent iterations preserves result |
| `unroll` | axis, factor | factor divides extent (or handle remainder) | Loop body replication equals sequential execution |
| `skew` | axis1, axis2, factor | Affine dependence allows it | Affine index bijection preserves coverage |

**Example proof sketch for `tile`:**

```lean
theorem tile_correct (e : LoopNest) (ax : Fin e.depth) (ts : Nat) (hts : ts > 0) :
    denote (tile ax ts e) = denote e := by
  -- The key insight: for any summation Σ_{i=0}^{N-1} f(i),
  -- tiling gives Σ_{t=0}^{⌈N/ts⌉-1} Σ_{k=0}^{min(ts, N-t*ts)-1} f(t*ts + k)
  -- These are equal by Finset.sum_bUnion (partition into blocks)
  ext idx
  simp [denote, tile]
  rw [Finset.sum_partition_blocks ts hts]
  -- Finset.sum_partition_blocks is a library lemma we prove once:
  -- splitting a sum into contiguous blocks preserves the total
```

#### Category 2: Memory Transformations

| Tactic | Parameters | Preconditions | Proof Obligation |
|--------|-----------|---------------|------------------|
| `cache_read` | buffer, scope (shared/register) | Buffer is read-only in scope | Reading from cache = reading from original |
| `cache_write` | buffer, scope | Buffer is written only in scope | Write-back produces same final state |
| `layout_transform` | buffer, index_map | index_map is bijective | Bijective remapping preserves all values |
| `pack` | buffer, tile_dims | Tile dims divide buffer dims (or pad) | Packed layout contains same data |

**Proof strategy for `cache_read`:**

The proof shows that (a) the cache is loaded with exactly the values needed, (b) reads from the cache return the same values as reads from the original buffer. This reduces to showing that the index mapping from cache coordinates to original coordinates is correct — essentially proving an array copy preserves values.

#### Category 3: Parallelism Transformations

| Tactic | Parameters | Preconditions | Proof Obligation |
|--------|-----------|---------------|------------------|
| `parallel` | axis | No loop-carried dependencies on axis | Independence: iterations don't interfere |
| `bind_gpu_block` | axis | No dependencies (same as parallel) | Same as parallel + mapping to hardware |
| `bind_gpu_thread` | axis | No dependencies | Same as parallel |
| `parallel_reduce` | axis, combiner | Combiner is associative + commutative | Associativity allows arbitrary grouping |

**Critical precondition: dependency analysis.** The `parallel` tactic requires proving that iterations along the given axis are independent. This is a dependence analysis problem. For affine loop nests, dependence analysis is decidable (using integer linear programming / Presburger arithmetic) and can be automated. Our Lean4 implementation will include:

```lean
-- A tactic that automatically checks independence
-- using a decision procedure for Presburger arithmetic
tactic check_independence (ax : Fin depth) : Bool :=
  -- For each pair of statements s1, s2 that access the same buffer,
  -- check: ∀ i1 ≠ i2 along ax, access(s1, i1) ≠ access(s2, i2)
  -- OR one of them is a read
  decide_presburger (build_independence_formula ax statements)
```

#### Category 4: Numerical/Approximate Transformations

These are unique to ML compilers — they introduce bounded error.

| Tactic | Parameters | Preconditions | Proof Obligation |
|--------|-----------|---------------|------------------|
| `quantize` | dtype_from, dtype_to, scale | scale > 0 | Error bound: |orig - quantized| ≤ ε(scale, dtype) |
| `use_tensor_core` | matmul op, format | Dimensions satisfy hardware constraints | ε-equivalence for TF32/FP16 accumulation |
| `fast_math` | op, approximation | Op supports the approximation | Specified error bound holds |
| `mixed_precision` | regions, dtypes | Type compatibility | Composed error bound for the graph |

**ε-equivalence framework:**

```lean
-- Exact transformations produce equal results
def ExactEquiv (e1 e2 : TExpr s α) : Prop :=
  ∀ idx, denote e1 idx = denote e2 idx

-- Approximate transformations produce bounded-error results
def ApproxEquiv (ε : Float) (e1 e2 : TExpr s Float) : Prop :=
  ∀ idx, |denote e1 idx - denote e2 idx| ≤ ε

-- Error composition theorem (proved once)
theorem approx_compose (h1 : ApproxEquiv ε₁ e1 e2) (h2 : ApproxEquiv ε₂ e2 e3)
    (hL : ∀ idx, |denote e2 idx| ≤ M) :
    ApproxEquiv (ε₁ + ε₂ + ε₁ * ε₂ / M) e1 e3 := by
  -- Triangle inequality + error propagation
  ...
```

### 4.4 Tactic Composition Language

The AI optimizer speaks a simple composition language:

```
optimize matmul_1024 for NVIDIA_A100:
  tile [i, j] by [128, 128]       -- tile output dimensions
  tile [k] by [32]                  -- tile reduction dimension
  reorder [i_outer, j_outer, k_outer, i_inner, k_inner, j_inner]
  cache_read A to shared            -- A tile to shared memory
  cache_read B to shared            -- B tile to shared memory
  bind_gpu_block [i_outer, j_outer] -- map outer tiles to GPU blocks
  bind_gpu_thread [i_inner]         -- map inner tile to threads
  vectorize j_inner by 4            -- vectorize innermost
  use_tensor_core format=tf32       -- use tensor cores (approx)
  unroll k_inner                    -- fully unroll inner reduction
```

The system processes this as follows:

1. **Parse** the sequence into a list of tactic applications.
2. **Check preconditions** sequentially. If `tile [i, j] by [128, 128]` is valid, apply it, yielding a new IR state. Then check `reorder [...]` against the new state.
3. **If any precondition fails**, reject the sequence and report which tactic failed and why.
4. **If all preconditions pass**, the final IR is provably equivalent to the original (or ε-equivalent, with a computed error bound).

---

## 5. The AI Search Agent

### 5.1 Interface

The AI agent receives:
- The Semantic IR graph
- The target hardware description (GPU model, memory hierarchy, compute units)
- The tactic library (available tactics with their parameter spaces)
- A cost model (learned or analytical)

It produces:
- A sequence of tactic applications (the "schedule")
- The sequence is verified by the tactic composition engine

### 5.2 Search Strategies

**Strategy 1: LLM-based proposal.** Use a large language model (fine-tuned on successful schedules) to propose complete tactic sequences. The verifier accepts or rejects. Failed sequences provide negative training signal.

**Strategy 2: RL with tactic actions.** Train an RL agent where:
- **State** = current IR after partial tactic application
- **Action** = next tactic with parameters
- **Reward** = estimated performance improvement (from cost model or profiling)
- **Invalid actions** are filtered by preconditions — the agent never wastes time on incorrect transformations

**Strategy 3: Equality saturation + extraction.** Represent the space of equivalent programs as an e-graph (using techniques from TENSAT/egg), where rewrite rules correspond to verified tactics. Use extraction with a cost function to select the optimal equivalent program.

**Strategy 4: Hybrid.** Use the LLM to propose a coarse schedule (tile sizes, parallelization strategy), then use RL or equality saturation to refine details (unroll factors, cache placement, vectorization widths).

### 5.3 Training the AI

The verified tactic system enables a uniquely clean training loop:

```
for episode in training:
    program = sample_program()           # e.g., matmul, conv, attention
    target = sample_hardware()           # e.g., A100, H100, MI300X

    # Agent proposes tactics
    schedule = agent.propose(program, target)

    # Verifier checks (instant, no compilation needed for checking)
    if not verifier.check_preconditions(program, schedule):
        reward = INVALID_PENALTY
        agent.update(reward, schedule)
        continue

    # Apply verified schedule, compile, benchmark
    optimized_ir = apply_tactics(program, schedule)
    code = codegen(optimized_ir, target)
    perf = benchmark(code, target)

    reward = perf / baseline_perf
    agent.update(reward, schedule)
```

Key advantage: **zero time wasted on incorrect programs**. In conventional AI kernel generation (GEAK, TritonRL, etc.), a large fraction of generated kernels fail compilation or produce wrong results. With verified tactics, every proposed schedule that passes precondition checking produces a correct kernel. The agent focuses purely on performance.

---

## 6. Implementation Plan

### Phase 1: Foundation (Months 1-6)

**Deliverable:** Core IR + 8 verified tactics + a simple search agent

1. Define the Semantic IR in Lean 4, covering: elementwise ops, reductions, matmul (as reduce), reshape, transpose, gather/scatter.

2. Implement and verify 8 core tactics in Lean 4:
   - `tile` (with remainder handling)
   - `split`
   - `fuse`
   - `reorder`
   - `unroll`
   - `vectorize`
   - `parallel` (with affine dependence checker)
   - `cache_read`

3. Build the tactic composition engine in Lean 4 / Rust that checks preconditions and applies tactics sequentially.

4. Build a simple code generator that lowers the scheduled IR to C / CUDA (unverified initially, but tested extensively).

5. Implement a brute-force search agent that enumerates small tactic sequences and benchmarks them.

**Target:** Correctly optimize a dense matmul to within 70% of cuBLAS on a single GPU.

### Phase 2: AI Agent & More Tactics (Months 7-12)

**Deliverable:** RL-based search agent + 15 additional tactics + GPU support

6. Add memory transformation tactics: `cache_write`, `layout_transform`, `pack`, `set_scope (register/shared/global)`.

7. Add GPU-specific tactics: `bind_gpu_block`, `bind_gpu_thread`, `use_tensor_core`.

8. Add numerical tactics: `quantize`, `mixed_precision` with ε-tracking.

9. Train an RL agent using PPO/GRPO on a curriculum of operators: matmul → conv2d → elementwise fusion → attention.

10. Build a learned cost model from profiling data (supplement or replace analytical model).

**Target:** Match or exceed TVM autotuning performance on a standard operator benchmark (matmul, conv2d, depthwise conv, batch norm, softmax) across NVIDIA A100/H100.

### Phase 3: Graph-Level Optimization (Months 13-18)

**Deliverable:** Whole-graph optimization for real models

11. Extend the Semantic IR to represent computation graphs (multiple ops with data dependencies).

12. Add graph-level tactics: `fuse_ops` (operator fusion), `recompute` (trade memory for compute), `pipeline` (overlap stages).

13. Integrate equality saturation (egg/egglog) for graph-level rewrite exploration, with verified rewrite rules.

14. Optimize full models: ResNet-50, BERT, GPT-2 attention blocks.

**Target:** End-to-end inference performance competitive with TensorRT on standard models.

### Phase 4: Verified Code Generation & Ecosystem (Months 19-24)

15. Verify the code generator (CompCert-style) from scheduled IR to LLVM IR or PTX, for at least the core loop transformations.

16. Integrate with PyTorch/JAX via graph tracing — users write normal Python and the VeriTac compiler optimizes the traced graph.

17. Open-source the tactic library and encourage community contributions of new verified tactics.

18. Use LLMs to help *generate proofs* for new tactics: given a proposed transformation and its specification, the LLM suggests Lean 4 proof steps, which are then machine-checked.

---

## 7. Key Research Challenges

### 7.1 Proof Automation

The biggest bottleneck is writing the proofs. For each tactic, we need a Lean 4 proof that it preserves semantics. Some strategies to manage this:

- **Domain-specific automation.** Build Lean 4 tactics (meta-level) that automatically discharge common proof obligations: sum decomposition for tiling, commutativity for reordering, bijection checking for layout transforms. The Halide verification work showed that Z3 can automatically verify 88% of rewrite rules; we aim for similar automation rates.

- **Proof templates.** Many tactics follow similar patterns. A tiling proof for matmul looks structurally identical to a tiling proof for conv2d. Parameterize proof templates over the operation being tiled.

- **AI-assisted proving.** Use LLMs (trained on Lean proofs from Mathlib) to suggest proof steps. The human/AI writes the proof sketch; Lean checks it. This creates a virtuous cycle where AI helps verify AI-generated optimizations.

### 7.2 Floating Point

All our proofs over reals must account for floating-point behavior in practice. Three approaches:

1. **Abstract over FP.** Prove correctness over abstract ordered fields, leaving FP as a model. This gives correctness up to FP semantics — the same guarantee CompCert provides.

2. **Use Flocq-style FP formalization.** The Flocq library (used by CompCert) provides Coq-verified IEEE 754 arithmetic. Porting key results to Lean 4 would enable FP-precise proofs.

3. **Pragmatic:** Prove structural transformations (tiling, reordering, fusion) over abstract arithmetic. Only use FP-specific reasoning for numerical tactics (quantize, fast_math). Structural transforms don't change the arithmetic — they change loop structure — so FP precision is preserved automatically.

Approach 3 is recommended as the starting point.

### 7.3 Error Budget Tracking

For approximate transformations, errors compose nonlinearly. We need to track error propagation through the graph. Key insight: most ML workloads have a natural "error budget" (training noise, quantization noise). We can express this as:

```lean
structure ErrorBudget where
  per_op_bound : Float          -- max error introduced per op
  total_bound : Float           -- max accumulated error at output
  composition_rule : ...        -- how errors compose through the graph
```

The AI agent learns to stay within the error budget while maximizing performance.

### 7.4 Expressiveness vs. Decidability

The tactic precondition language must be:
- **Expressive enough** to capture real optimization preconditions (dependency analysis, divisibility, alignment).
- **Decidable enough** that precondition checking is automatic and fast.

For affine loop nests (which cover ~90% of ML workloads), Presburger arithmetic is both expressive and decidable. For non-affine cases (data-dependent indexing, dynamic shapes), we fall back to runtime checks or conservative approximations.

---

## 8. Comparison with Alternatives

| | Traditional (MLIR/TVM) | Translation Validation | VeriTac (This Proposal) |
|---|---|---|---|
| **When is correctness checked?** | Testing only | After each compilation | Before deployment (once per tactic) |
| **What is trusted?** | Every pass author | The validator | The Lean 4 kernel (~10k loc) |
| **Can AI generate optimizations?** | Yes, but may be wrong | Yes, validated after | Yes, correct by construction |
| **Overhead** | None | Per-compilation validation | Per-tactic verification (one-time) |
| **Handles new ops** | New pass needed | New validation rules | New tactic + proof needed |
| **Approximate transforms** | Ad hoc | Hard to validate | ε-tracking built in |

---

## 9. Conclusion

VeriTac represents a convergence of three trends: the maturation of dependent type theory (Lean 4), the explosion of AI-driven code optimization (GEAK, TritonRL, etc.), and the growing need for correctness guarantees as ML systems become safety-critical. By encoding optimization knowledge as a verified tactic library, we create a system where the AI is free to search aggressively — every valid tactic sequence produces correct code. The proof burden is finite, one-time, and amortizable across all future compilations.

The key bet is that a small set (~30) of verified tactics, composed creatively by AI, can match or exceed the performance of thousands of unverified hand-written passes. The prior work on Halide (algorithm/schedule separation), TVM (schedule primitives), CompCert (verified compilation), egg (equality saturation), and recent AI kernel generation suggests this bet is well-founded.
