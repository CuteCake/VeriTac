/-
  VeriTac.Gemmini.Plan
  Lean-checked plan legality for Gemmini GEMM workloads.

  A plan names a GEMM workload (m x n output, k reduction), the systolic array
  dimension `dim`, the on-chip row capacities, and one of two schedules:

  - `baseline`: each m x n x k GEMM is executed as a sequence of dim x dim x dim
    mesh operations; the B tile is re-streamed for every output tile row, so
    the accumulator must hold one full output row block of `dim` rows.
  - `reuse_b`: the B tile is reused across output-row blocks while the
    accumulator retains `m` rows of partial outputs for one output-column tile.

  Both schedules stage an A tile and a B tile of `dim` rows each in the
  scratchpad, i.e. `2 * dim` scratchpad rows.

  Numerical contract: signed int8 inputs, exact int32 accumulation, no scaling.
  Each int8 product has magnitude at most `2^7 * 2^7 = 16384`, so a k-term dot
  product has magnitude at most `k * 16384`; the plan is legal only if that
  worst case still fits in a signed int32 (`≤ 2^31 - 1 = 2147483647`), which
  makes overflow impossible for *any* int8 inputs.

  Proof boundary: plan legality alone does NOT establish the correctness of any
  emitted C code or Gemmini instruction stream. It is a necessary static
  precondition on workload shape and on-chip capacities.
-/

namespace VeriTac.Gemmini

/-! ## Plan types -/

/-- Schedule variant of a Gemmini GEMM plan. -/
inductive GemminiSchedule where
  | baseline
  | reuseB
  deriving Repr, DecidableEq, Inhabited

/-- A Gemmini GEMM plan: workload dimensions, on-chip capacities, schedule. -/
structure GemminiPlan where
  /-- Output rows. -/
  m : Nat
  /-- Output columns. -/
  n : Nat
  /-- Reduction depth. -/
  k : Nat
  /-- Systolic array dimension (dim x dim mesh). -/
  dim : Nat
  /-- Scratchpad rows available. -/
  scratchpadRows : Nat
  /-- Accumulator rows available. -/
  accumulatorRows : Nat
  /-- Schedule variant. -/
  schedule : GemminiSchedule
  deriving Repr, DecidableEq, Inhabited

/-! ## Numerical constants -/

/-- Max magnitude of one signed-int8 product: `2^7 * 2^7`. -/
def int8ProductBound : Nat := 16384

/-- Max signed int32 value: `2^31 - 1`. -/
def int32Max : Nat := 2147483647

/-! ## Resource requirements -/

/-- Scratchpad rows a plan must have: both schedules stage one `dim`-row A tile
    and one `dim`-row B tile. -/
def requiredScratchpadRows (p : GemminiPlan) : Nat := 2 * p.dim

/-- Accumulator rows a plan must have: `baseline` holds one `dim`-row output
    block; `reuse_b` retains partial outputs for `m` output rows. -/
def requiredAccumulatorRows (p : GemminiPlan) : Nat :=
  match p.schedule with
  | .baseline => p.dim
  | .reuseB => p.m

/-! ## Legality -/

/-- The full legality boundary for a Gemmini GEMM plan.  All fields are
    structural `Nat`s (JSON parsing must reject negatives, nonintegers and
    missing fields before constructing the plan).  Positivity of `m n k dim`,
    divisibility of the workload by the mesh dimension, the int32 overflow
    bound, and the per-schedule on-chip capacity requirements are all part of
    the boundary.  An `abbrev` so `decide` unfolds it into a decidable
    conjunction. -/
abbrev planLegalProps (p : GemminiPlan) : Prop :=
  p.m > 0 ∧ p.n > 0 ∧ p.k > 0 ∧ p.dim > 0
    ∧ p.m % p.dim = 0 ∧ p.n % p.dim = 0 ∧ p.k % p.dim = 0
    ∧ p.k * int8ProductBound ≤ int32Max
    ∧ p.scratchpadRows ≥ requiredScratchpadRows p
    ∧ p.accumulatorRows ≥ requiredAccumulatorRows p

/-- Boolean checker deciding `planLegalProps`. -/
def checkGemminiPlan (p : GemminiPlan) : Bool :=
  decide (planLegalProps p)

/-- Soundness: an accepted plan satisfies every legality clause. -/
theorem checkGemminiPlan_sound (p : GemminiPlan) (h : checkGemminiPlan p = true) :
    planLegalProps p := of_decide_eq_true h

/-- Completeness: every legal plan is accepted. -/
theorem checkGemminiPlan_complete (p : GemminiPlan) (h : planLegalProps p) :
    checkGemminiPlan p = true := decide_eq_true h

/-- The checker is exactly the legality boundary. -/
theorem checkGemminiPlan_iff (p : GemminiPlan) :
    checkGemminiPlan p = true ↔ planLegalProps p :=
  ⟨checkGemminiPlan_sound p, checkGemminiPlan_complete p⟩

/-- Explicit overflow statement: a dot product of `k` signed-int8 products whose
    magnitude sum is bounded by `k * 16384` fits in a signed int32 whenever the
    plan's bound holds. -/
theorem dotProduct_fits_int32 (k s : Nat)
    (hbound : k * int8ProductBound ≤ int32Max) (hsum : s ≤ k * int8ProductBound) :
    s ≤ int32Max := Nat.le_trans hsum hbound

/-- Under plan legality, the plan's reduction depth satisfies the int32 bound. -/
theorem legalPlan_bound (p : GemminiPlan) (h : planLegalProps p) :
    p.k * int8ProductBound ≤ int32Max := h.2.2.2.2.2.2.2.1

/-! ## Deterministic rejection diagnostics -/

/-- Human-readable rejection reason for a plan.  The first violated clause
    wins; the ordering is fixed so the output is stable.  An accepted plan
    always reports `"accepted"`. -/
def planDiagnostic (p : GemminiPlan) : String :=
  if checkGemminiPlan p then "accepted"
  else if p.m = 0 then "m must be positive"
  else if p.n = 0 then "n must be positive"
  else if p.k = 0 then "k must be positive"
  else if p.dim = 0 then "dim must be positive"
  else if p.m % p.dim != 0 then "m must be divisible by dim"
  else if p.n % p.dim != 0 then "n must be divisible by dim"
  else if p.k % p.dim != 0 then "k must be divisible by dim"
  else if p.k * int8ProductBound > int32Max then
    "k * 16384 exceeds the int32 accumulation bound"
  else if p.scratchpadRows < requiredScratchpadRows p then
    "scratchpad rows below 2 * dim"
  else if p.accumulatorRows < requiredAccumulatorRows p then
    "accumulator rows below the schedule requirement"
  else "rejected"

/-- Accepted plans report `"accepted"`. -/
theorem planDiagnostic_accepted (p : GemminiPlan) (h : checkGemminiPlan p = true) :
    planDiagnostic p = "accepted" := if_pos h

end VeriTac.Gemmini
