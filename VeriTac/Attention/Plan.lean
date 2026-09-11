/-
  VeriTac.Attention.Plan
  A Lean-checked tiling resource plan for the installed Metal FP32 fused
  attention shader, together with the per-row partition legality that any
  tiling must satisfy.

  The shader runs on Metal with SIMD-32 SIMD groups: one SIMD group covers every
  8 query rows, so a query tile of `q` uses `q / 8` SIMD groups and
  `(q / 8) * 32` threads.  The measured FP32 threadgroup-memory layouts are

      Q bytes = queryTile * (headDim + 4) * 4
      K bytes = (keyTile + 4) * headDim * 4
      V bytes = keyTile * (headDim + 4) * 4

  and the simultaneously-live bytes are `Q + (K ∨ V)` when K and V share
  storage.  `aliasKV = false` is a *resource-legal planning* state that stages Q,
  K and V separately (`Q + K + V`); only `aliasKV = true` is executable by the
  final runner (`executable p`), because the installed vendor shader always
  aliases K/V into the same storage.  `reuse_kv_storage` turns a planning state
  into an executable one.

  `checkPlan` decides the full `PlanLegal` boundary (resources plus, crucially,
  that every causal row prefix `1..seqLen` has a *legally checked* key-tile
  range partition).  The algebraic content is not re-proved here: `Partition.lean`
  supplies `merged_equals_direct`, and we derive per-row Real semantic
  preservation for any accepted plan.

  This file makes no claim about floating-point accuracy or about any compiler
  / backend refinement; it reasons about the abstract Real semantics and the
  measured resource accounting only.
-/
import VeriTac.Hardware.Target
import VeriTac.Attention.Partition

open VeriTac.Hardware

namespace VeriTac.Attention

/-! ## Tiling resource plan -/

/-- SIMD group count for a query tile: the shader assigns one SIMD group to every
    8 query rows. -/
def simdGroups (queryTile : Nat) : Nat := queryTile / 8

/-- Thread count under the Metal SIMD-32 layout. -/
def planThreads (queryTile : Nat) : Nat := simdGroups queryTile * 32

/-- FP32 threadgroup-memory bytes for the staged Q tile. -/
def qBytes (queryTile headDim : Nat) : Nat := queryTile * (headDim + 4) * 4

/-- FP32 threadgroup-memory bytes for the staged K tile. -/
def kBytes (keyTile headDim : Nat) : Nat := (keyTile + 4) * headDim * 4

/-- FP32 threadgroup-memory bytes for the staged V tile. -/
def vBytes (keyTile headDim : Nat) : Nat := keyTile * (headDim + 4) * 4

/-- Simultaneously live threadgroup memory.  Q always lives; when `aliasKV` is
    true the vendor shader reuses one storage for K and V (only the larger is
    live), otherwise both are live. -/
def planSharedBytes (queryTile keyTile headDim : Nat) (aliasKV : Bool) : Nat :=
  qBytes queryTile headDim +
    (if aliasKV then max (kBytes keyTile headDim) (vBytes keyTile headDim)
     else kBytes keyTile headDim + vBytes keyTile headDim)

/-- A tiled attention plan for the installed FP32 fused Metal shader. -/
structure TiledAttentionPlan where
  seqLen : Nat
  headDim : Nat
  queryTile : Nat
  keyTile : Nat
  aliasKV : Bool
  threadsPerThreadgroup : Nat
  deriving Repr

/-- Simultaneously live threadgroup-memory bytes of a plan. -/
def sharedBytes (p : TiledAttentionPlan) : Nat :=
  planSharedBytes p.queryTile p.keyTile p.headDim p.aliasKV

/-- The plan is runnable by the installed vendor shader only when K and V share
    storage (`aliasKV = true`).  An `aliasKV = false` plan is a resource-legal
    *planning* state (stages K and V separately, so uses `Q + K + V` bytes); it
    is not executable and must never be dispatched. -/
def executable (p : TiledAttentionPlan) : Prop := p.aliasKV = true

/-! ## Causal key-tile range partition for a row prefix -/

/-- Number of `tile`-wide tiles covering `prefix` key indices (`ceil (prefix/tile)`). -/
def numTiles (pre tile : Nat) : Nat := (pre + tile - 1) / tile

/-- The contiguous range partition of a causal row prefix `[0, prefix)` into
    `tile`-wide tiles: `[j*tile, min prefix ((j+1)*tile))` for `j` in
    `0 .. numTiles-1`.  The final tile may be an irregular (short) tail. -/
def partitionRanges (pre tile : Nat) : List Range :=
  (List.range (numTiles pre tile)).map (fun j => (j * tile, min pre ((j + 1) * tile)))

/-- The key-tile range partition used for the `row`-th causal row of a plan. -/
def rowPartition (p : TiledAttentionPlan) (row : Nat) : List Range :=
  partitionRanges row p.keyTile

/-- The checker enumerates every row prefix `1..seqLen` (via the bounded index
    `r ∈ 0 .. seqLen-1`, giving `r+1 ∈ 1..seqLen`) and verifies that the
    key-tile range partition of that prefix is a legal causal partition. -/
def plansLegalPartitions (p : TiledAttentionPlan) : Bool :=
  (List.range p.seqLen).all (fun r => legalCausalPartition (r + 1) (partitionRanges (r + 1) p.keyTile))

/-! ## Plan legality -/

/-- The full legality boundary for a tiling plan on a given target.  Besides the
    resource constraints (backend, SIMD width, threads, shared memory, supported
    dimensions) it commits to the decidable per-row partition legality
    `partitions_legal`.  `aliasKV` is deliberately NOT required to be `true`:
    an `aliasKV = false` plan is resource-legal (it stages `Q + K + V`) but is
    not executable (`executable` is a separate predicate the dispatcher must
    honor). -/
structure PlanLegal (t : Target) (p : TiledAttentionPlan) : Prop where
  backend_metal : t.backend = "metal"
  target_limits_positive : t.maxThreadsPerThreadgroup > 0 ∧ t.maxThreadgroupMemoryBytes > 0
  seq_len_positive : p.seqLen > 0
  head_supported : p.headDim = 192 ∨ p.headDim = 256
  query_tile_supported : p.queryTile = 8 ∨ p.queryTile = 16 ∨ p.queryTile = 24 ∨ p.queryTile = 32
  key_tile_supported : p.keyTile = 8 ∨ p.keyTile = 16 ∨ p.keyTile = 32
  simd_width : t.simdWidth = 32
  threads : p.threadsPerThreadgroup = planThreads p.queryTile
  threads_within : p.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup
  shared_within : sharedBytes p ≤ t.maxThreadgroupMemoryBytes
  partitions_legal : plansLegalPartitions p = true

/-- `PlanLegal` is decidable: every field is a decidable proposition, so the
    checker can decide it at runtime. -/
instance instDecidablePlanLegal (t : Target) (p : TiledAttentionPlan) :
    Decidable (PlanLegal t p) :=
  if h1 : t.backend = "metal" then
    if h2 : t.maxThreadsPerThreadgroup > 0 ∧ t.maxThreadgroupMemoryBytes > 0 then
      if h3 : p.seqLen > 0 then
        if h4 : p.headDim = 192 ∨ p.headDim = 256 then
          if h5 : p.queryTile = 8 ∨ p.queryTile = 16 ∨ p.queryTile = 24 ∨ p.queryTile = 32 then
            if h6 : p.keyTile = 8 ∨ p.keyTile = 16 ∨ p.keyTile = 32 then
              if h7 : t.simdWidth = 32 then
                if h8 : p.threadsPerThreadgroup = planThreads p.queryTile then
                  if h9 : p.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup then
                    if h10 : sharedBytes p ≤ t.maxThreadgroupMemoryBytes then
                      if h11 : plansLegalPartitions p = true then
                        isTrue ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩
                      else isFalse (fun f => h11 f.partitions_legal)
                    else isFalse (fun f => h10 f.shared_within)
                  else isFalse (fun f => h9 f.threads_within)
                else isFalse (fun f => h8 f.threads)
              else isFalse (fun f => h7 f.simd_width)
            else isFalse (fun f => h6 f.key_tile_supported)
          else isFalse (fun f => h5 f.query_tile_supported)
        else isFalse (fun f => h4 f.head_supported)
      else isFalse (fun f => h3 f.seq_len_positive)
    else isFalse (fun f => h2 f.target_limits_positive)
  else isFalse (fun f => h1 f.backend_metal)

/-- Boolean acceptance predicate for a tiling plan: decides `PlanLegal`. -/
def checkPlan (t : Target) (p : TiledAttentionPlan) : Bool :=
  decide (PlanLegal t p)

/-- A successful check witnesses the full `PlanLegal` boundary. -/
theorem checkPlan_legal (t : Target) (p : TiledAttentionPlan)
    (h : checkPlan t p = true) : PlanLegal t p :=
  of_decide_eq_true h

/-- Checker success implies the thread count equals the SIMD-derived value. -/
theorem checkPlan_threads_derived (t : Target) (p : TiledAttentionPlan)
    (h : checkPlan t p = true) :
    p.threadsPerThreadgroup = planThreads p.queryTile := by
  exact (checkPlan_legal t p h).threads

/-- Checker success implies the thread count fits the target limit. -/
theorem checkPlan_threads_within (t : Target) (p : TiledAttentionPlan)
    (h : checkPlan t p = true) :
    p.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup := by
  exact (checkPlan_legal t p h).threads_within

/-- Checker success implies the exact requested threadgroup-memory bytes fit the
    target limit. -/
theorem checkPlan_shared_within (t : Target) (p : TiledAttentionPlan)
    (h : checkPlan t p = true) :
    sharedBytes p ≤ t.maxThreadgroupMemoryBytes := by
  exact (checkPlan_legal t p h).shared_within

/-! Checker success does not require `aliasKV = true`: an `aliasKV = false`
    plan is resource-legal.  Executability is a separate predicate that the
    dispatcher must honor before benchmarking (see `executable`). -/

/-- Checker success implies every row prefix has a legally checked key-tile
    partition (the `List.all` soundness theorem: a true `all` holds pointwise
    over the enumerated range `0 .. seqLen-1`). -/
theorem plan_partitions_legal (p : TiledAttentionPlan) (h : plansLegalPartitions p = true) :
    ∀ r, r < p.seqLen →
      legalCausalPartition (r + 1) (partitionRanges (r + 1) p.keyTile) = true := by
  intro r hr
  have hmem : r ∈ List.range p.seqLen := List.mem_range.mpr hr
  exact (List.all_eq_true.mp h) r hmem

/-- Checker success implies each row prefix has a legal key-tile partition. -/
theorem checkPlan_row_partition_legal (t : Target) (p : TiledAttentionPlan)
    (h : checkPlan t p = true) (r : Nat) (hr : r < p.seqLen) :
    legalCausalPartition (r + 1) (partitionRanges (r + 1) p.keyTile) = true :=
  plan_partitions_legal p (checkPlan_legal t p h).partitions_legal r hr

/-! ## Resource budget theorems -/

/-- Under K/V aliasing the simultaneously-live bytes are
    `Q + max(K, V) = max(Q + K, Q + V)`, so a budget that fits `Q + max(K, V)`
    fits both `Q + K` and `Q + V` individually. -/
theorem alias_budget_both (q k v budget : Nat) (h : q + max k v ≤ budget) :
    q + k ≤ budget ∧ q + v ≤ budget := by
  constructor
  · exact le_trans (Nat.add_le_add_left (Nat.le_max_left k v) q) h
  · exact le_trans (Nat.add_le_add_left (Nat.le_max_right k v) q) h

/-- `a + max b c ≤ a + (b + c)`: aliasing two staged buffers (taking only the
    larger) never needs more bytes than keeping both. -/
theorem add_max_le_add (a b c : Nat) : a + max b c ≤ a + (b + c) := by
  apply Nat.add_le_add_left
  exact max_le (Nat.le_add_right b c) (Nat.le_add_left c b)

/-- Aliasing K/V storage never increases the simultaneously-live shared-memory
    bytes: `Q + max(K, V) ≤ Q + K + V`.  So switching a plan to `aliasKV = true`
    (`reuse_kv_storage`) can only reduce (or keep) the measured footprint. -/
theorem alias_shared_le_noalias (p : TiledAttentionPlan) :
    sharedBytes { p with aliasKV := true } ≤ sharedBytes { p with aliasKV := false } := by
  unfold sharedBytes planSharedBytes
  simp

/-- If an executable (`aliasKV = true`) plan fits its shared-memory budget, then
    both `Q + K` and `Q + V` individually fit the same budget. -/
theorem plan_alias_budget_both (p : TiledAttentionPlan) (budget : Nat)
    (halias : p.aliasKV = true) (h : sharedBytes p ≤ budget) :
    qBytes p.queryTile p.headDim + kBytes p.keyTile p.headDim ≤ budget ∧
    qBytes p.queryTile p.headDim + vBytes p.keyTile p.headDim ≤ budget := by
  rw [sharedBytes, planSharedBytes] at h
  simp [halias] at h
  exact alias_budget_both (qBytes p.queryTile p.headDim)
    (kBytes p.keyTile p.headDim) (vBytes p.keyTile p.headDim) budget h

/-! ## Plan tactics -/

/-- A refinement tactic for a tiling plan: set the query tile (which also
    recomputes the thread count), set the key tile, or switch to K/V storage
    reuse (making the plan executable). -/
inductive PlanTactic where
  | setQueryTile (q : Nat)
  | setKeyTile (k : Nat)
  | reuseKVStorage

/-- Apply a single tactic, recomputing the thread count whenever the query tile
    changes. -/
def applyPlanTactic (p : TiledAttentionPlan) (tac : PlanTactic) : TiledAttentionPlan :=
  match tac with
  | .setQueryTile q => { p with queryTile := q, threadsPerThreadgroup := planThreads q }
  | .setKeyTile k => { p with keyTile := k }
  | .reuseKVStorage => { p with aliasKV := true }

/-- `reuse_kv_storage` is a real transformation: it makes the plan executable by
    the vendor shader. -/
theorem reuse_makes_executable (p : TiledAttentionPlan) :
    executable (applyPlanTactic p .reuseKVStorage) := by
  unfold executable applyPlanTactic
  trivial

/-- Run a tactic sequence: every intermediate state must be legal, else `none`. -/
def checkPlanTacticsAux (t : Target) (p : TiledAttentionPlan) :
    List PlanTactic → Option TiledAttentionPlan
  | [] => some p
  | tac :: rest =>
      if checkPlan t (applyPlanTactic p tac) then
        checkPlanTacticsAux t (applyPlanTactic p tac) rest
      else none

/-- Run a tactic sequence against a target: the `initial` plan and every state
    after a tactic must be legal, else `none`; otherwise the final plan. -/
def checkPlanTactics (t : Target) (initial : TiledAttentionPlan)
    (tactics : List PlanTactic) : Option TiledAttentionPlan :=
  if checkPlan t initial then checkPlanTacticsAux t initial tactics else none

/-- Every state accepted along a plan tactic sequence is legal. -/
theorem checkPlanTacticsAux_legal (t : Target) (p : TiledAttentionPlan) :
    ∀ (tactics : List PlanTactic) (final : TiledAttentionPlan),
      PlanLegal t p → checkPlanTacticsAux t p tactics = some final → PlanLegal t final := by
  intro tactics
  induction tactics generalizing p with
  | nil =>
      intro final hl h
      simp [checkPlanTacticsAux] at h
      subst final
      exact hl
  | cons tac rest ih =>
      intro final hl h
      simp [checkPlanTacticsAux] at h
      cases hc : checkPlan t (applyPlanTactic p tac) with
      | false => simp [hc] at h
      | true =>
          simp [hc] at h
          have hl' : PlanLegal t (applyPlanTactic p tac) :=
            checkPlan_legal t (applyPlanTactic p tac) hc
          exact ih (applyPlanTactic p tac) final hl' h

/-- Correctness of `checkPlanTactics`: a returned final plan is legal. -/
theorem checkPlanTactics_legal (t : Target) (initial : TiledAttentionPlan)
    (tactics : List PlanTactic) (final : TiledAttentionPlan)
    (h : checkPlanTactics t initial tactics = some final) :
    PlanLegal t final := by
  simp [checkPlanTactics] at h
  cases hc : checkPlan t initial with
  | false => simp [hc] at h
  | true =>
      simp [hc] at h
      have hl : PlanLegal t initial := checkPlan_legal t initial hc
      exact checkPlanTacticsAux_legal t initial tactics final hl h

/-! ## Per-row Real semantic preservation -/

/-- For any legally partitioned causal row prefix, the merged (rescaled)
    stable-softmax summary over the key-tile ranges equals the direct summary
    over the whole prefix.  This is `merged_equals_direct` specialized to the
    tile range partition of a prefix. -/
theorem row_merged_equals_direct (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (pre tile : Nat)
    (hlegal : legalCausalPartition pre (partitionRanges pre tile) = true) :
    ((partitionRanges pre tile).map
        (fun pt => num value score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum /
      ((partitionRanges pre tile).map
        (fun pt => denom score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum
    = num value score M (Finset.range pre) / denom score M (Finset.range pre) :=
  merged_equals_direct value score pre (partitionRanges pre tile) m M hlegal

/-- Main objective: an accepted plan gives, for every causal row `r` (`r <
    seqLen`), that the tile-merged softmax output equals the direct softmax
    output over that row's key prefix, for arbitrary score / value / reference
    functions `m`.  This is the Real-algebraic semantic preservation; it makes
    no claim about floating-point behavior. -/
theorem accepted_plan_row_preserves (t : Target) (p : TiledAttentionPlan)
    (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (h : checkPlan t p = true) (r : Nat) (hr : r < p.seqLen) :
    ((partitionRanges (r + 1) p.keyTile).map
        (fun pt => num value score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum /
      ((partitionRanges (r + 1) p.keyTile).map
        (fun pt => denom score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum
    = num value score M (Finset.range (r + 1)) / denom score M (Finset.range (r + 1)) := by
  have hpart : legalCausalPartition (r + 1) (partitionRanges (r + 1) p.keyTile) = true :=
    checkPlan_row_partition_legal t p h r hr
  exact row_merged_equals_direct value score m M (r + 1) p.keyTile hpart

/-! ## Examples -/

/-- A concrete Metal target with a 32 KiB threadgroup-memory budget and SIMD-32
    SIMD width, matching the hardware schema used by the CLI. -/
def demoTarget : Target :=
  { backend := "metal", deviceName := "test-device", maxThreadsPerThreadgroup := 1024,
    maxThreadgroupMemoryBytes := 32768, simdWidth := 32, toolchain := "", provenance := "" }

/-- A default tiling (qt 8, kt 8, alias true) fits a 32 KiB threadgroup budget
    for `head_dim 256` and is accepted for a small positive sequence length. -/
example :
    checkPlan demoTarget { seqLen := 4, headDim := 256, queryTile := 8, keyTile := 8,
                           aliasKV := true, threadsPerThreadgroup := planThreads 8 } = true := by
  decide

/-- An `aliasKV = false` plan is *resource-legal* (it fits the budget by staging
    `Q + K + V`) and is accepted by the checker, but it is not executable. -/
example :
    checkPlan demoTarget { seqLen := 4, headDim := 256, queryTile := 8, keyTile := 8,
                           aliasKV := false, threadsPerThreadgroup := planThreads 8 } = true := by
  decide

/-- The same `aliasKV = false` plan is not executable by the vendor shader. -/
example :
    ¬ executable { seqLen := 4, headDim := 256, queryTile := 8, keyTile := 8,
                   aliasKV := false, threadsPerThreadgroup := planThreads 8 } := by
  unfold executable
  simp

/-- `reuse_kv_storage` turns the (resource-legal, non-executable) planning state
    into an executable one. -/
example :
    executable (applyPlanTactic { seqLen := 4, headDim := 256, queryTile := 8, keyTile := 8,
                                  aliasKV := false, threadsPerThreadgroup := planThreads 8 }
                                .reuseKVStorage) := by
  unfold executable applyPlanTactic
  trivial

/-- A query tile not divisible into SIMD-32 groups (e.g. 12) is rejected. -/
example :
    checkPlan demoTarget { seqLen := 4, headDim := 256, queryTile := 12, keyTile := 8,
                           aliasKV := true, threadsPerThreadgroup := planThreads 12 } = false := by
  decide

end VeriTac.Attention
