/-
  VeriTac.Attention.Scheduling
  Structural proofs for the creative CUDA scheduling / scratch-lifetime
  transformations used by the attention candidate kernels.

  This file is deliberately scoped.  It proves abstract facts only:

   1. Block scheduling over slots `Fin n` using `Fin.rev`: the forward /
      reverse schedules are permutations, so every logical block maps to a
      unique physical slot and vice versa (exactly-once coverage), and a
      block-indexed output is delivered intact to its explicit destination
      slot (value preservation under the bijection, with unchanged row
      ownership).  This models the all-four `query_start` index substitutions
      in the reversed derivative headers.

   2. Half-open scratch lifetimes over a discrete timeline: two time-disjoint
      phases share a buffer of capacity `max sizeA sizeB`, which fits either
      phase and never exceeds the sum of the two separate sizes; no time point
      is live in both phases.  The ordered MM1-then-epilogue example is
      instantiated.

   3. A connection to the accepted CUDA row-semantics theorem where reasonable,
      with the boundaries stated explicitly.

  Honest boundary.  We do NOT prove the GPU compiler executes blocks in any
  particular order, that it honors a `min_blocks_per_sm` launch hint, that
  `cp.async` hardware drains on command, or any floating-point behavior.  Those
  are runtime observations / controller validation, not theorems here.  There
  are no axioms, `sorry`, or `native_decide` in this file.
-/
import Mathlib.Data.Fin.Rev
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Interval
import Mathlib.Data.Finset.Card
import VeriTac.Attention.CudaPlan

namespace VeriTac.Attention.Scheduling

open Function

universe u

/-! ## 1. Forward / reverse block scheduling over slots `Fin n` -/

/-- A logical query-block index among `n` blocks. -/
abbrev BlockIndex (n : Nat) := Fin n

/-- Forward scheduling: the physical slot equals the logical block index
    (`blockIdx.x` in the installed kernel). -/
def forwardSlot (n : Nat) (b : BlockIndex n) : BlockIndex n := b

/-- Reverse scheduling: the physical slot is the reversed logical block index
    (`gridDim.x - 1 - blockIdx.x`, i.e. `Fin.rev`), as used by the reversed
    derivative headers. -/
def reverseSlot (n : Nat) (b : BlockIndex n) : BlockIndex n := Fin.rev b

/-- `Fin.rev` is an involution, so reverse scheduling is its own inverse. -/
theorem reverseSlot_involution (n : Nat) (b : BlockIndex n) :
    reverseSlot n (reverseSlot n b) = b := by
  unfold reverseSlot
  exact Fin.rev_rev b

/-- Reverse scheduling is injective: distinct logical blocks map to distinct
    physical slots (every logical block has a unique physical slot). -/
theorem reverseSlot_injective (n : Nat) : Injective (reverseSlot n) := by
  unfold reverseSlot
  exact Fin.rev_injective

/-- Reverse scheduling is surjective: every physical slot is reached by some
    logical block (every physical slot is the image of a logical block). -/
theorem reverseSlot_surjective (n : Nat) : Surjective (reverseSlot n) := by
  unfold reverseSlot
  exact Fin.rev_surjective

/-- Reverse scheduling is a bijection between logical blocks and physical
    slots: exactly-once coverage (each logical block → a unique slot, and every
    slot is covered by exactly one logical block). -/
theorem reverseSlot_bijective (n : Nat) : Bijective (reverseSlot n) := by
  unfold reverseSlot
  exact Fin.rev_bijective

/-- Explicit exactly-once: two different logical blocks never share a physical
    slot (contrapositive of injectivity). -/
theorem reverse_distinct_blocks_distinct_slots (n : Nat) {b c : BlockIndex n}
    (h : b ≠ c) : reverseSlot n b ≠ reverseSlot n c := by
  exact fun hsame => h (reverseSlot_injective n hsame)

/-- The block running at physical slot `p` is the (unique) logical block whose
    reverse is `p`; `Fin.rev` provides it. -/
theorem reverse_slot_of_destination (n : Nat) (p : BlockIndex n) :
    reverseSlot n (Fin.rev p) = p := by
  unfold reverseSlot
  simp

/-! ### Value preservation for block-indexed outputs -/

/-- The value computed by execution slot `p` under `sched`. This helper does
    not describe a memory address; `reverseWrite` below includes that address. -/
def scheduledOutput {n : Nat} {α : Type u} (sched : BlockIndex n → BlockIndex n)
    (blockOutput : BlockIndex n → α) (p : BlockIndex n) : α :=
  blockOutput (sched p)

/-- Execution slot `reverseSlot n b` computes the value for logical block `b`.
    The separate `reverseWrite_destination` theorem also preserves its address. -/
theorem reverse_value_preserved {n : Nat} {α : Type u} (blockOutput : BlockIndex n → α)
    (b : BlockIndex n) :
    scheduledOutput Fin.rev blockOutput (reverseSlot n b) = blockOutput b := by
  unfold scheduledOutput reverseSlot
  simp

/-- The output content at a physical slot `p` is the logical block `reverseSlot
    n p`'s output: a bijective relabelling of the logical outputs, covering
    every block exactly once. -/
theorem scheduledOutput_identity {n : Nat} {α : Type u} (blockOutput : BlockIndex n → α)
    (p : BlockIndex n) :
    scheduledOutput Fin.rev blockOutput p = blockOutput (reverseSlot n p) := by
  unfold scheduledOutput reverseSlot
  simp

/-! ### Explicit output address (destination) under reverse scheduling -/

/--
  The (destination, value) pair written by physical slot `p` under the reverse
  schedule.  The physical execution slot `p` COMPUTES logical block `Fin.rev p`
  and WRITES the output destined for logical block `Fin.rev p` — the execution
  slot is NOT the output address; the destination is the logical block index.
-/
def reverseWrite {n : Nat} {α : Type u} (blockOutput : BlockIndex n → α)
    (p : BlockIndex n) : BlockIndex n × α :=
  (Fin.rev p, blockOutput (Fin.rev p))

/-- At the physical slot `p = Fin.rev b` (the unique slot whose computed block
    is `b`), `reverseWrite` writes exactly the destination `b` with the value
    `blockOutput b`: both logical destination and value are preserved. -/
theorem reverseWrite_destination {n : Nat} {α : Type u} (blockOutput : BlockIndex n → α)
    (b : BlockIndex n) :
    reverseWrite blockOutput (Fin.rev b) = (b, blockOutput b) := by
  unfold reverseWrite
  simp

/-- Every destination logical block `b` is written by exactly one physical
    slot: `Fin.rev b`, the unique slot whose computed block is `b`. -/
theorem reverseWrite_dest_unique {n : Nat} {α : Type u} (blockOutput : BlockIndex n → α)
    (b : BlockIndex n) :
    ∃! p : BlockIndex n, (reverseWrite blockOutput p).1 = b := by
  refine ⟨Fin.rev b, ?_, ?_⟩
  · simp [reverseWrite]
  · intro p hp
    have hrev : Fin.rev p = b := by
      simp [reverseWrite] at hp
      exact hp
    calc
      p = Fin.rev (Fin.rev p) := (Fin.rev_rev p).symm
      _ = Fin.rev b := by rw [hrev]

/-! ### Row ownership -/

/-- The query rows owned by a logical block, under `Q` queries per block:
    `[b.val*Q, (b.val+1)*Q)`.  Row ownership is a function of the LOGICAL
    block, so reversing the schedule changes only which block runs in which
    physical slot — never which rows a block computes. -/
def ownedRows (Q n : Nat) (b : BlockIndex n) : Set Nat :=
  Set.Ico (b.val * Q) ((b.val + 1) * Q)

/-- A block's row range is preserved: the logical block `b` computes the same
    rows whether it is scheduled forward or reversed (it is just placed in a
    different physical slot).  Concretely, the block whose reverse is `b`
    (i.e. the block running at slot `reverseSlot b`) owns the rows of `b`. -/
theorem reverse_keeps_rows (Q n : Nat) (b : BlockIndex n) :
    ownedRows Q n b = ownedRows Q n (reverseSlot n (reverseSlot n b)) := by
  rw [reverseSlot_involution]

/-! ## 2. Half-open scratch lifetimes and shared capacity -/

/-- A scratch phase occupies the half-open time interval `[lo, hi)`, needs
    `size` bytes of shared memory, and has `pending t` outstanding asynchronous
    copies still to drain at time `t`. -/
structure ScratchPhase where
  lo : Nat   -- half-open start (inclusive)
  hi : Nat   -- half-open end (exclusive); renamed from `end` (reserved keyword)
  size : Nat
  pending : Nat → Nat   -- outstanding async copies at time t (0 when drained)

/-- A phase is live at time `t` iff `start ≤ t < end`. -/
def LiveAt (p : ScratchPhase) (t : Nat) : Prop := p.lo ≤ t ∧ t < p.hi

/-- If phase `a` ends no later than phase `b` starts (`a.hi ≤ b.lo`), no
    time point is live in both: the two lifetimes are disjoint. -/
theorem no_time_live_in_both (a b : ScratchPhase) (hab : a.hi ≤ b.lo) (t : Nat) :
    ¬ (LiveAt a t ∧ LiveAt b t) := by
  rintro ⟨ha, hb⟩
  have ht_lt_aend : t < a.hi := ha.2
  have ht_ge_bstart : b.lo ≤ t := hb.1
  have ht_lt_bstart : t < b.lo := lt_of_lt_of_le ht_lt_aend hab
  omega

/-- Symmetric disjointness: if `b` ends no later than `a` starts, no time point
    is live in both. -/
theorem no_time_live_in_both_sym (a b : ScratchPhase) (hba : b.hi ≤ a.lo) (t : Nat) :
    ¬ (LiveAt a t ∧ LiveAt b t) := by
  rintro ⟨ha, hb⟩
  have ht_lt_bend : t < b.hi := hb.2
  have ht_ge_astart : a.lo ≤ t := ha.1
  have ht_lt_astart : t < a.lo := lt_of_lt_of_le ht_lt_bend hba
  omega

/-- A shared buffer of capacity `max sizeA sizeB` is enough for either phase:
    each phase's requirement fits. -/
theorem max_capacity_covers_each (a b : ScratchPhase) :
    a.size ≤ max a.size b.size ∧ b.size ≤ max a.size b.size := by
  exact ⟨Nat.le_max_left _ _, Nat.le_max_right _ _⟩

/-- The shared capacity never exceeds the sum of the two separate allocations,
    so aliasing the two phases into one buffer is no larger than (and typically
    smaller than) keeping both. -/
theorem max_capacity_le_sum (a b : ScratchPhase) :
    max a.size b.size ≤ a.size + b.size := by
  omega

/-- A phase's asynchronous copies are fully drained: from its end `hi`
    onward, no copies remain outstanding (`pending t = 0`).  This is a model of
    `cp_async_fence` / `cp_async_wait` being honored before the phase's
    lifetime ends; it asserts NO hardware truth, only a per-phase abstract
    condition on `pending`. -/
def AsyncDrained (p : ScratchPhase) : Prop :=
  ∀ t : Nat, p.hi ≤ t → p.pending t = 0

/-- If phase `a` ends before phase `b` starts (`a.hi ≤ b.lo`) and `a`'s
    asynchronous copies are drained from `a.hi` onward, then at every time
    `t` live in `b` there are no outstanding copies from `a`.  This is the
    meaningful conditional guarantee: a later phase reusing `a`'s scratch
    never observes a live copy from `a`. -/
theorem drained_implies_pending_zero_in_later_phase (a b : ScratchPhase)
    (hab : a.hi ≤ b.lo) (hdrained : AsyncDrained a) (t : Nat) (ht : LiveAt b t) :
    a.pending t = 0 := by
  have hale : a.hi ≤ t := le_trans hab ht.1
  exact hdrained t hale

/-- Positive example: a phase whose pending-copy function is identically zero
    is drained, so a later phase sees zero outstanding copies. -/
def pendingfun0 : Nat → Nat := fun _ => 0

theorem pendingfun0_drained (size m1_end : Nat) :
    AsyncDrained { lo := 0, hi := m1_end, size := size, pending := pendingfun0 } := by
  unfold AsyncDrained pendingfun0
  simp

theorem pendingfun0_later_zero (a b : ScratchPhase) (ha0 : a.pending = pendingfun0)
    (hab : a.hi ≤ b.lo) (t : Nat) (ht : LiveAt b t) :
    a.pending t = 0 := by
  have hdrained : AsyncDrained a := by
    unfold AsyncDrained
    intro t' ht'
    rw [ha0]
    unfold pendingfun0
    rfl
  exact drained_implies_pending_zero_in_later_phase a b hab hdrained t ht

/-- Negative example: a phase whose pending-copy function is identically one is
    NOT drained, so the shared-buffer guarantee above cannot fire for it. -/
def pendingfun1 : Nat → Nat := fun _ => 1

theorem pendingfun1_not_drained (size m1_end : Nat) :
    ¬ AsyncDrained { lo := 0, hi := m1_end, size := size, pending := pendingfun1 } := by
  intro h
  have h1 : pendingfun1 m1_end = 0 := h m1_end (le_rfl)
  unfold pendingfun1 at h1
  norm_num at h1

/-! ### Instantiation: ordered MM1 then epilogue -/

/-- The MM1 phase (second matmul `attn @ V`) lives over `[0, m1_end)` and needs
    `sizeMM1` bytes. -/
def mm1Phase (sizeMM1 m1_end : Nat) (pending : Nat → Nat) : ScratchPhase :=
  { lo := 0, hi := m1_end, size := sizeMM1, pending := pending }

/-- The epilogue phase lives over `[m1_end, e_end)` and needs `sizeEpilogue`
    bytes. -/
def epiloguePhase (sizeEpilogue m1_end e_end : Nat) (pending : Nat → Nat) : ScratchPhase :=
  { lo := m1_end, hi := e_end, size := sizeEpilogue, pending := pending }

/-- MM1 then epilogue are time-disjoint: no time point is live in both. -/
theorem mm1_epilogue_no_overlap (sizeMM1 sizeEpilogue m1_end e_end : Nat)
    (pendingA pendingB : Nat → Nat) (t : Nat) :
    ¬ (LiveAt (mm1Phase sizeMM1 m1_end pendingA) t ∧
       LiveAt (epiloguePhase sizeEpilogue m1_end e_end pendingB) t) := by
  apply no_time_live_in_both
  exact le_rfl

/-- A shared buffer of capacity `max sizeMM1 sizeEpilogue` serves both the MM1
    and epilogue phases. -/
theorem mm1_epilogue_shared_capacity (sizeMM1 sizeEpilogue m1_end e_end : Nat)
    (pendingA pendingB : Nat → Nat) :
    (mm1Phase sizeMM1 m1_end pendingA).size ≤ max sizeMM1 sizeEpilogue ∧
      (epiloguePhase sizeEpilogue m1_end e_end pendingB).size ≤ max sizeMM1 sizeEpilogue := by
  exact max_capacity_covers_each (mm1Phase sizeMM1 m1_end pendingA)
    (epiloguePhase sizeEpilogue m1_end e_end pendingB)

/-! ## 3. Connection to the accepted CUDA row semantics -/

/--
  The block running at a physical slot under the reverse schedule delivers the
  abstract row output of the logical block it represents.  Together with the
  accepted `cuda_configs_row_semantics_agree` / `cuda_accepted_row_preserves`
  theorems (per-row Real softmax output is invariant under config / tile / hint
  selection), this means reversing the query-block launch order does NOT change
  any row's abstract Real value — it only relocates which physical slot computes
  that row's block.  This is the honest connection: the scheduling permutation
  is orthogonal to, and consistent with, the per-row semantic invariance.
-/
theorem reverse_delivers_block_row_output {n : Nat} {α : Type u} (rowOut : BlockIndex n → α)
    (b : BlockIndex n) :
    scheduledOutput Fin.rev rowOut (reverseSlot n b) = rowOut b :=
  reverse_value_preserved rowOut b

/-- Any accepted CUDA plan preserves, for every row `r`, the direct per-row
    Real softmax output (restated here so the scheduling file's connection is
    self-contained; this is the accepted theorem from `CudaPlan.lean`). -/
theorem accepted_row_semantics_restated (dev : VeriTac.Hardware.Target)
    (catalog : List VeriTac.Attention.CudaConfig)
    (plan : VeriTac.Attention.CudaPlan)
    (value score : Nat → ℝ) (m : VeriTac.Attention.Range → ℝ) (M : ℝ)
    (h : VeriTac.Attention.checkCudaPlan dev catalog plan = true)
    (r : Nat) (hr : r < plan.seqLen) :
    VeriTac.Attention.cudaRowMerged value score m M
        (VeriTac.Attention.selectedConfig catalog plan.configId).keysPerBlock (r + 1) =
      VeriTac.Attention.num value score M (Finset.range (r + 1)) /
        VeriTac.Attention.denom score M (Finset.range (r + 1)) :=
  VeriTac.Attention.cuda_accepted_row_preserves dev catalog plan value score m M h r hr

end Scheduling
