/-
  VeriTac.Attention.Partition
  Algebraic semantics of partitioned stable-softmax attention.

  Stable softmax computes, over a finite set of keys, the summary
      denom m s = ∑ i ∈ s, exp (score i - m)          (scale reference m)
      num   m s = ∑ i ∈ s, value i * exp (score i - m) (weighted numerator)
  where `value` stands for a single component of a (possibly vector) value; the
  vector numerator is obtained componentwise, so every result here is
  backend-independent.

  Central result.  If the key set `K` is written as a disjoint union of finite
  partitions `parts`, each summarized against its own local scale reference
  `m p`, then rescaling every partial summary by `exp (m p - M)` and summing
  reproduces exactly the full summary computed against the global reference
  `M`.  Consequently normalized attention is invariant under partitioning, and
  merging is additive (associative).

  A decidable structural legality checker `legalCausalPartition` guarantees
  that a list of contiguous half-open ranges covers the causal key prefix
  `[0, N)` exactly once (even with irregular tails); when such a legal partition
  is used, the merged summary provably equals the direct summary.

  Scope: this file proves the *abstract Real semantics* of the transformation.
  It makes no claim about floating-point accuracy or about any compiler /
  backend refinement.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Interval
import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Pairwise
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Pairwise
import Mathlib.Data.Finset.Powerset
import Mathlib.Analysis.SpecialFunctions.Exp

namespace VeriTac.Attention

/-! ## Stable-softmax summaries over a finite key set -/

variable {ι : Type}

/-- Denominator of stable softmax over `s` with scale reference `m`:
    `∑ i ∈ s, exp (score i - m)`. -/
noncomputable def denom (score : ι → ℝ) (m : ℝ) (s : Finset ι) : ℝ :=
  ∑ i ∈ s, Real.exp (score i - m)

/-- Per-component weighted numerator of stable softmax over `s`:
    `∑ i ∈ s, value i * exp (score i - m)`. -/
noncomputable def num (value score : ι → ℝ) (m : ℝ) (s : Finset ι) : ℝ :=
  ∑ i ∈ s, value i * Real.exp (score i - m)

/-! ## Rescaling a summary against a different scale reference -/

/-- Changing the scale reference from `m` to `M` rescales the denominator by
    `exp (m - M)`.  Holds for any references `m, M`, not only for running maxima. -/
theorem denom_rescale (score : ι → ℝ) (s : Finset ι) (m M : ℝ) :
    denom score M s = denom score m s * Real.exp (m - M) := by
  unfold denom
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  rw [← Real.exp_add]
  congr 1
  ring_nf

/-- Changing the scale reference from `m` to `M` rescales the per-component
    numerator by `exp (m - M)`. -/
theorem num_rescale (value score : ι → ℝ) (s : Finset ι) (m M : ℝ) :
    num value score M s = num value score m s * Real.exp (m - M) := by
  unfold num
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  rw [mul_assoc]
  congr 1
  rw [← Real.exp_add]
  congr 1
  ring_nf

/-- The numerator of an empty key set is zero. -/
theorem num_empty (value score : ι → ℝ) (m : ℝ) :
    num value score m ∅ = 0 := by
  unfold num
  simp

/-- The denominator of an empty key set is zero. -/
theorem denom_empty (score : ι → ℝ) (m : ℝ) :
    denom score m ∅ = 0 := by
  unfold denom
  simp

/-- The denominator of a nonempty key set is strictly positive, so the
    stable-softmax normalization is well defined (no division by zero / NaN in
    the abstract semantics). -/
theorem denom_pos (score : ι → ℝ) (s : Finset ι) (m : ℝ) (hs : s.Nonempty) :
    0 < denom score m s := by
  unfold denom
  exact Finset.sum_pos (fun i hi => Real.exp_pos (score i - m)) hs

/-! ## Merging partial summaries of disjoint partitions -/

/-- Sum over partitions of rescaled partial denominators, targeting reference `M`. -/
noncomputable def denomMerged (score : ι → ℝ) (parts : Finset (Finset ι))
    (m : Finset ι → ℝ) (M : ℝ) : ℝ :=
  ∑ p ∈ parts, denom score (m p) p * Real.exp (m p - M)

/-- Sum over partitions of rescaled partial numerators, targeting reference `M`. -/
noncomputable def numMerged (value score : ι → ℝ) (parts : Finset (Finset ι))
    (m : Finset ι → ℝ) (M : ℝ) : ℝ :=
  ∑ p ∈ parts, num value score (m p) p * Real.exp (m p - M)

/-- Partition invariance for the denominator: if `parts` is a pairwise-disjoint
    cover of `K`, then the merged (rescaled) denominator equals the direct
    denominator over all of `K`. -/
theorem denom_merge_disjoint [DecidableEq ι] (score : ι → ℝ) (K : Finset ι)
    (parts : Finset (Finset ι)) (m : Finset ι → ℝ) (M : ℝ)
    (hdisj : (parts : Set (Finset ι)).PairwiseDisjoint id)
    (hcover : parts.biUnion id = K) :
    denomMerged score parts m M = denom score M K := by
  unfold denomMerged
  calc
    (∑ p ∈ parts, denom score (m p) p * Real.exp (m p - M))
        = ∑ p ∈ parts, denom score M p := by
            apply Finset.sum_congr rfl
            intro p hp
            exact (denom_rescale score p (m p) M).symm
    _ = ∑ i ∈ parts.biUnion id, Real.exp (score i - M) := by
            simp [denom]
            rw [Finset.sum_biUnion hdisj]
            rfl
    _ = denom score M K := by
            rw [hcover]
            rfl

/-- Partition invariance for the (per-component) numerator: merging rescaled
    partial numerators over a disjoint cover of `K` equals the direct numerator
    over `K`. -/
theorem num_merge_disjoint [DecidableEq ι] (value score : ι → ℝ) (K : Finset ι)
    (parts : Finset (Finset ι)) (m : Finset ι → ℝ) (M : ℝ)
    (hdisj : (parts : Set (Finset ι)).PairwiseDisjoint id)
    (hcover : parts.biUnion id = K) :
    numMerged value score parts m M = num value score M K := by
  unfold numMerged
  calc
    (∑ p ∈ parts, num value score (m p) p * Real.exp (m p - M))
        = ∑ p ∈ parts, num value score M p := by
            apply Finset.sum_congr rfl
            intro p hp
            exact (num_rescale value score p (m p) M).symm
    _ = ∑ i ∈ parts.biUnion id, value i * Real.exp (score i - M) := by
            simp [num]
            rw [Finset.sum_biUnion hdisj]
            rfl
    _ = num value score M K := by
            rw [hcover]
            rfl

/-- Additivity (associativity) of merging: merging two disjoint partition sets
    gives the sum of their individual merged denominators. -/
theorem denom_merge_add [DecidableEq ι] (score : ι → ℝ)
    (parts1 parts2 : Finset (Finset ι)) (m : Finset ι → ℝ) (M : ℝ)
    (hdisj : Disjoint parts1 parts2) :
    denomMerged score (parts1 ∪ parts2) m M =
      denomMerged score parts1 m M + denomMerged score parts2 m M := by
  unfold denomMerged
  rw [Finset.sum_union hdisj]

/-- Normalized attention is invariant under partitioning: computing the softmax
    output directly over all keys `K` or by merging per-partition summaries
    against a common reference `M` gives the same value.  `hK` records that the
    key set is nonempty (the stable denominator is positive via `denom_pos`). -/
theorem attention_merge_disjoint [DecidableEq ι] (value score : ι → ℝ) (K : Finset ι)
    (parts : Finset (Finset ι)) (m : Finset ι → ℝ) (M : ℝ)
    (hdisj : (parts : Set (Finset ι)).PairwiseDisjoint id)
    (hcover : parts.biUnion id = K) (hK : K.Nonempty) :
    num value score M K / denom score M K =
      numMerged value score parts m M / denomMerged score parts m M := by
  have _hpos : 0 < denom score M K := denom_pos score K M hK
  rw [num_merge_disjoint value score K parts m M hdisj hcover,
      denom_merge_disjoint score K parts m M hdisj hcover]

/-! ## Structural legality of contiguous range partitions -/

/-- A half-open range of key indices `[lo, hi)`. -/
abbrev Range := Nat × Nat

/-- The key indices covered by a range. -/
def rangeSet (r : Range) : Finset Nat := Finset.Ico r.1 r.2

/-- Union of a list of finite sets. -/
def listUnion {α : Type} [DecidableEq α] : List (Finset α) → Finset α
  | [] => ∅
  | s :: rest => s ∪ listUnion rest

/-- Union of the key sets of a list of ranges. -/
def unionRanges (ranges : List Range) : Finset Nat :=
  listUnion (ranges.map rangeSet)

/-- Structural legality: the ranges, read in order, begin exactly at `pos`, are
    well formed (`lo ≤ hi`, empty ranges allowed), each next range starts where
    the previous one ended, and the walk finishes exactly at `N`. -/
def legalFrom (pos N : Nat) : List Range → Bool
  | [] => decide (pos = N)
  | (lo, hi) :: rest => decide (lo = pos) && decide (lo ≤ hi) && legalFrom hi N rest

/-- Decidable legality check for a causal partition of the key prefix `[0, N)`:
    contiguous, well-formed ranges covering it exactly once (empty leading/tail
    ranges and irregular tails allowed). -/
def legalCausalPartition (N : Nat) (ranges : List Range) : Bool :=
  legalFrom 0 N ranges

/-- A list of finite sets is a partition of its union: every element is disjoint
    from the union of the remaining elements. -/
def ListPartition {α : Type} [DecidableEq α] : List (Finset α) → Prop
  | [] => True
  | s :: rest => Disjoint s (listUnion rest) ∧ ListPartition rest

/-- A legal walk never overshoots the end `N`. -/
theorem legalFrom_pos_le (pos N : Nat) :
    ∀ ranges : List Range, legalFrom pos N ranges = true → pos ≤ N := by
  intro ranges
  induction ranges generalizing pos with
  | nil =>
      intro h
      exact Nat.le_of_eq (of_decide_eq_true h)
  | cons r rest ih =>
      intro h
      rcases r with ⟨lo, hi⟩
      simp only [legalFrom, Bool.and_eq_true, decide_eq_true_eq] at h
      rcases h with ⟨⟨hpos, hord⟩, hrest⟩
      have hhi : hi ≤ N := ih hi hrest
      subst lo
      exact le_trans hord hhi

/-- Adjacent half-open intervals `[a, b)` and `[b, c)` are disjoint. -/
theorem Ico_disjoint_right (a b c : Nat) :
    Disjoint (Finset.Ico a b) (Finset.Ico b c) := by
  rw [Finset.disjoint_left]
  intro x hx1 hx2
  have hxb : x < b := (Finset.mem_Ico.mp hx1).2
  have hbx : b ≤ x := (Finset.mem_Ico.mp hx2).1
  exact (not_lt_of_ge hbx) hxb

/-- Every member of a range list is a subset of the union of all of them. -/
theorem mem_unionRanges_subset {r : Range} (ranges : List Range) (hr : r ∈ ranges) :
    rangeSet r ⊆ unionRanges ranges := by
  induction ranges with
  | nil => simp at hr
  | cons r' rest ih =>
      simp only [List.mem_cons] at hr
      rcases hr with rfl | hr'
      · simp [unionRanges, listUnion]
      · exact (ih hr').trans (by simp [unionRanges, listUnion])

/-- A legal walk covers `[pos, N)` exactly (union of the ranges). -/
theorem unionRanges_contig (pos N : Nat) :
    ∀ ranges : List Range, legalFrom pos N ranges = true →
      unionRanges ranges = Finset.Ico pos N := by
  intro ranges
  induction ranges generalizing pos with
  | nil =>
      intro h
      have : pos = N := of_decide_eq_true h
      subst pos
      simp [unionRanges, listUnion]
  | cons r rest ih =>
      intro h
      rcases r with ⟨lo, hi⟩
      simp only [legalFrom, Bool.and_eq_true, decide_eq_true_eq] at h
      rcases h with ⟨⟨hpos, hord⟩, hrest⟩
      have hrest' : unionRanges rest = Finset.Ico hi N := ih hi hrest
      have hhi : hi ≤ N := legalFrom_pos_le hi N rest hrest
      subst lo
      change rangeSet (pos, hi) ∪ unionRanges rest = Finset.Ico pos N
      rw [hrest']
      simp [rangeSet]
      exact Finset.Ico_union_Ico_eq_Ico hord hhi

/-- A legal walk yields a partition of its ranges' key sets: the head range is
    disjoint from the union of the remaining ones, and recursively so. -/
theorem legalFrom_listPartition (pos N : Nat) :
    ∀ ranges : List Range, legalFrom pos N ranges = true →
      ListPartition (ranges.map rangeSet) := by
  intro ranges
  induction ranges generalizing pos with
  | nil => intro h; simp [ListPartition]
  | cons r rest ih =>
      intro h
      rcases r with ⟨lo, hi⟩
      simp only [legalFrom, Bool.and_eq_true, decide_eq_true_eq] at h
      rcases h with ⟨⟨hpos, hord⟩, hrest⟩
      have ihrest : ListPartition (rest.map rangeSet) := ih hi hrest
      have hcontig : unionRanges rest = Finset.Ico hi N := unionRanges_contig hi N rest hrest
      subst lo
      constructor
      · rw [← unionRanges, hcontig]
        exact Ico_disjoint_right pos hi N
      · exact ihrest

/-- The sum over a partition's elements of their internal sums equals the sum
    over the partition's union. -/
theorem sum_list_disjoint {α : Type} [DecidableEq α] (f : α → ℝ) :
    ∀ ps : List (Finset α), ListPartition ps →
      (ps.map (fun s => ∑ i ∈ s, f i)).sum = ∑ i ∈ listUnion ps, f i := by
  intro ps
  induction ps with
  | nil => intro h; simp [listUnion]
  | cons s rest ih =>
      intro h
      rcases h with ⟨hdisj, hrest⟩
      have hrest_sum := ih hrest
      simp [listUnion]
      rw [hrest_sum]
      rw [Finset.sum_union hdisj]

/-- Specialization of `sum_list_disjoint` to ranges. -/
theorem sum_ranges_disjoint (f : Nat → ℝ) (ranges : List Range)
    (hpart : ListPartition (ranges.map rangeSet)) :
    (ranges.map (fun p => ∑ i ∈ rangeSet p, f i)).sum = ∑ i ∈ unionRanges ranges, f i := by
  have h := sum_list_disjoint f (ranges.map rangeSet) hpart
  simpa [unionRanges, List.map_map] using h

/-- For a partition, the merged (rescaled) denominator over the ranges equals
    the direct denominator over their union. -/
theorem denom_merged_list (score : Nat → ℝ) (ranges : List Range)
    (m : Range → ℝ) (M : ℝ) (hpart : ListPartition (ranges.map rangeSet)) :
    (ranges.map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
      = denom score M (unionRanges ranges) := by
  calc
    (ranges.map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
        = (ranges.map (fun p => denom score M (rangeSet p))).sum := by
            apply congrArg List.sum
            apply List.map_congr_left
            intro p hp
            exact (denom_rescale score (rangeSet p) (m p) M).symm
    _ = denom score M (unionRanges ranges) := by
            change (ranges.map (fun p => denom score M (rangeSet p))).sum =
              ∑ i ∈ unionRanges ranges, Real.exp (score i - M)
            rw [← sum_ranges_disjoint (fun i => Real.exp (score i - M)) ranges hpart]
            apply congrArg List.sum
            apply List.map_congr_left
            intro p hp
            rfl

/-- For a partition, the merged (rescaled) per-component numerator over the
    ranges equals the direct numerator over their union. -/
theorem num_merged_list (value score : Nat → ℝ) (ranges : List Range)
    (m : Range → ℝ) (M : ℝ) (hpart : ListPartition (ranges.map rangeSet)) :
    (ranges.map (fun p => num value score (m p) (rangeSet p) * Real.exp (m p - M))).sum
      = num value score M (unionRanges ranges) := by
  calc
    (ranges.map (fun p => num value score (m p) (rangeSet p) * Real.exp (m p - M))).sum
        = (ranges.map (fun p => num value score M (rangeSet p))).sum := by
            apply congrArg List.sum
            apply List.map_congr_left
            intro p hp
            exact (num_rescale value score (rangeSet p) (m p) M).symm
    _ = num value score M (unionRanges ranges) := by
            change (ranges.map (fun p => num value score M (rangeSet p))).sum =
              ∑ i ∈ unionRanges ranges, value i * Real.exp (score i - M)
            rw [← sum_ranges_disjoint (fun i => value i * Real.exp (score i - M)) ranges hpart]
            apply congrArg List.sum
            apply List.map_congr_left
            intro p hp
            rfl

/-! ## Semantic preservation linked to the structural legality checker -/

/-- Main objective.  If `ranges` is a *legally* contiguous partition of the
    causal key prefix `[0, N)`, then merging each range's stable-softmax summary
    (rescaled by `exp (m p - M)` from its local reference `m p` to the global
    reference `M`) produces exactly the direct softmax output over all keys.
    The merged normalized output is therefore independent of how the causal
    prefix is split, including irregular tails. -/
theorem merged_equals_direct
    (value score : Nat → ℝ) (N : Nat) (ranges : List Range)
    (m : Range → ℝ) (M : ℝ)
    (hlegal : legalCausalPartition N ranges = true) :
    (ranges.map (fun p => num value score (m p) (rangeSet p) * Real.exp (m p - M))).sum /
      (ranges.map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
    = num value score M (Finset.range N) / denom score M (Finset.range N) := by
  have hpart : ListPartition (ranges.map rangeSet) :=
    legalFrom_listPartition 0 N ranges hlegal
  have hcover : unionRanges ranges = Finset.range N := by
    rw [unionRanges_contig 0 N ranges hlegal]
    rw [← Finset.range_eq_Ico]
  have hnum : (ranges.map (fun p => num value score (m p) (rangeSet p) * Real.exp (m p - M))).sum
      = num value score M (Finset.range N) := by
    rw [num_merged_list value score ranges m M hpart]
    rw [hcover]
  have hden : (ranges.map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
      = denom score M (Finset.range N) := by
    rw [denom_merged_list score ranges m M hpart]
    rw [hcover]
  rw [hnum, hden]

/-! ## Legality-checker examples -/

/-- A one-key causal prefix `[0,1)` split into an empty leading range and the
    key itself is accepted (empty ranges are legal). -/
example : legalCausalPartition 1 [(0, 0), (0, 1)] = true := by
  decide

/-- An irregular tail is accepted: the last range does not align to any fixed
    block size, it just closes out the prefix `[0,5)`. -/
example : legalCausalPartition 5 [(0, 2), (2, 5)] = true := by
  decide

/-- Overlapping ranges are rejected. -/
example : legalCausalPartition 3 [(0, 2), (1, 3)] = false := by
  decide

/-- A gap (keys `1` and `2` uncovered) is rejected. -/
example : legalCausalPartition 4 [(0, 1), (2, 4)] = false := by
  decide

/-- An out-of-bounds range (extending past `N`) is rejected. -/
example : legalCausalPartition 3 [(0, 4)] = false := by
  decide

end VeriTac.Attention
