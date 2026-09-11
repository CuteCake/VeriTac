/-
  VeriTac.AttentionPartition
  Smoke test for the partitioned stable-softmax semantics.  Imports the module
  (a full typecheck) and exercises the main theorem on a concrete legal causal
  partition, confirming the merged output equals the direct output.
-/
import VeriTac.Attention.Partition

namespace VeriTac.Attention

/- The legality checker accepts an irregular tail and an empty leading range. -/
example : legalCausalPartition 5 [(0, 0), (0, 2), (2, 5)] = true := by
  decide

/- It rejects overlaps and gaps. -/
example : legalCausalPartition 4 [(0, 2), (2, 3), (2, 4)] = false := by
  decide
example : legalCausalPartition 4 [(0, 2), (3, 4)] = false := by
  decide

/- Concrete semantic preservation: for the legal partition of `[0,4)` into
   `[0,2)`, `[2,4)`, the merged (rescaled) summaries equal the direct summary,
   no matter the local references `m`. -/
theorem concrete_partition_merge
    (score : Nat → ℝ) (m : Range → ℝ) (M : ℝ) :
    ([(0, 2), (2, 4)].map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
      = denom score M (Finset.range 4) := by
  have hlegal : legalCausalPartition 4 [(0, 2), (2, 4)] = true := by
    decide
  have hpart : ListPartition ([(0, 2), (2, 4)].map rangeSet) :=
    legalFrom_listPartition 0 4 [(0, 2), (2, 4)] hlegal
  calc
    ([(0, 2), (2, 4)].map (fun p => denom score (m p) (rangeSet p) * Real.exp (m p - M))).sum
        = denom score M (unionRanges [(0, 2), (2, 4)]) := denom_merged_list score [(0, 2), (2, 4)] m M hpart
    _ = denom score M (Finset.range 4) := by
        rw [unionRanges_contig 0 4 [(0, 2), (2, 4)] hlegal]
        rw [← Finset.range_eq_Ico]

end VeriTac.Attention
