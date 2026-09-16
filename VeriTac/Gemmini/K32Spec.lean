import VeriTac.Gemmini.Semantics

namespace VeriTac.Gemmini
open scoped BigOperators

def k32Result (a b : Nat → Int) (r c : Nat) : Int :=
  signed32 (signed32 (∑ t ∈ Finset.range 16, a (r * 32 + t) * b (t * 16 + c)) +
    ∑ t ∈ Finset.range 16, a (r * 32 + 16 + t) * b ((16 + t) * 16 + c))

theorem k32Result_exact (p : GemminiPlan) (a b : Nat → Int)
    (hn : p.n = 16) (hk : p.k = 32)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i))
    (r c : Nat) (hr : r < p.m) (hc : c < p.n) :
    k32Result a b r c = gemm p a b r c := by
  have ha8 : ∀ t < 32, GemminiExact.Int8 (a (r * 32 + t)) := by
    intro t ht
    apply ha
    simpa [hk] using (rowMajor_index_lt hr (show t < p.k by omega))
  have hb8 : ∀ t < 32, GemminiExact.Int8 (b (t * 16 + c)) := by
    intro t ht
    apply hb
    simpa [hn] using (rowMajor_index_lt (show t < p.k by omega) hc)
  have hsum : GemminiExact.dot (fun t => a (r * 32 + t)) (fun t => b (t * 16 + c)) 32 =
      (∑ t ∈ Finset.range 16, a (r * 32 + t) * b (t * 16 + c)) +
      ∑ t ∈ Finset.range 16, a (r * 32 + 16 + t) * b ((16 + t) * 16 + c) := by
    unfold GemminiExact.dot
    rw [show (32 : Nat) = 16 + 16 from rfl, Finset.sum_range_add]
    simp only [Nat.add_assoc]
  unfold k32Result
  rw [signed32_add_wrapped, ← hsum]
  have hw := signed32_dot_prefix (fun t => a (r * 32 + t)) (fun t => b (t * 16 + c))
    32 32 (by omega) (by decide) ha8 hb8
  simpa [gemm, hn, hk] using hw

end VeriTac.Gemmini
