import VeriTac.Gemmini.Plan
import VeriTac.GemminiExact

namespace VeriTac.Gemmini

/-- Accepted plans justify exact blocked reduction and all scalar int32
    prefix bounds. This connects the executable plan checker to mathematical
    semantics, not to the unverified Python/C instruction emitter. -/
theorem checked_reduction_correct (p : GemminiPlan)
    (accepted : checkGemminiPlan p = true) (a b : Nat → Int)
    (ha : ∀ t < p.k, GemminiExact.Int8 (a t))
    (hb : ∀ t < p.k, GemminiExact.Int8 (b t)) :
    GemminiExact.tiledDot a b p.dim (p.k / p.dim) = GemminiExact.dot a b p.k ∧
    ∀ count ≤ p.k,
      -2147483648 ≤ GemminiExact.dot a b count ∧
      GemminiExact.dot a b count ≤ 2147483647 := by
  have legal := checkGemminiPlan_sound p accepted
  constructor
  · apply GemminiExact.tiledDot_eq_dot_of_dvd
    exact Nat.dvd_of_mod_eq_zero legal.2.2.2.2.2.2.1
  · intro count hc
    exact GemminiExact.prefix_fits_int32 a b p.k count hc (legalPlan_bound p legal) ha hb

end VeriTac.Gemmini
