import Mathlib

/-!
Exact integer arithmetic for the initial Gemmini subset. These theorems
justify reduction blocking and bound every reduction count. They do not
model DMA, address encoding, or the C emitter.
-/
namespace VeriTac.GemminiExact

open scoped BigOperators

def dot (a b : Nat → Int) (k : Nat) : Int :=
  ∑ t ∈ Finset.range k, a t * b t

/-- A sequence of tile reductions accumulated in increasing K-block order. -/
def tiledDot (a b : Nat → Int) (dim : Nat) : Nat → Int
  | 0 => 0
  | blocks + 1 => tiledDot a b dim blocks +
      ∑ t ∈ Finset.range dim, a (blocks * dim + t) * b (blocks * dim + t)

theorem tiledDot_eq_dot (a b : Nat → Int) (dim blocks : Nat) :
    tiledDot a b dim blocks = dot a b (blocks * dim) := by
  induction blocks with
  | zero => simp [tiledDot, dot]
  | succ blocks ih =>
    simp only [tiledDot, ih, dot, Nat.succ_mul, Finset.sum_range_add]

theorem tiledDot_eq_dot_of_dvd (a b : Nat → Int) (dim k : Nat)
    (h : dim ∣ k) : tiledDot a b dim (k / dim) = dot a b k := by
  rw [tiledDot_eq_dot, Nat.div_mul_cancel h]

def Int8 (x : Int) : Prop := -128 ≤ x ∧ x ≤ 127

theorem int8_abs (x : Int) (h : Int8 x) : |x| ≤ 128 := by
  apply abs_le.mpr
  constructor
  · exact h.1
  · exact le_trans h.2 (by norm_num)

theorem int8_product_abs (x y : Int) (hx : Int8 x) (hy : Int8 y) :
    |x * y| ≤ 16384 := by
  rw [abs_mul]
  have hx' := int8_abs x hx
  have hy' := int8_abs y hy
  nlinarith [abs_nonneg x, abs_nonneg y]

theorem dot_abs_le (a b : Nat → Int) (k : Nat)
    (ha : ∀ t < k, Int8 (a t)) (hb : ∀ t < k, Int8 (b t)) :
    |dot a b k| ≤ (k : Int) * 16384 := by
  unfold dot
  calc
    |∑ t ∈ Finset.range k, a t * b t| ≤
        ∑ t ∈ Finset.range k, |a t * b t| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _t ∈ Finset.range k, (16384 : Int) := by
      apply Finset.sum_le_sum
      intro t ht
      exact int8_product_abs (a t) (b t)
        (ha t (Finset.mem_range.mp ht)) (hb t (Finset.mem_range.mp ht))
    _ = (k : Int) * 16384 := by simp

/-- Every scalar reduction count fits signed int32, not just the final sum. -/
theorem prefix_fits_int32 (a b : Nat → Int) (k count : Nat)
    (hp : count ≤ k) (hk : k * 16384 ≤ 2147483647)
    (ha : ∀ t < k, Int8 (a t)) (hb : ∀ t < k, Int8 (b t)) :
    -2147483648 ≤ dot a b count ∧ dot a b count ≤ 2147483647 := by
  have h := dot_abs_le a b count
    (fun t ht => ha t (lt_of_lt_of_le ht hp))
    (fun t ht => hb t (lt_of_lt_of_le ht hp))
  have hbound : (count : Int) * 16384 ≤ 2147483647 := by
    exact_mod_cast (le_trans (Nat.mul_le_mul_right 16384 hp) hk)
  have habs : |dot a b count| ≤ 2147483647 := le_trans h hbound
  have hs := abs_le.mp habs
  constructor
  · omega
  · exact hs.2

end VeriTac.GemminiExact
