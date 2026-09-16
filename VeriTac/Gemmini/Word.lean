import VeriTac.GemminiExact

/-!
Signed machine-word interpretation for the exact Gemmini subset. The target
can use this arithmetic even outside the no-overflow domain; the bounds prove
that supported computations agree with mathematical integers.
-/
namespace VeriTac.Gemmini

/-- Two's-complement signed int32 interpretation, including wraparound. -/
def signed32 (x : Int) : Int := (x + 2147483648) % 4294967296 - 2147483648

theorem signed32_eq_of_bounds (x : Int)
    (hlo : -2147483648 ≤ x) (hhi : x ≤ 2147483647) : signed32 x = x := by
  unfold signed32
  rw [Int.emod_eq_of_lt (by omega : 0 ≤ x + 2147483648)
    (by omega : x + 2147483648 < 4294967296)]
  omega

/-- Every scalar reduction prefix has the same machine and integer value. -/
theorem signed32_dot_prefix (a b : Nat → Int) (k count : Nat)
    (hp : count ≤ k) (hk : k * 16384 ≤ 2147483647)
    (ha : ∀ t < k, GemminiExact.Int8 (a t))
    (hb : ∀ t < k, GemminiExact.Int8 (b t)) :
    signed32 (GemminiExact.dot a b count) = GemminiExact.dot a b count := by
  obtain ⟨hlo, hhi⟩ := GemminiExact.prefix_fits_int32 a b k count hp hk ha hb
  exact signed32_eq_of_bounds _ hlo hhi

/-- Repeated int32 accumulation is congruent to one final wrap. This holds even
outside the no-overflow contract and supports symbolic execution of many K tiles. -/
theorem signed32_add_wrapped (x y : Int) :
    signed32 (signed32 x + y) = signed32 (x + y) := by
  unfold signed32
  omega

theorem rowMajor_index_lt {r c rows cols : Nat} (hr : r < rows) (hc : c < cols) :
    r * cols + c < rows * cols := by
  nlinarith [Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)]

end VeriTac.Gemmini
