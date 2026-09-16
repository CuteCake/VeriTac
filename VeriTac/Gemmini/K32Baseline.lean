import VeriTac.Gemmini.Execution16
import VeriTac.Gemmini.K32Spec

namespace VeriTac.Gemmini

/-!
Concrete K=32 baseline program and its execution certificate.

`baselineK32` is `gen_baseline` instantiated at M=32, N=16, K=32 (verified
field-by-field against the Python generator): two output-row tiles, two K
tiles each, the second k tile accumulating onto the first through the bit-30
accumulator address onto the nonzero first-tile partial sums.
-/

def baselineK32 : Program := [
  .configEx 1, .configSt 64, .configLd 0 32 true, .configLd 1 16 true,
  .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvin 0 .a 16 0 16 16, .mvin 1 .b 256 16 16 16,
  .preload 16 3758096384, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 512 0 16 16, .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvin 0 .a 528 0 16 16, .mvin 1 .b 256 16 16 16,
  .preload 16 3758096384, .compute false 0 4294967295,
  .mvout 256 3758096384 16 16, .fence]

set_option maxRecDepth 8000 in
set_option maxHeartbeats 1600000 in
theorem run_baselineK32_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 32) (hn : p.n = 16) (hk : p.k = 32)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 16 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 32) (hc : c < 16) :
    observe (runProgram p a b baselineK32) (r * 16 + c) =
      some (k32Result a b r c, true, true) := by
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  by_cases hlo : r < 16
  · have hncover : ¬ 256 ≤ r * 16 + c := by omega
    have hidx1 : ∀ t : Nat, 16 + r * 32 + t = r * 32 + 16 + t := by intro t; omega
    have hidx2 : ∀ t : Nat, 256 + t * 16 + c = (16 + t) * 16 + c := by intro t; omega
    simp [observe, runProgram, runFrom, baselineK32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, k32Result,
      hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hc, hlo, hncover, hidx1, hidx2]
  · have hcover : 256 ≤ r * 16 + c := by omega
    have hdiv2 : (r * 16 + c - 256) / 16 = r - 16 := by omega
    have hmod2 : (r * 16 + c - 256) % 16 = c := by omega
    have hr2 : r - 16 < 16 := by omega
    have hidx1 : ∀ t : Nat, 512 + (r - 16) * 32 + t = r * 32 + t := by intro t; omega
    have hidx2 : ∀ t : Nat, 528 + (r - 16) * 32 + t = r * 32 + 16 + t := by intro t; omega
    have hidx3 : ∀ t : Nat, 256 + t * 16 + c = (16 + t) * 16 + c := by intro t; omega
    simp [observe, runProgram, runFrom, baselineK32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, k32Result,
      hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hc, hlo,
      hcover, hdiv2, hmod2, hr2, hidx1, hidx2, hidx3]

end VeriTac.Gemmini
