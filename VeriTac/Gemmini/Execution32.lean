import VeriTac.Gemmini.Execution16
import VeriTac.Gemmini.Program32

namespace VeriTac.Gemmini

set_option maxRecDepth 8000 in
set_option maxHeartbeats 1600000 in
theorem run_baseline32_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 32) (hn : p.n = 16) (hk : p.k = 16)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 16 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 32) (hc : c < 16) :
    observe (runProgram p a b baseline32) (r * 16 + c) =
      some (signed32 (GemminiExact.dot (fun t => a (r * 16 + t))
        (fun t => b (t * 16 + c)) 16), true, true) := by
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  by_cases hlo : r < 16
  · have hncover : ¬ 256 ≤ r * 16 + c := by omega
    simp [observe, runProgram, runFrom, baseline32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
      hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hc, hlo, hncover]
  · have hcover : 256 ≤ r * 16 + c := by omega
    have hdiv2 : (r * 16 + c - 256) / 16 = r - 16 := by omega
    have hmod2 : (r * 16 + c - 256) % 16 = c := by omega
    have hr2 : r - 16 < 16 := by omega
    have hidx : ∀ t : Nat, 256 + (r - 16) * 16 + t = r * 16 + t := by omega
    simp [observe, runProgram, runFrom, baseline32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
      hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hc, hlo,
      hcover, hdiv2, hmod2, hr2, hidx]


set_option maxRecDepth 8000 in
set_option maxHeartbeats 1600000 in
theorem run_reuse32_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 32) (hn : p.n = 16) (hk : p.k = 16)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 32 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 32) (hc : c < 16) :
    observe (runProgram p a b reuse32) (r * 16 + c) =
      some (signed32 (GemminiExact.dot (fun t => a (r * 16 + t))
        (fun t => b (t * 16 + c)) 16), true, true) := by
  have hac16 : 16 ≤ p.accumulatorRows := by omega
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  by_cases hlo : r < 16
  · have hncover : ¬ 256 ≤ r * 16 + c := by omega
    simp [observe, runProgram, runFrom, reuse32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
      hm, hn, hk, hsp, hsp16, hac, hac16, hdiv, hmod, hc, hlo, hncover]
  · have hcover : 256 ≤ r * 16 + c := by omega
    have hdiv2 : (r * 16 + c - 256) / 16 = r - 16 := by omega
    have hmod2 : (r * 16 + c - 256) % 16 = c := by omega
    have hr2 : r - 16 < 16 := by omega
    have hidx : ∀ t : Nat, 256 + (r - 16) * 16 + t = r * 16 + t := by omega
    simp [observe, runProgram, runFrom, reuse32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
      hm, hn, hk, hsp, hsp16, hac, hac16, hdiv, hmod, hc, hlo,
      hcover, hdiv2, hmod2, hr2, hidx]

end VeriTac.Gemmini
