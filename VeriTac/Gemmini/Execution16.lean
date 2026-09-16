import VeriTac.Gemmini.Semantics
namespace VeriTac.Gemmini

def observe (s : Option MachineState) (idx : Nat) : Option (Int × Bool × Bool) :=
  s.map fun st => (st.output idx, st.written idx, st.drained)

set_option maxRecDepth 4000 in
set_option maxHeartbeats 800000 in
theorem run_baseline16_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 16) (hn : p.n = 16) (hk : p.k = 16)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 16 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 16) (hc : c < 16) :
    observe (runProgram p a b baseline16) (r * 16 + c) =
      some (signed32 (GemminiExact.dot (fun t => a (r * 16 + t))
        (fun t => b (t * 16 + c)) 16), true, true) := by
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  simp [observe, runProgram, runFrom, baseline16, step, initialState,
    tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
    hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hr, hc]

set_option maxRecDepth 4000 in
set_option maxHeartbeats 800000 in
theorem run_reuse16_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 16) (hn : p.n = 16) (hk : p.k = 16)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 16 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 16) (hc : c < 16) :
    observe (runProgram p a b reuse16) (r * 16 + c) =
      some (signed32 (GemminiExact.dot (fun t => a (r * 16 + t))
        (fun t => b (t * 16 + c)) 16), true, true) := by
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  simp [observe, runProgram, runFrom, reuse16, step, initialState,
    tileFits, accAddress, accRow, accumulates, tileProduct, GemminiExact.dot,
    hm, hn, hk, hsp, hsp16, hac, hdiv, hmod, hr, hc]
end VeriTac.Gemmini
