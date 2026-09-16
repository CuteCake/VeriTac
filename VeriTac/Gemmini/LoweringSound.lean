import VeriTac.Gemmini.Lowering
import VeriTac.Gemmini.SymbolicSound

namespace VeriTac.Gemmini

/-- Parameterized validated lowering: the same theorem covers every returned
program, shape, row batch and int8 input, rather than four fixed instances. -/
theorem checkedLowering_sound (p : GemminiPlan) (checked : CheckedProgram p)
    (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i)) :
    ∃ s, runProgram p a b checked.val = some s ∧ s.drained = true ∧
      ∀ r < p.m, ∀ c < p.n,
        s.output (r * p.n + c) = gemm p a b r c ∧ s.written (r * p.n + c) = true :=
  Symbolic.check_sound p checked.val checked.property a b ha hb

/-- Both endpoints execute successfully and agree on every GEMM output.
The proposer and its rewrite implementation are outside the trust boundary. -/
theorem checkedRewrite_sound (p : GemminiPlan) (source : Program)
    (edge : CheckedRewrite p source) (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i)) :
    ∃ before after,
      runProgram p a b source = some before ∧
      runProgram p a b edge.destination = some after ∧
      before.drained = true ∧ after.drained = true ∧
      ∀ r < p.m, ∀ c < p.n,
        before.output (r * p.n + c) = after.output (r * p.n + c) := by
  obtain ⟨before, hbRun, hbDone, hbOut⟩ := Symbolic.check_sound p source edge.sourceAccepted a b ha hb
  obtain ⟨after, haRun, haDone, haOut⟩ := Symbolic.check_sound p edge.destination edge.destinationAccepted a b ha hb
  refine ⟨before, after, hbRun, haRun, hbDone, haDone, ?_⟩
  intro r hr c hc
  exact (hbOut r hr c hc).1.trans (haOut r hr c hc).1.symm

end VeriTac.Gemmini
