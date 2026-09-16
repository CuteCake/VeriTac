import VeriTac.Gemmini.Symbolic
import VeriTac.Gemmini.EngineMap

namespace VeriTac.Gemmini.Symbolic
open scoped BigOperators

def evalRef (a b : Nat → Int) : InputRef → Int
  | .a i => a i
  | .b i => b i

def evalTerms (a b : Nat → Int) (terms : Terms) : Int :=
  (terms.map (fun t => evalRef a b t.1 * evalRef a b t.2)).sum

def evalAcc (a b : Nat → Int) (terms : Terms) : Int := signed32 (evalTerms a b terms)

private theorem sum_list_range (f : Nat → Int) (n : Nat) :
    ((List.range n).map f).sum = ∑ t ∈ Finset.range n, f t := by
  have hrange : (List.range n).toFinset = Finset.range n := by
    ext i
    simp
  simpa only [hrange] using (List.sum_toFinset (l := List.range n) f List.nodup_range).symm

theorem eval_zero (a b : Nat → Int) : evalAcc a b ops.zero = intOps.zero := by
  simp [evalAcc, evalTerms, ops, intOps, signed32]

theorem eval_mac (a b : Nat → Int) (old : Terms) (left right : Nat → InputRef) :
    evalAcc a b (ops.mac old left right) =
      intOps.mac (evalAcc a b old) (fun t => evalRef a b (left t)) (fun t => evalRef a b (right t)) := by
  simp only [evalAcc, evalTerms, ops, intOps, List.map_append, List.sum_append, List.map_map,
    Function.comp_def, sum_list_range, signed32_add_wrapped]

theorem run_sound (p : GemminiPlan) (program : Program) (a b : Nat → Int) :
    (run p program).map (mapGState (evalRef a b) (evalAcc a b)) = runProgram p a b program := by
  have h := gRunProgram_map (evalRef a b) (evalAcc a b) ops intOps
    (eval_zero a b) (eval_mac a b) p InputRef.a InputRef.b program
  simpa [run, evalRef, gRunProgram_int] using h

theorem eval_expected (p : GemminiPlan) (a b : Nat → Int) (r c : Nat) :
    evalTerms a b (expected p r c) = gemm p a b r c := by
  simp [evalTerms, expected, List.map_map, Function.comp_def, evalRef, sum_list_range, gemm, GemminiExact.dot]

/-- One theorem for all shapes and all submitted instruction programs. The
checker proves the required addressing/initialization, symbolic output identity
and completion before this theorem derives numerical correctness. -/
theorem check_sound (p : GemminiPlan) (program : Program) (h : check p program = true)
    (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i)) :
    ∃ s, runProgram p a b program = some s ∧ s.drained = true ∧
      ∀ r < p.m, ∀ c < p.n,
        s.output (r * p.n + c) = gemm p a b r c ∧ s.written (r * p.n + c) = true := by
  simp only [check, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨hplan, _hdim⟩, hrun⟩ := h
  have legal := checkGemminiPlan_sound p hplan
  have hbound := legalPlan_bound p legal
  cases hs : run p program with
  | none => simp [hs] at hrun
  | some symbolic =>
    have accepted : symbolic.drained = true ∧ cellsCorrect p symbolic = true := by
      simpa [hs, Bool.and_eq_true] using hrun
    let concrete := mapGState (evalRef a b) (evalAcc a b) symbolic
    have hexec : runProgram p a b program = some concrete := by
      simpa [hs, concrete] using (run_sound p program a b).symm
    refine ⟨concrete, hexec, accepted.1, ?_⟩
    intro r hr c hc
    have hrows := List.all_eq_true.mp accepted.2
    have hcols := List.all_eq_true.mp (hrows r (List.mem_range.mpr hr))
    have hcell := Bool.and_eq_true_iff.mp (hcols c (List.mem_range.mpr hc))
    have hterms : symbolic.output (r * p.n + c) = expected p r c := of_decide_eq_true hcell.2
    have har : ∀ t < p.k, GemminiExact.Int8 (a (r * p.k + t)) := by
      intro t ht
      exact ha _ (rowMajor_index_lt hr ht)
    have hbc : ∀ t < p.k, GemminiExact.Int8 (b (t * p.n + c)) := by
      intro t ht
      exact hb _ (rowMajor_index_lt ht hc)
    have hw := signed32_dot_prefix (fun t => a (r * p.k + t)) (fun t => b (t * p.n + c))
      p.k p.k (by omega) hbound har hbc
    constructor
    · change evalAcc a b (symbolic.output (r * p.n + c)) = gemm p a b r c
      rw [hterms]
      simpa [evalAcc, eval_expected, gemm] using hw
    · exact hcell.1

end VeriTac.Gemmini.Symbolic
