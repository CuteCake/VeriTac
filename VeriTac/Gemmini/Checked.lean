import VeriTac.Gemmini.Execution32

namespace VeriTac.Gemmini

private theorem observation_exact (p : GemminiPlan) (a b : Nat → Int) (program : Program)
    (hn : p.n = 16) (hk : p.k = 16)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i))
    (r c : Nat) (hr : r < p.m) (hc : c < p.n)
    (he : observe (runProgram p a b program) (r * 16 + c) =
      some (signed32 (GemminiExact.dot (fun t => a (r * 16 + t))
        (fun t => b (t * 16 + c)) 16), true, true)) :
    observe (runProgram p a b program) (r * p.n + c) = some (gemm p a b r c, true, true) := by
  have ha8 : ∀ t < 16, GemminiExact.Int8 (a (r * 16 + t)) := by
    intro t ht
    apply ha
    simp only [hk]
    omega
  have hb8 : ∀ t < 16, GemminiExact.Int8 (b (t * 16 + c)) := by
    intro t ht
    apply hb
    simp only [hk, hn]
    omega
  have hw := signed32_dot_prefix (fun t => a (r * 16 + t))
    (fun t => b (t * 16 + c)) 16 16 (by omega) (by decide) ha8 hb8
  simpa [gemm, hk, hn, hw] using he

/-- Concrete instruction acceptance establishes each mathematical GEMM output
for all signed int8 inputs, as well as output coverage and final completion. -/
theorem checkSupportedProgram_observe (p : GemminiPlan) (program : Program)
    (h : checkSupportedProgram p program = true) (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i))
    (r c : Nat) (hr : r < p.m) (hc : c < p.n) :
    observe (runProgram p a b program) (r * p.n + c) = some (gemm p a b r c, true, true) := by
  rcases Bool.or_eq_true_iff.mp h with h | h
  · simp only [checkProgram, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨hp, hm⟩, hn⟩, hk⟩, hd⟩, hsched⟩ := h
    obtain ⟨_, _, _, _, _, _, _, _, hsp, hac⟩ := checkGemminiPlan_sound p hp
    have hsp32 : 32 ≤ p.scratchpadRows := by simpa [requiredScratchpadRows, hd] using hsp
    apply observation_exact p a b program hn hk ha hb r c hr hc
    cases hs : p.schedule with
    | baseline =>
      have heq : program = baseline16 := by simpa [hs] using hsched
      subst program
      apply run_baseline16_observe p a b hm hn hk hsp32
      · simpa [requiredAccumulatorRows, hs, hd] using hac
      · omega
      · omega
    | reuseB =>
      have heq : program = reuse16 := by simpa [hs] using hsched
      subst program
      apply run_reuse16_observe p a b hm hn hk hsp32
      · simpa [requiredAccumulatorRows, hs, hm] using hac
      · omega
      · omega
  · simp only [checkProgram32, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨hp, hm⟩, hn⟩, hk⟩, hd⟩, hsched⟩ := h
    obtain ⟨_, _, _, _, _, _, _, _, hsp, hac⟩ := checkGemminiPlan_sound p hp
    have hsp32 : 32 ≤ p.scratchpadRows := by simpa [requiredScratchpadRows, hd] using hsp
    apply observation_exact p a b program hn hk ha hb r c hr hc
    cases hs : p.schedule with
    | baseline =>
      have heq : program = baseline32 := by simpa [hs] using hsched
      subst program
      apply run_baseline32_observe p a b hm hn hk hsp32
      · simpa [requiredAccumulatorRows, hs, hd] using hac
      · omega
      · omega
    | reuseB =>
      have heq : program = reuse32 := by simpa [hs] using hsched
      subst program
      apply run_reuse32_observe p a b hm hn hk hsp32
      · simpa [requiredAccumulatorRows, hs, hm] using hac
      · omega
      · omega

theorem checkSupportedProgram_legal (p : GemminiPlan) (program : Program)
    (h : checkSupportedProgram p program = true) : planLegalProps p := by
  rcases Bool.or_eq_true_iff.mp h with h | h
  · simp only [checkProgram, Bool.and_eq_true, beq_iff_eq] at h
    exact checkGemminiPlan_sound p h.1.1.1.1.1
  · simp only [checkProgram32, Bool.and_eq_true, beq_iff_eq] at h
    exact checkGemminiPlan_sound p h.1.1.1.1.1

/-- Acceptance proves successful execution, completion, and all outputs. Neither
interpreter success nor output correctness is a premise of this theorem. -/
theorem checkSupportedProgram_sound (p : GemminiPlan) (program : Program)
    (h : checkSupportedProgram p program = true) (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i)) :
    ∃ s, runProgram p a b program = some s ∧ s.drained = true ∧
      ∀ r < p.m, ∀ c < p.n,
        s.output (r * p.n + c) = gemm p a b r c ∧ s.written (r * p.n + c) = true := by
  have legal := checkSupportedProgram_legal p program h
  have hzero := checkSupportedProgram_observe p program h a b ha hb 0 0 legal.1 legal.2.1
  cases he : runProgram p a b program with
  | none => simp [observe, he] at hzero
  | some s =>
    have hz : (s.output 0, s.written 0, s.drained) = (gemm p a b 0 0, true, true) := by
      exact Option.some.inj (by simpa [observe, he] using hzero)
    refine ⟨s, rfl, congrArg (fun x : Int × Bool × Bool => x.2.2) hz, ?_⟩
    intro r hr c hc
    have ho := checkSupportedProgram_observe p program h a b ha hb r c hr hc
    have ho' : (s.output (r * p.n + c), s.written (r * p.n + c), s.drained) =
        (gemm p a b r c, true, true) := Option.some.inj (by simpa [observe, he] using ho)
    exact ⟨congrArg Prod.fst ho', congrArg (fun x : Int × Bool × Bool => x.2.1) ho'⟩

end VeriTac.Gemmini
