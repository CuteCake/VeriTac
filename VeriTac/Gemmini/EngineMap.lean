import VeriTac.Gemmini.Semantics

namespace VeriTac.Gemmini

def mapTile {α γ : Type} (f : α → γ) (tile : ValueTile α) : ValueTile γ :=
  fun r c => f (tile r c)

def mapGState {α β γ δ : Type} (f : α → γ) (g : β → δ) (s : GState α β) : GState γ δ := {
  ws := s.ws, ld0 := s.ld0, ld1 := s.ld1, stStride := s.stStride
  spad := fun row => (s.spad row).map (mapTile f)
  acc := fun row => (s.acc row).map (mapTile g)
  pending := s.pending
  weights := s.weights.map (mapTile f)
  output := fun idx => g (s.output idx)
  written := s.written
  drained := s.drained
}

private theorem option_map_if {α β : Type} (f : α → β) (P : Prop) [Decidable P]
    (a b : Option α) : (if P then a else b).map f = if P then a.map f else b.map f := by
  split <;> rfl

private theorem option_bind_if {α β : Type} (f : α → Option β) (P : Prop) [Decidable P]
    (a b : Option α) : (if P then a else b).bind f = if P then a.bind f else b.bind f := by
  split <;> rfl

theorem mapTile_mac {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (ops : EvalOps α β) (ops' : EvalOps γ δ)
    (hm : ∀ old a b, g (ops.mac old a b) =
      ops'.mac (g old) (fun t => f (a t)) (fun t => f (b t)))
    (old : ValueTile β) (a b : ValueTile α) :
    mapTile g (fun r c => ops.mac (old r c) (fun t => a r t) (fun t => b t c)) =
      (fun r c => ops'.mac (g (old r c)) (fun t => f (a r t)) (fun t => f (b t c))) := by
  funext r c
  exact hm _ _ _

theorem gstep_map {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (ops : EvalOps α β) (ops' : EvalOps γ δ)
    (hz : g ops.zero = ops'.zero)
    (hm : ∀ old a b, g (ops.mac old a b) =
      ops'.mac (g old) (fun t => f (a t)) (fun t => f (b t)))
    (p : GemminiPlan) (a b : Nat → α) (s : GState α β) (i : Instr) :
    (gstep ops p a b s i).map (mapGState f g) =
      gstep ops' p (fun idx => f (a idx)) (fun idx => f (b idx)) (mapGState f g s) i := by
  cases i <;> unfold gstep <;>
    simp [mapGState, mapTile, Option.map_bind, Option.bind_map, option_map_if,
      option_bind_if, apply_ite g, apply_ite f, hz, hm]
  case mvin slot buf offset row cols rows =>
    cases buf <;> dsimp only [mapTile] <;> rfl
  case compute retained ar bd =>
    simp only [mapTile_mac f g ops ops' hm, hz]

theorem mapGState_initial {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (zero : β) (zero' : δ) (hz : g zero = zero') :
    mapGState f g (initialGState zero) = initialGState zero' := by
  simp [mapGState, initialGState, hz]

theorem gRunFrom_map {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (ops : EvalOps α β) (ops' : EvalOps γ δ)
    (hz : g ops.zero = ops'.zero)
    (hm : ∀ old a b, g (ops.mac old a b) =
      ops'.mac (g old) (fun t => f (a t)) (fun t => f (b t)))
    (p : GemminiPlan) (a b : Nat → α) (program : Program) (s : GState α β) :
    (gRunFrom ops p a b program s).map (mapGState f g) =
      gRunFrom ops' p (fun idx => f (a idx)) (fun idx => f (b idx)) program (mapGState f g s) := by
  induction program generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [gRunFrom, Option.map_bind, Function.comp_def, ih]
    rw [← gstep_map f g ops ops' hz hm]
    simp only [Option.bind_map, Function.comp_def]

theorem gRunProgram_map {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (ops : EvalOps α β) (ops' : EvalOps γ δ)
    (hz : g ops.zero = ops'.zero)
    (hm : ∀ old a b, g (ops.mac old a b) =
      ops'.mac (g old) (fun t => f (a t)) (fun t => f (b t)))
    (p : GemminiPlan) (a b : Nat → α) (program : Program) :
    (gRunProgram ops p a b program).map (mapGState f g) =
      gRunProgram ops' p (fun idx => f (a idx)) (fun idx => f (b idx)) program := by
  unfold gRunProgram
  rw [gRunFrom_map f g ops ops' hz hm, mapGState_initial f g _ _ hz]

end VeriTac.Gemmini
