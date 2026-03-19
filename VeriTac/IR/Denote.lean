/-
  VeriTac.IR.Denote
  Denotational semantics: maps TExpr to mathematical functions.
-/
import VeriTac.IR.TExpr

namespace VeriTac

/-- Fold a function over Fin n, accumulating from an initial value. -/
def foldFin {α : Type} (f : α → α → α) (init : α) : (n : Nat) → (Fin n → α) → α
  | 0, _ => init
  | k + 1, g => f (foldFin f init k (fun i => g ⟨i.val, by omega⟩)) (g ⟨k, by omega⟩)

/-- Denotational semantics: interpret a TExpr as a function from indices to values. -/
def denote {α : Type} {s : Shape} : TExpr α s → (Index s → α)
  | .const v => fun _ => v
  | .tensor f => f
  | .map g e => fun idx => g (denote e idx)
  | .zip g e1 e2 => fun idx => g (denote e1 idx) (denote e2 idx)
  | @TExpr.reduce _ n _ f init e => fun idx =>
      foldFin f init n (fun i => denote e (Index.cons i idx))

/-- Map fusion: map g (map f e) = map (g ∘ f) e -/
theorem denote_map_map {α : Type} {s : Shape} (f g : α → α) (e : TExpr α s) :
    denote (.map g (.map f e)) = denote (.map (g ∘ f) e) := by
  funext idx
  unfold denote
  rfl

end VeriTac
