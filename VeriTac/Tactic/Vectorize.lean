/-
  VeriTac.Tactic.Vectorize
  Vectorize transformation: annotate innermost loop for SIMD execution.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Reorder

namespace VeriTac.Tactic

/-- Check if a statement contains any loop. -/
def hasLoop : Stmt → Bool
  | .loop _ _ _ _ _ => true
  | .seq s1 s2 => hasLoop s1 || hasLoop s2
  | .alloc _ _ body => hasLoop body
  | _ => false

/-- Check if a statement is an innermost loop (no nested loops in body). -/
def isInnermostLoop : Stmt → Bool
  | .loop _ _ _ _ body => !hasLoop body
  | _ => false

/-- Annotate a loop as vectorized. Precondition: must be innermost loop. -/
def vectorize (stmt : Stmt) (targetVar : VarId) : Option Stmt :=
  match stmt with
  | .loop v lo hi _ann body =>
    if v == targetVar then
      if isInnermostLoop (.loop v lo hi _ann body) then
        some (.loop v lo hi .vectorize body)
      else none
    else
      match vectorize body targetVar with
      | some body' => some (.loop v lo hi _ann body')
      | none => none
  | .seq s1 s2 =>
    match vectorize s1 targetVar with
    | some s1' => some (.seq s1' s2)
    | none => match vectorize s2 targetVar with
      | some s2' => some (.seq s1 s2')
      | none => none
  | _ => none

/-- Vectorize correctness: annotation doesn't change semantics.
    `execStmt` discards the loop annotation (`_ann`), so a loop and its vectorized
    form execute identically. -/
theorem vectorize_correct (v : VarId) (lo hi : SExpr) (ann : Annotation) (body : Stmt)
    (_hinner : isInnermostLoop (.loop v lo hi ann body) = true) :
    .loop v lo hi ann body ≈ₛ .loop v lo hi .vectorize body := by
  intro fuel env store
  cases fuel <;> rfl

end VeriTac.Tactic
