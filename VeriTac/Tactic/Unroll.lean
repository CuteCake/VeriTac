/-
  VeriTac.Tactic.Unroll
  Unroll transformation: replicate loop body to eliminate loop overhead.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Generate `n` copies of a statement body, substituting the loop variable
    with successive values starting from `base`. -/
def unrollBody (v : VarId) (base : SExpr) (n : Nat) (body : Stmt) : Stmt :=
  match n with
  | 0 => .skip
  | 1 => substStmtVar v base body
  | k + 1 =>
    .seq (substStmtVar v base body)
         (unrollBody v (.add base (.lit 1)) k body)

/-- Fully unroll a loop with known constant bounds.
    Transforms `loop v 0 N body` into `body[v:=0]; body[v:=1]; ...; body[v:=N-1]`. -/
def unroll (stmt : Stmt) (targetVar : VarId) : Option Stmt :=
  match stmt with
  | .loop v (.lit lo) (.lit hi) _ann body =>
    if v == targetVar then
      let n := (hi - lo).toNat
      if n == 0 then some .skip
      else some (unrollBody v (.lit lo) n body)
    else
      match unroll body targetVar with
      | some body' => some (.loop v (.lit lo) (.lit hi) _ann body')
      | none => none
  | .seq s1 s2 =>
    match unroll s1 targetVar with
    | some s1' => some (.seq s1' s2)
    | none => match unroll s2 targetVar with
      | some s2' => some (.seq s1 s2')
      | none => none
  | .loop v lo hi ann body =>
    match unroll body targetVar with
    | some body' => some (.loop v lo hi ann body')
    | none => none
  | _ => none

/-- Unroll correctness: unrolling a loop with constant bounds preserves semantics. -/
theorem unroll_correct (v : VarId) (N : Nat) (body : Stmt) (ann : Annotation) :
    .loop v (.lit 0) (.lit N) ann body ≈ₛ
    unrollBody v (.lit 0) N body := by
  sorry -- Induction on N, using substitution correctness

end VeriTac.Tactic
