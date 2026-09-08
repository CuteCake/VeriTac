/-
  VeriTac.Tactic.Unroll
  Unroll transformation: replicate loop body to eliminate loop overhead.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Generate `n` copies of a statement body, substituting the loop variable
    with successive literal values starting from `base`. -/
def unrollBody (v : VarId) (base : Int) (n : Nat) (body : Stmt) : Stmt :=
  match n with
  | 0 => .skip
  | k + 1 => .seq (substStmtVar v (.lit base) body) (unrollBody v (base + 1) k body)

/-- Fully unroll a loop with known constant bounds.
    Transforms `loop v lo hi body` into `body[v:=lo]; body[v:=lo+1]; ...; body[v:=hi-1]`. -/
def unroll (stmt : Stmt) (targetVar : VarId) : Option Stmt :=
  match stmt with
  | .loop v (.lit lo) (.lit hi) _ann body =>
    if v = targetVar then
      let n := (hi - lo).toNat
      if n == 0 then some .skip
      else some (unrollBody v lo n body)
    else
      match unroll body targetVar with
      | some body' => some (.loop v (.lit lo) (.lit hi) _ann body')
      | none => none
  | .loop v lo hi ann body =>
    match unroll body targetVar with
    | some body' => some (.loop v lo hi ann body')
    | none => none
  | .seq s1 s2 =>
    match unroll s1 targetVar with
    | some s1' => some (.seq s1' s2)
    | none => match unroll s2 targetVar with
      | some s2' => some (.seq s1 s2')
      | none => none
  | _ => none

/-- Connecting lemma: iterating a loop body via `execLoopIters` is the same as
    running the sequentially-unrolled body. This is where loop semantics meets the
    literal-substitution lemma. -/
theorem unrollBody_correct (v : VarId) (fuel : Nat) (body : Stmt) :
    ∀ (env : Env) (lo : Int) (n : Nat) (store : Store),
      loopBinds v body = false →
      execLoopIters (fun env' st => execStmt fuel env' st body) env v lo n store
      = execStmt fuel env store (unrollBody v lo n body) := by
  intro env lo n store
  induction n generalizing lo store with
  | zero => intro _; simp [execLoopIters, unrollBody, execStmt]
  | succ k ih =>
      intro hb
      have hsub : execStmt fuel (Env.set env v lo) store body =
          execStmt fuel env store (substStmtVar v (.lit lo) body) :=
        substStmtVar_lit_correct v lo store fuel env body hb
      simp only [execLoopIters, unrollBody]
      rw [hsub]
      congr 1
      funext st'
      exact ih (lo + 1) st' hb

/-- Unroll correctness: unrolling a loop with constant bounds preserves semantics. -/
theorem unroll_correct (v : VarId) (N : Nat) (body : Stmt) (ann : Annotation)
    (hb : loopBinds v body = false) :
    .loop v (.lit 0) (.lit N) ann body ≈ₛ unrollBody v 0 N body := by
  intro fuel env store
  simpa [execStmt, evalExpr] using (unrollBody_correct v fuel body env 0 N store hb)

end VeriTac.Tactic
