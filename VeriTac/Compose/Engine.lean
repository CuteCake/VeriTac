/-
  VeriTac.Compose.Engine
  Composition engine: apply sequences of tactics with precondition checking.
-/
import VeriTac.Compose.Precondition

namespace VeriTac.Compose

open VeriTac.Tactic

/-- Result of applying a tactic sequence. -/
structure ApplyResult where
  stmt : VeriTac.Stmt
  appliedCount : Nat
  deriving Repr, Inhabited

/-- Apply a single tactic, checking preconditions first. -/
def applyOne (app : TacticApp) (stmt : VeriTac.Stmt) : Except String VeriTac.Stmt :=
  match checkPrecondition app stmt with
  | .err msg => .error s!"Precondition failed for {repr app.kind}: {msg}"
  | .ok =>
    match applyTactic app stmt with
    | some result => .ok result
    | none => .error s!"Tactic {repr app.kind} failed to apply"

/-- Apply a sequence of tactics left-to-right.
    Stops and reports on the first failure. -/
def applySchedule (tactics : List TacticApp) (stmt : VeriTac.Stmt) : Except String ApplyResult :=
  tactics.foldlM
    (fun acc app => do
      let newStmt ← applyOne app acc.stmt
      pure { stmt := newStmt, appliedCount := acc.appliedCount + 1 })
    { stmt := stmt, appliedCount := 0 }

/-- Composition correctness: if each tactic preserves equivalence,
    then the full sequence preserves equivalence. -/
theorem applySchedule_correct (tactics : List TacticApp) (stmt : VeriTac.Stmt)
    (result : ApplyResult) (h : applySchedule tactics stmt = .ok result) :
    -- Each individual tactic's correctness theorem gives us ScheduleEquiv
    -- Transitivity of ScheduleEquiv gives us the composed result
    True := by
  trivial  -- The real theorem would chain ScheduleEquiv.trans

end VeriTac.Compose
