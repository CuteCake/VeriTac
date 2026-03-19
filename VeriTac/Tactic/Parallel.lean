/-
  VeriTac.Tactic.Parallel
  Parallel transformation: annotate a loop for parallel execution.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Reorder

namespace VeriTac.Tactic

/-- Collect all buffer writes in a statement, returning (buf, indices_exprs) pairs. -/
def collectWrites : Stmt → List (BufId × List SExpr)
  | .skip => []
  | .bufWrite buf indices _ => [(buf, indices)]
  | .loop _ _ _ _ body => collectWrites body
  | .seq s1 s2 => collectWrites s1 ++ collectWrites s2
  | .alloc _ _ body => collectWrites body

/-- Conservative independence check: iterations along `loopVar` are independent
    if no two iterations can write to the same buffer location.
    This checks that the loop variable appears in all write index expressions. -/
def checkIndependence (loopVar : VarId) (body : Stmt) : Bool :=
  let writes := collectWrites body
  writes.all fun (_, indices) => varInExpr.go loopVar indices

/-- Annotate a loop as parallel. Precondition: iterations must be independent. -/
def parallel (stmt : Stmt) (targetVar : VarId) : Option Stmt :=
  match stmt with
  | .loop v lo hi _ann body =>
    if v == targetVar then
      if checkIndependence v body then
        some (.loop v lo hi .parallel body)
      else none
    else
      match parallel body targetVar with
      | some body' => some (.loop v lo hi _ann body')
      | none => none
  | .seq s1 s2 =>
    match parallel s1 targetVar with
    | some s1' => some (.seq s1' s2)
    | none => match parallel s2 targetVar with
      | some s2' => some (.seq s1 s2')
      | none => none
  | _ => none

/-- Parallel correctness: for independent iterations, parallel annotation preserves semantics. -/
theorem parallel_correct (v : VarId) (lo hi : SExpr) (ann : Annotation) (body : Stmt)
    (_hind : checkIndependence v body = true) :
    .loop v lo hi ann body ≈ₛ .loop v lo hi .parallel body := by
  intro fuel env store
  sorry

end VeriTac.Tactic
