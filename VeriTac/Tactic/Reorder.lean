/-
  VeriTac.Tactic.Reorder
  Reorder transformation: swap two adjacent independent loops.
-/
import VeriTac.Schedule.Equiv

namespace VeriTac.Tactic

/-- Check if a variable appears free in an expression. -/
def varInExpr (v : VarId) (e : SExpr) : Bool :=
  match e with
  | .lit _ => false
  | .var x => x == v
  | .add a b => varInExpr v a || varInExpr v b
  | .mul a b => varInExpr v a || varInExpr v b
  | .div a b => varInExpr v a || varInExpr v b
  | .mod a b => varInExpr v a || varInExpr v b
  | .bufRead _ indices => go v indices
where
  go (v : VarId) : List SExpr → Bool
    | [] => false
    | e :: es => varInExpr v e || go v es

/-- Check if a variable appears free in a statement. -/
def varInStmt (v : VarId) : Stmt → Bool
  | .skip => false
  | .bufWrite _ indices val =>
    varInExpr.go v indices || varInExpr v val
  | .loop lv lo hi _ body =>
    if lv == v then false
    else varInExpr v lo || varInExpr v hi || varInStmt v body
  | .seq s1 s2 => varInStmt v s1 || varInStmt v s2
  | .alloc _ shape body => varInExpr.go v shape || varInStmt v body

/-- Check if the bounds of loop2 depend on loop1's variable. -/
def loopsIndependent (var1 : VarId) (_var2 : VarId) (lo2 hi2 : SExpr) (_body : Stmt) : Bool :=
  !varInExpr var1 lo2 && !varInExpr var1 hi2

/-- Swap two adjacent nested loops if they are independent. -/
def reorder (stmt : Stmt) (var1 var2 : VarId) : Option Stmt :=
  match stmt with
  | .loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body) =>
    if v1 == var1 && v2 == var2 then
      if loopsIndependent v1 v2 lo2 hi2 body then
        some (.loop v2 lo2 hi2 ann2 (.loop v1 lo1 hi1 ann1 body))
      else none
    else
      match reorder body var1 var2 with
      | some body' => some (.loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body'))
      | none => none
  | .loop v lo hi ann body =>
    match reorder body var1 var2 with
    | some body' => some (.loop v lo hi ann body')
    | none => none
  | .seq s1 s2 =>
    match reorder s1 var1 var2 with
    | some s1' => some (.seq s1' s2)
    | none => match reorder s2 var1 var2 with
      | some s2' => some (.seq s1 s2')
      | none => none
  | _ => none

/-- Reorder correctness: swapping two independent loops preserves equivalence. -/
theorem reorder_correct (v1 v2 : VarId) (lo1 hi1 lo2 hi2 : SExpr)
    (ann1 ann2 : Annotation) (body : Stmt)
    (hind : loopsIndependent v1 v2 lo2 hi2 body = true) :
    .loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body) ≈ₛ
    .loop v2 lo2 hi2 ann2 (.loop v1 lo1 hi1 ann1 body) := by
  sorry

end VeriTac.Tactic
