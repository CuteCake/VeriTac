/-
  VeriTac.Tactic.Tile
  Tile transformation: split a loop into outer/inner loops.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Util.Finset

namespace VeriTac.Tactic

/-- Substitute a variable in an expression with a replacement expression. -/
def substExprVar (v : VarId) (replacement : SExpr) : SExpr → SExpr
  | .lit n => .lit n
  | .var x => if x == v then replacement else .var x
  | .add a b => .add (substExprVar v replacement a) (substExprVar v replacement b)
  | .mul a b => .mul (substExprVar v replacement a) (substExprVar v replacement b)
  | .div a b => .div (substExprVar v replacement a) (substExprVar v replacement b)
  | .mod a b => .mod (substExprVar v replacement a) (substExprVar v replacement b)
  | .bufRead buf indices => .bufRead buf (indices.map (substExprVar v replacement))
termination_by e => sizeOf e

/-- Substitute a variable in a statement with an expression. -/
def substStmtVar (v : VarId) (replacement : SExpr) : Stmt → Stmt
  | .skip => .skip
  | .bufWrite buf indices val =>
    .bufWrite buf (indices.map (substExprVar v replacement)) (substExprVar v replacement val)
  | .loop lv lo hi ann body =>
    if lv == v then .loop lv lo hi ann body  -- shadowed
    else .loop lv (substExprVar v replacement lo) (substExprVar v replacement hi)
                  ann (substStmtVar v replacement body)
  | .seq s1 s2 => .seq (substStmtVar v replacement s1) (substStmtVar v replacement s2)
  | .alloc buf shape body =>
    .alloc buf (shape.map (substExprVar v replacement)) (substStmtVar v replacement body)

/-- Apply the tile transformation to a loop statement.
    Precondition: `tileSize > 0`. -/
def tile (stmt : Stmt) (targetVar : VarId) (tileSize : Nat) : Option Stmt :=
  match stmt with
  | .loop v lo hi ann body =>
    if v == targetVar then
      let outerVar := v ++ "_outer"
      let innerVar := v ++ "_inner"
      let ts := SExpr.lit tileSize
      let outerHi := SExpr.div (.add hi (.lit (tileSize - 1))) ts
      let innerHi := SExpr.lit tileSize
      let replacement := SExpr.add (.mul (.var outerVar) ts) (.var innerVar)
      let newBody := substStmtVar v replacement body
      some (.loop outerVar lo outerHi ann
        (.loop innerVar (.lit 0) innerHi .none newBody))
    else
      match tile body targetVar tileSize with
      | some tiledBody => some (.loop v lo hi ann tiledBody)
      | none => none
  | .seq s1 s2 =>
    match tile s1 targetVar tileSize with
    | some s1' => some (.seq s1' s2)
    | none => match tile s2 targetVar tileSize with
      | some s2' => some (.seq s1 s2')
      | none => none
  | .alloc buf shape body =>
    match tile body targetVar tileSize with
    | some body' => some (.alloc buf shape body')
    | none => none
  | _ => none

/-- Tile correctness theorem. -/
theorem tile_correct (v : VarId) (N ts : Nat) (_hts : ts > 0) (body : Stmt)
    (ann : Annotation) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      execStmt fuel env store (.loop v (.lit 0) (.lit N) ann body) =
      execStmt fuel env store
        (match tile (.loop v (.lit 0) (.lit N) ann body) v ts with
         | some s => s
         | none => .loop v (.lit 0) (.lit N) ann body) := by
  sorry

end VeriTac.Tactic
