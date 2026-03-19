/-
  VeriTac.Tactic.Split
  Split transformation: thin wrapper over tile.
-/
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Split is an alias for tile — same transformation, different naming convention.
    `split v factor` = `tile v factor`. -/
def split (stmt : Stmt) (targetVar : VarId) (factor : Nat) : Option Stmt :=
  tile stmt targetVar factor

theorem split_correct (v : VarId) (N factor : Nat) (hf : factor > 0) (body : Stmt)
    (ann : Annotation) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      execStmt fuel env store (.loop v (.lit 0) (.lit N) ann body) =
      execStmt fuel env store
        (match split (.loop v (.lit 0) (.lit N) ann body) v factor with
         | some s => s
         | none => .loop v (.lit 0) (.lit N) ann body) := by
  intro fuel env store
  exact tile_correct v N factor hf body ann fuel env store

end VeriTac.Tactic
