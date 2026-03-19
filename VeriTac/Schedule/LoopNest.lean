/-
  VeriTac.Schedule.LoopNest
  Scheduled IR: Stmt, Expr, Env, Store, and fuel-based exec_stmt.
-/
import VeriTac.IR.Shape

namespace VeriTac

/-- Variable identifiers in the scheduled IR. -/
abbrev VarId := String

/-- Buffer identifiers. -/
abbrev BufId := String

/-- Annotations on loops for code generation. -/
inductive Annotation where
  | none
  | parallel
  | vectorize
  | unrolled
  deriving Repr, BEq, Inhabited

/-- Expressions in the scheduled IR. -/
inductive SExpr where
  | lit (n : Int)
  | var (v : VarId)
  | add (a b : SExpr)
  | mul (a b : SExpr)
  | div (a b : SExpr)
  | mod (a b : SExpr)
  | bufRead (buf : BufId) (indices : List SExpr)
  deriving Repr, BEq, Inhabited

/-- Statements in the scheduled IR. -/
inductive Stmt where
  | skip
  | bufWrite (buf : BufId) (indices : List SExpr) (val : SExpr)
  | loop (var : VarId) (lo hi : SExpr) (ann : Annotation) (body : Stmt)
  | seq (s1 s2 : Stmt)
  | alloc (buf : BufId) (shape : List SExpr) (body : Stmt)
  deriving Repr, BEq, Inhabited

/-- Environment: maps variable names to integer values. -/
abbrev Env := VarId → Int

/-- Store: maps (buffer, flat_index) to integer values. -/
abbrev Store := BufId → Int → Int

namespace Env

def empty : Env := fun _ => 0

def set (env : Env) (v : VarId) (val : Int) : Env :=
  fun x => if x == v then val else env x

end Env

/-- Evaluate an expression in an environment and store. -/
def evalExpr (env : Env) (store : Store) : SExpr → Int
  | .lit n => n
  | .var v => env v
  | .add a b => evalExpr env store a + evalExpr env store b
  | .mul a b => evalExpr env store a * evalExpr env store b
  | .div a b =>
    let bv := evalExpr env store b
    if bv = 0 then 0 else evalExpr env store a / bv
  | .mod a b =>
    let bv := evalExpr env store b
    if bv = 0 then 0 else evalExpr env store a % bv
  | .bufRead buf indices =>
    let flat := indices.foldl (fun acc idx => acc * 1000 + evalExpr env store idx) 0
    store buf flat

/-- Compute a flat index from a list of index expressions. -/
def flatIndex (env : Env) (store : Store) (indices : List SExpr) : Int :=
  indices.foldl (fun acc idx => acc * 1000 + evalExpr env store idx) 0

/-- Execute loop iterations using Nat count for termination. -/
def execLoopIters (execBody : Env → Store → Option Store)
    (env : Env) (v : VarId) (cur : Int) (remaining : Nat)
    (store : Store) : Option Store :=
  match remaining with
  | 0 => some store
  | n + 1 => do
    let store' ← execBody (env.set v cur) store
    execLoopIters execBody env v (cur + 1) n store'

/-- Execute a statement with fuel-based termination.
    Returns `none` if fuel is exhausted. -/
def execStmt : Nat → Env → Store → Stmt → Option Store
  | _, _, store, .skip => some store
  | fuel, env, store, .bufWrite buf indices valExpr =>
    let val := evalExpr env store valExpr
    let flat := flatIndex env store indices
    some (fun b i => if b == buf && i == flat then val else store b i)
  | fuel, env, store, .seq s1 s2 => do
    let store1 ← execStmt fuel env store s1
    execStmt fuel env store1 s2
  | 0, _, _, .loop _ _ _ _ _ => Option.none
  | fuel + 1, env, store, .loop v lo hi _ann body =>
    let loVal := evalExpr env store lo
    let hiVal := evalExpr env store hi
    let iters := (hiVal - loVal).toNat
    execLoopIters (fun env' st => execStmt fuel env' st body) env v loVal iters store
  | fuel, env, store, .alloc _buf _shape body =>
    execStmt fuel env store body

end VeriTac
