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

/-- Set `v` to `val`. We use propositional equality on the key so that the
    substitution / environment lemmas are stated over `=` (String `==` is BEq and
    does not simplify under `simp`). -/
def set (env : Env) (v : VarId) (val : Int) : Env :=
  fun x => if x = v then val else env x

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

namespace Env

/-- Setting two distinct keys commutes. -/
theorem set_set_comm {env : Env} {x y : VarId} {vx vy : Int} (h : x ≠ y) :
    Env.set (Env.set env x vx) y vy = Env.set (Env.set env y vy) x vx := by
  funext z
  by_cases hx : z = x
  · subst z
    simp [Env.set, h]
  · by_cases hy : z = y
    · subst z
      simp [Env.set, Ne.symm h]
    · simp [Env.set, hx, hy]

end Env

/-- If iteration bodies relate via a `v`-substitution in the environment, the two
    iteration folds agree even when their base environments differ by that binding.
    This is the key invariant behind tiling/unrolling: binding `v` directly is the
    same as binding the *index* variables and substituting `v ↦ (index expression)`.
    -/
theorem execLoopIters_subst (f1 f2 : Env → Store → Option Store) (env : Env)
    (v lv : VarId) (k : Int) (lo : Int) (iters : Nat) (store : Store)
    (hlv : lv ≠ v)
    (h : ∀ env' st, f1 (Env.set env' v k) st = f2 env' st) :
    execLoopIters f1 (Env.set env v k) lv lo iters store = execLoopIters f2 env lv lo iters store := by
  induction iters generalizing lo store with
  | zero => rfl
  | succ n ih =>
      simp only [execLoopIters]
      have hfirst : f1 (Env.set (Env.set env v k) lv lo) store = f2 (Env.set env lv lo) store := by
        rw [Env.set_set_comm (Ne.symm hlv)]
        exact h (Env.set env lv lo) store
      rw [hfirst]
      congr 1
      funext st'
      exact ih (lo + 1) st'

/-- Execute a statement. Fuel is carried for interface compatibility but is *not*
    consumed: every loop has a finite iteration count (`(hi - lo).toNat`), so the
    interpreter is total and `execStmt` never runs out of budget. Restructuring a
    loop nest (tiling, unrolling, ...) therefore cannot make the *leaf* computations
    receive less budget, which is exactly what makes the transformation-equivalence
    theorems provable for every `fuel`. -/
def execStmt : Nat → Env → Store → Stmt → Option Store
  | _, _, store, .skip => some store
  | fuel, env, store, .bufWrite buf indices valExpr =>
    let val := evalExpr env store valExpr
    let flat := flatIndex env store indices
    some (fun b i => if b == buf && i == flat then val else store b i)
  | fuel, env, store, .seq s1 s2 => do
    let store1 ← execStmt fuel env store s1
    execStmt fuel env store1 s2
  | fuel, env, store, .loop v lo hi _ann body =>
    let loVal := evalExpr env store lo
    let hiVal := evalExpr env store hi
    let iters := (hiVal - loVal).toNat
    execLoopIters (fun env' st => execStmt fuel env' st body) env v loVal iters store
  | fuel, env, store, .alloc _buf _shape body =>
    execStmt fuel env store body

end VeriTac
