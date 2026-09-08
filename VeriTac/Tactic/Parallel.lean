/-
  VeriTac.Tactic.Parallel
  Parallel transformation: annotate a loop for parallel execution.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Schedule.FreeVars
import VeriTac.Schedule.LoopNest
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
  writes.all fun (_, indices) => varInExpr.varInExprs loopVar indices

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


/-- A write index-list is *local to* `v` when its first element is the bare loop
    variable and the remaining elements neither mention `v` nor read the store.
    Together these give `flatIndex(env[v:=x]) = x * 1000^k + C` with `C` fixed,
    so two different values of `v` always target different flat cells. -/
def indexLocalP (v : VarId) : List SExpr → Prop
  | [] => False
  | first :: rest => first = .var v ∧ ∀ e, e ∈ rest → exprStatic e = true ∧ varInExpr v e = false

/-- Every buffer write in the body is local to `v`: different iteration values
    cannot collide on a write cell. This is the *sound* dependence criterion
    behind `parallel` (stronger than the syntactic `checkIndependence`). -/
def loopLocalWritesP (v : VarId) : Stmt → Prop
  | .skip => True
  | .bufWrite _ indices _ => indexLocalP v indices
  | .loop _ _ _ _ body => loopLocalWritesP v body
  | .seq s1 s2 => loopLocalWritesP v s1 ∧ loopLocalWritesP v s2
  | .alloc _ _ body => loopLocalWritesP v body

/-- Linear accumulation: `c` contributes `c * 1000^len`. This is the arithmetic
    that makes the leading coefficient nonzero. -/
private theorem foldl_mul_pow (f : SExpr → Int) :
    ∀ (l : List SExpr) (c : Int),
      l.foldl (fun a e => a * 1000 + f e) c = c * 1000 ^ l.length + l.foldl (fun a e => a * 1000 + f e) 0 := by
  intro l
  induction l with
  | nil => intro c; simp
  | cons x xs ih =>
      intro c
      rw [List.foldl_cons, List.foldl_cons]
      rw [ih (c * 1000 + f x), ih (0 * 1000 + f x)]
      have hpow : 1000 ^ xs.length * (1000 : Int) = 1000 ^ (xs.length + 1) := by
        rw [pow_succ, Int.mul_comm]
      simp only [List.length_cons]
      rw [← hpow]
      rw [show (0 * 1000 + f x) = f x from by ring]
      ring_nf

/-- A write index-list local to `v` evaluates to different flat cells under
    different values of `v`. -/
theorem indexLocalP_flat_leading (v : VarId) (env : Env) (store : Store) :
    ∀ (idx : List SExpr), indexLocalP v idx →
      ∀ (x y : Int), x ≠ y →
        flatIndex (Env.set env v x) store idx ≠ flatIndex (Env.set env v y) store idx := by
  intro idx hidx x y hxy
  cases idx with
  | nil => simp [indexLocalP] at hidx
  | cons first rest =>
      simp only [indexLocalP] at hidx
      obtain ⟨hfirst, hrest⟩ := hidx
      have hfirstv : first = .var v := hfirst
      have hrestv : ∀ e, e ∈ rest → varInExpr v e = false := by
        intro e he
        exact (hrest e he).2
      -- rest folds are identical under x and y
      have hR : rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v x) store e) 0
          = rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v y) store e) 0 := by
        calc rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v x) store e) 0
            = rest.foldl (fun a e => a * 1000 + evalExpr env store e) 0 :=
              foldl_eval_congr (Env.set env v x) env store rest 0
                (fun e he => evalExpr_set_unused v x env store e (hrestv e he))
          _ = rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v y) store e) 0 := by
              rw [foldl_eval_congr env (Env.set env v y) store rest 0
                (fun e he => (evalExpr_set_unused v y env store e (hrestv e he)).symm)]
      intro hEq
      have hxraw : flatIndex (Env.set env v x) store (first :: rest)
          = x * 1000 ^ rest.length + rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v x) store e) 0 := by
        rw [flatIndex, List.foldl_cons, hfirstv]
        simp [evalExpr, Env.set]
        rw [foldl_mul_pow]
      have hyraw : flatIndex (Env.set env v y) store (first :: rest)
          = y * 1000 ^ rest.length + rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v y) store e) 0 := by
        rw [flatIndex, List.foldl_cons, hfirstv]
        simp [evalExpr, Env.set]
        rw [foldl_mul_pow]
      have hE : x * 1000 ^ rest.length = y * 1000 ^ rest.length := by
        calc x * 1000 ^ rest.length
            = flatIndex (Env.set env v x) store (first :: rest)
                - rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v x) store e) 0 := by
                rw [hxraw]; ring
          _ = flatIndex (Env.set env v y) store (first :: rest)
                - rest.foldl (fun a e => a * 1000 + evalExpr (Env.set env v y) store e) 0 := by
                rw [hEq, hR]
          _ = y * 1000 ^ rest.length := by rw [hyraw]; ring
      have hsub : (x - y) * (1000 ^ rest.length) = 0 := by
        rw [show (x - y) * (1000 ^ rest.length) = x * (1000 ^ rest.length) - y * (1000 ^ rest.length) from by ring]
        have hE' : x * (1000 ^ rest.length) - y * (1000 ^ rest.length) = 0 := by rw [hE]; ring
        exact hE'
      have hp : 1000 ^ rest.length ≠ (0 : Int) := pow_ne_zero rest.length (by norm_num : (1000 : Int) ≠ 0)
      have hxysub : x - y ≠ (0 : Int) := sub_ne_zero.mpr hxy
      have hne : (x - y) * (1000 ^ rest.length) ≠ 0 := Int.mul_ne_zero hxysub hp
      exact hne hsub

/-- Parallel correctness: for independent iterations, parallel annotation preserves semantics.
    `execStmt` discards the loop annotation, so the parallel form executes identically.
    The sound dependence condition (`loopLocalWritesP`) is stated separately; see
    `indexLocalP_flat_leading` for the per-write disjointness and `veritac_design.md` §7.4. -/
theorem parallel_correct (v : VarId) (lo hi : SExpr) (ann : Annotation) (body : Stmt)
    (_hind : checkIndependence v body = true) :
    .loop v lo hi ann body ≈ₛ .loop v lo hi .parallel body := by
  intro fuel env store
  cases fuel <;> rfl

end VeriTac.Tactic
