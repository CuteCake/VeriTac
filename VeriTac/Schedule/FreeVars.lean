/-
  VeriTac.Schedule.FreeVars
  Free-variable analysis for the scheduled IR, plus the lemmas connecting
  "variable not free" to invariance of evaluation.
-/
import Mathlib.Tactic
import VeriTac.Schedule.LoopNest

namespace VeriTac

/-! ## Manual induction principle for the nested inductive `SExpr` -/

/-- Induction principle for `SExpr`, which nests a `List SExpr` inside
    `bufRead` and therefore needs `SExpr.rec` with two motives. -/
theorem SExpr_induct {P : SExpr → Prop}
    (hlit : ∀ (n : Int), P (.lit n))
    (hvar : ∀ (v : VarId), P (.var v))
    (hadd : ∀ (a b : SExpr), P a → P b → P (.add a b))
    (hmul : ∀ (a b : SExpr), P a → P b → P (.mul a b))
    (hdiv : ∀ (a b : SExpr), P a → P b → P (.div a b))
    (hmod : ∀ (a b : SExpr), P a → P b → P (.mod a b))
    (hbuf : ∀ (buf : BufId) (indices : List SExpr),
      (∀ e, e ∈ indices → P e) → P (.bufRead buf indices)) :
    ∀ e : SExpr, P e := by
  intro e
  exact SExpr.rec (motive_1 := P) (motive_2 := fun l => ∀ e, e ∈ l → P e)
    (lit := hlit) (var := hvar)
    (add := hadd) (mul := hmul) (div := hdiv) (mod := hmod)
    (bufRead := fun buf indices h => hbuf buf indices h)
    (nil := fun e h => False.elim (List.not_mem_nil h))
    (cons := fun head tail ph pt e h =>
      match h with
      | List.Mem.head _ => ph
      | List.Mem.tail _ ht => pt e ht)
    e

/-! ## Free variables -/

/-- Does the variable appear free in an expression? -/
def varInExpr (v : VarId) (e : SExpr) : Bool :=
  match e with
  | .lit _ => false
  | .var x => x = v
  | .add a b => varInExpr v a || varInExpr v b
  | .mul a b => varInExpr v a || varInExpr v b
  | .div a b => varInExpr v a || varInExpr v b
  | .mod a b => varInExpr v a || varInExpr v b
  | .bufRead _ indices => varInExprs v indices
where
  varInExprs (v : VarId) : List SExpr → Bool
    | [] => false
    | e :: es => varInExpr v e || varInExprs v es

/-- Is the buffer only read (never written) in the statement? -/
def isReadOnly (buf : BufId) : Stmt → Bool
  | .skip => true
  | .bufWrite b _ _ => b != buf
  | .loop _ _ _ _ body => isReadOnly buf body
  | .seq s1 s2 => isReadOnly buf s1 && isReadOnly buf s2
  | .alloc _ _ body => isReadOnly buf body

/-- Membership fact: `varInExprs w l = false` means no element mentions `w`. -/
theorem varInExprs_false (w : VarId) :
    ∀ (l : List SExpr), varInExpr.varInExprs w l = false → ∀ e, e ∈ l → varInExpr w e = false := by
  intro l
  induction l with
  | nil => intro _ e he; exact absurd he (by simp)
  | cons i is ih =>
      intro h e he
      simp only [varInExpr.varInExprs, Bool.or_eq_false_iff] at h
      rcases List.mem_cons.mp he with rfl | he'
      · exact h.1
      · exact ih (by simp [h.2]) e he'

/-- Does the variable occur free in a statement, in the sense that *binding it
    in the environment can change the statement's execution*? Loop bounds are
    always evaluated in the ambient environment (the loop's own variable is
    bound only for its body), so they always count. Occurrences of `v` inside
    the body of `loop v ...` are shadowed and do not count. -/
def varFreeStmt (v : VarId) : Stmt → Bool
  | .skip => false
  | .bufWrite _ indices val =>
    varInExpr.varInExprs v indices || varInExpr v val
  | .loop lv lo hi _ body =>
    varInExpr v lo || varInExpr v hi ||
      (if lv = v then false else varFreeStmt v body)
  | .seq s1 s2 => varFreeStmt v s1 || varFreeStmt v s2
  | .alloc _ shape body => varInExpr.varInExprs v shape || varFreeStmt v body

/-- Does any loop inside `s` bind the variable `v`? `false` means substituting
    `v` never collides with a binder of the same name. -/
def loopBinds (v : VarId) : Stmt → Bool
  | .loop lv _ _ _ body => lv = v || loopBinds v body
  | .seq s1 s2 => loopBinds v s1 || loopBinds v s2
  | .alloc _ _ body => loopBinds v body
  | _ => false

/-! ## Expression evaluation is invariant under unused bindings -/

theorem foldl_eval_congr (env1 env2 : Env) (store : Store) :
    ∀ (l : List SExpr) (acc : Int),
      (∀ e, e ∈ l → evalExpr env1 store e = evalExpr env2 store e) →
      l.foldl (fun a idx => a * 1000 + evalExpr env1 store idx) acc =
      l.foldl (fun a idx => a * 1000 + evalExpr env2 store idx) acc := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons i is ih =>
      intro acc h
      simp only [List.foldl_cons]
      rw [h i (by simp)]
      exact ih (acc * 1000 + evalExpr env2 store i) (fun e he => h e (by simp [he]))

theorem evalExpr_set_unused (w : VarId) (x : Int) (env : Env) (store : Store) :
    ∀ e : SExpr, varInExpr w e = false →
      evalExpr (Env.set env w x) store e = evalExpr env store e := by
  apply SExpr_induct
  · intro n; simp [evalExpr]
  · intro v
    by_cases hv : v = w
    · subst v; simp [varInExpr, evalExpr, Env.set]
    · simp [varInExpr, evalExpr, Env.set, hv]
  · intro a b ha hb hia
    simp only [varInExpr, Bool.or_eq_false_iff] at hia
    simp only [evalExpr]
    rw [ha hia.1, hb hia.2]
  · intro a b ha hb hia
    simp only [varInExpr, Bool.or_eq_false_iff] at hia
    simp only [evalExpr]
    rw [ha hia.1, hb hia.2]
  · intro a b ha hb hia
    simp only [varInExpr, Bool.or_eq_false_iff] at hia
    simp only [evalExpr]
    rw [ha hia.1, hb hia.2]
  · intro a b ha hb hia
    simp only [varInExpr, Bool.or_eq_false_iff] at hia
    simp only [evalExpr]
    rw [ha hia.1, hb hia.2]
  · intro buf indices hIdx hidx
    simp only [varInExpr] at hidx
    simp only [evalExpr]
    congr 1
    exact foldl_eval_congr _ _ store indices 0
      (fun e he => hIdx e he (varInExprs_false w indices hidx e he))

theorem foldl_eval_congr_store (env : Env) (st1 st2 : Store) :
    ∀ (l : List SExpr) (acc : Int), (∀ e, e ∈ l → evalExpr env st1 e = evalExpr env st2 e) →
      l.foldl (fun a idx => a * 1000 + evalExpr env st1 idx) acc =
      l.foldl (fun a idx => a * 1000 + evalExpr env st2 idx) acc := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons i is ih =>
      intro acc h
      rw [show (i :: is).foldl (fun a idx => a * 1000 + evalExpr env st1 idx) acc =
            is.foldl (fun a idx => a * 1000 + evalExpr env st1 idx)
              (acc * 1000 + evalExpr env st1 i) from rfl,
        h i (by simp),
        show (i :: is).foldl (fun a idx => a * 1000 + evalExpr env st2 idx) acc =
          is.foldl (fun a idx => a * 1000 + evalExpr env st2 idx)
            (acc * 1000 + evalExpr env st2 i) from rfl]
      exact ih (acc * 1000 + evalExpr env st2 i) (fun e he => h e (by simp [he]))

theorem flatIndex_set_unused (w : VarId) (x : Int) (env : Env) (store : Store)
    (indices : List SExpr) (h : ∀ e, e ∈ indices → varInExpr w e = false) :
    flatIndex (Env.set env w x) store indices = flatIndex env store indices :=
  foldl_eval_congr _ _ store indices 0
    (fun e he => evalExpr_set_unused w x env store e (h e he))

/-! ## Store-independent expressions -/

/-- `true` when the expression contains no buffer reads, so its evaluation is
    independent of the store. -/
def exprStatic : SExpr → Bool
  | .lit _ => true
  | .var _ => true
  | .add a b => exprStatic a && exprStatic b
  | .mul a b => exprStatic a && exprStatic b
  | .div a b => exprStatic a && exprStatic b
  | .mod a b => exprStatic a && exprStatic b
  | .bufRead _ _ => false

/-- Evaluating a store-independent expression gives the same value in any
    store. This is what lets a constant substitution survive store changes. -/
theorem exprStatic_storeIrrel (env : Env) (s1 s2 : Store) :
    ∀ e : SExpr, exprStatic e = true → evalExpr env s1 e = evalExpr env s2 e := by
  intro e
  cases e with
  | lit n => intro _; simp [evalExpr]
  | var v => intro _; simp [evalExpr]
  | add a b =>
      intro h
      simp only [exprStatic, Bool.and_eq_true] at h
      obtain ⟨ha, hb⟩ := h
      simp [evalExpr, exprStatic_storeIrrel env s1 s2 a ha,
        exprStatic_storeIrrel env s1 s2 b hb]
  | mul a b =>
      intro h
      simp only [exprStatic, Bool.and_eq_true] at h
      obtain ⟨ha, hb⟩ := h
      simp [evalExpr, exprStatic_storeIrrel env s1 s2 a ha,
        exprStatic_storeIrrel env s1 s2 b hb]
  | div a b =>
      intro h
      simp only [exprStatic, Bool.and_eq_true] at h
      obtain ⟨ha, hb⟩ := h
      have hva := exprStatic_storeIrrel env s1 s2 a ha
      have hvb := exprStatic_storeIrrel env s1 s2 b hb
      simp [evalExpr, hva, hvb]
  | mod a b =>
      intro h
      simp only [exprStatic, Bool.and_eq_true] at h
      obtain ⟨ha, hb⟩ := h
      have hva := exprStatic_storeIrrel env s1 s2 a ha
      have hvb := exprStatic_storeIrrel env s1 s2 b hb
      simp [evalExpr, hva, hvb]
  | bufRead buf idx =>
      intro h
      simp only [exprStatic] at h
      exact absurd h (by simp)

end VeriTac
