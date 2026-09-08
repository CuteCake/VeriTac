/-
  VeriTac.Tactic.Tile
  Tile transformation: split a loop into outer/inner loops.

  Correctness rests on `execLoopIters_partition` (VeriTac.Schedule.LoopComposition)
  plus the general substitution lemma `substStmtVar_correct`. The tactic is sound
  when the loop body never re-binds the target variable, the generated names
  `_outer` / `_inner` are fresh (neither loop-bound nor free-occurring in the
  body), and the tile size divides the loop extent.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Schedule.FreeVars
import VeriTac.Schedule.ExecLemmas
import VeriTac.Schedule.LoopComposition

namespace VeriTac.Tactic

/-- Substitute a variable in an expression with a replacement expression. -/
def substExprVar (v : VarId) (replacement : SExpr) : SExpr → SExpr
  | .lit n => .lit n
  | .var x => if x = v then replacement else .var x
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
    if lv = v then .loop lv lo hi ann body  -- shadowed
    else .loop lv (substExprVar v replacement lo) (substExprVar v replacement hi)
                  ann (substStmtVar v replacement body)
  | .seq s1 s2 => .seq (substStmtVar v replacement s1) (substStmtVar v replacement s2)
  | .alloc buf shape body =>
    .alloc buf (shape.map (substExprVar v replacement)) (substStmtVar v replacement body)

/-- Folding an index list under an env that binds `v` equals folding the
    substituted index list under the original env. -/
private theorem evalIdxList_subst (v : VarId) (rep : SExpr) (env : Env) (store : Store) :
    ∀ (l : List SExpr) (acc : Int),
      (∀ e, e ∈ l →
        evalExpr (Env.set env v (evalExpr env store rep)) store e =
        evalExpr env store (substExprVar v rep e)) →
      l.foldl (fun a idx => a * 1000 + evalExpr (Env.set env v (evalExpr env store rep)) store idx) acc =
      (l.map (substExprVar v rep)).foldl (fun a idx => a * 1000 + evalExpr env store idx) acc := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons x xs ih =>
      intro acc h
      simp only [List.foldl_cons, List.map_cons, List.foldl_cons]
      have hx := h x (by simp)
      have hbase : acc * 1000 + evalExpr (Env.set env v (evalExpr env store rep)) store x
                     = acc * 1000 + evalExpr env store (substExprVar v rep x) := by
        rw [hx]
      have htail : ∀ e, e ∈ xs →
          evalExpr (Env.set env v (evalExpr env store rep)) store e =
          evalExpr env store (substExprVar v rep e) := by
        intro e he; exact h e (by simp [he])
      have ih' := ih (acc * 1000 + evalExpr env store (substExprVar v rep x)) htail
      rw [hbase]
      exact ih'

/-- Expression-substitution lemma: substituting `v ↦ rep` and evaluating under an
    env that binds `v` to `evalExpr env store rep` is the same as evaluating the
    substituted expression under `env`. -/
theorem substExprVar_eval (v : VarId) (rep : SExpr) (env : Env) (store : Store) :
    ∀ e : SExpr,
      evalExpr (Env.set env v (evalExpr env store rep)) store e =
      evalExpr env store (substExprVar v rep e) := by
  apply SExpr_induct
  · intro n; simp [evalExpr, substExprVar]
  · intro x
    by_cases hxv : x = v
    · subst x
      simp [evalExpr, substExprVar, Env.set]
    · simp [evalExpr, substExprVar, Env.set, hxv]
  · intro a b hia hib; simp [evalExpr, substExprVar, hia, hib]
  · intro a b hia hib; simp [evalExpr, substExprVar, hia, hib]
  · intro a b hia hib; simp [evalExpr, substExprVar, hia, hib]
  · intro a b hia hib; simp [evalExpr, substExprVar, hia, hib]
  · intro buf indices hIdx
    simp only [evalExpr, substExprVar]
    congr 1
    exact evalIdxList_subst v rep env store indices 0 hIdx

/-- Statement-substitution lemma for a *literal* replacement.
    Substituting `v ↦ lit k` throughout `s` (which never re-binds `v`) is equivalent
    to executing `s` with `v` bound to `k`. -/
theorem substStmtVar_lit_correct (v : VarId) (k : Int) (store : Store) :
    ∀ (fuel : Nat) (env : Env) (s : Stmt), loopBinds v s = false →
      execStmt fuel (Env.set env v k) store s =
      execStmt fuel env store (substStmtVar v (.lit k) s) := by
  intro fuel env s
  induction s generalizing fuel env store with
  | skip => intro _; rfl
  | bufWrite buf indices val =>
      intro h
      have hVal : evalExpr (Env.set env v k) store val =
          evalExpr env store (substExprVar v (.lit k) val) := by
        simpa [evalExpr] using substExprVar_eval v (.lit k) env store val
      have hIdx : flatIndex (Env.set env v k) store indices =
          flatIndex env store (indices.map (substExprVar v (.lit k))) := by
        simpa [evalExpr, flatIndex] using evalIdxList_subst v (.lit k) env store indices 0
          (fun e _ => substExprVar_eval v (.lit k) env store e)
      simp only [execStmt, substStmtVar]
      congr 1
      funext b i
      rw [hVal, hIdx]
  | loop lv lo hi ann b ih =>
      intro h
      by_cases hlv : lv = v
      · subst lv
        exfalso
        simpa [loopBinds] using h
      · have hb : loopBinds v b = false := by
          simp [loopBinds, hlv] at h
          exact h
        have hlo : evalExpr (Env.set env v k) store lo =
            evalExpr env store (substExprVar v (.lit k) lo) := by
          simpa [evalExpr] using substExprVar_eval v (.lit k) env store lo
        have hhi : evalExpr (Env.set env v k) store hi =
            evalExpr env store (substExprVar v (.lit k) hi) := by
          simpa [evalExpr] using substExprVar_eval v (.lit k) env store hi
        simp [execStmt, substStmtVar, hlv]
        rw [hlo, hhi]
        refine execLoopIters_subst
          (f1 := fun env' st => execStmt fuel env' st b)
          (f2 := fun env' st => execStmt fuel env' st (substStmtVar v (.lit k) b))
          env v lv k _ _ store hlv ?hbody
        · intro env' st
          exact ih st fuel env' hb
  | seq s1 s2 ih1 ih2 =>
      intro h
      have h1 : loopBinds v s1 = false := by
        by_cases hs : loopBinds v s1 = true
        · have ht : (loopBinds v s1 || loopBinds v s2) = true := by
            rw [hs]; simp
          have hf : (loopBinds v s1 || loopBinds v s2) = false := by
            simpa [loopBinds] using h
          rw [ht] at hf
          contradiction
        · exact Bool.eq_false_iff.mpr hs
      have h2 : loopBinds v s2 = false := by
        by_cases hs : loopBinds v s2 = true
        · have ht : (loopBinds v s1 || loopBinds v s2) = true := by
            rw [hs]; simp
          have hf : (loopBinds v s1 || loopBinds v s2) = false := by
            simpa [loopBinds] using h
          rw [ht] at hf
          contradiction
        · exact Bool.eq_false_iff.mpr hs
      simp only [execStmt, substStmtVar]
      rw [ih1 store fuel env h1]
      congr 1
      funext st
      exact ih2 st fuel env h2
  | alloc buf shape body ih =>
      intro h
      have hb : loopBinds v body = false := by
        simpa [loopBinds] using h
      simp only [execStmt, substStmtVar]
      exact ih store fuel env hb

/-! ## General statement substitution -/

/-- **General substitution lemma.** Substituting `v ↦ rep` throughout `s` is
    equivalent to executing `s` with `v` bound to `evalExpr env store rep`,
    provided (1) no loop inside `s` re-binds `v`, and (2) no variable mentioned
    in `rep` is loop-bound inside `s` (so that the replacement's value is stable
    across the loop's own iterations). -/
theorem substStmtVar_correct (v : VarId) (rep : SExpr) :
    ∀ (fuel : Nat) (env : Env) (store : Store) (s : Stmt),
      loopBinds v s = false →
      (∀ x, varInExpr x rep = true → loopBinds x s = false) →
      exprStatic rep = true →
      execStmt fuel (Env.set env v (evalExpr env store rep)) store s =
      execStmt fuel env store (substStmtVar v rep s) := by
  intro fuel env store s
  induction s generalizing env store with
  | skip => intro _ _ _; rfl
  | bufWrite buf indices val =>
      intro _ _ _
      have hVal : evalExpr (Env.set env v (evalExpr env store rep)) store val =
          evalExpr env store (substExprVar v rep val) :=
        substExprVar_eval v rep env store val
      have hIdx : flatIndex (Env.set env v (evalExpr env store rep)) store indices =
          flatIndex env store (indices.map (substExprVar v rep)) := by
        simpa [evalExpr, flatIndex] using evalIdxList_subst v rep env store indices 0
          (fun e _ => substExprVar_eval v rep env store e)
      simp only [execStmt, substStmtVar]
      congr 1
      funext b i
      rw [hVal, hIdx]
  | loop lv lo hi ann body ih =>
      intro hb hrep hstatic
      obtain ⟨hlv, hbody⟩ : ¬(lv = v) ∧ loopBinds v body = false := by
        simpa [loopBinds] using hb
      have hlvne : lv ≠ v := hlv
      have hrepbody : ∀ x, varInExpr x rep = true → loopBinds x body = false := by
        intro x hx
        have hx2 : loopBinds x (.loop lv lo hi ann body) = false := hrep x hx
        simp only [loopBinds, Bool.or_eq_false_iff] at hx2
        exact hx2.2
      have hlvrep : varInExpr lv rep = false := by
        by_cases h : varInExpr lv rep = true
        · exfalso
          have hlb : loopBinds lv (.loop lv lo hi ann body) = false := hrep lv h
          simp [loopBinds] at hlb
        · exact Bool.eq_false_iff.mpr h
      simp only [execStmt]
      rw [substStmtVar, if_neg hlvne, substExprVar_eval v rep env store lo,
        substExprVar_eval v rep env store hi]
      refine execLoopIters_congr
        (f1 := fun e' st' => execStmt fuel e' st' body)
        (f2 := fun e' st' => execStmt fuel e' st' (substStmtVar v rep body))
        (Env.set env v (evalExpr env store rep)) env lv ?_ _ _ store
      intro c st
      show execStmt fuel (Env.set (Env.set env v (evalExpr env store rep)) lv c) st body =
           execStmt fuel (Env.set env lv c) st (substStmtVar v rep body)
      rw [Env.set_set_comm (Ne.symm hlvne)]
      rw [show evalExpr env store rep = evalExpr (Env.set env lv c) st rep from (
        calc evalExpr env store rep = evalExpr env st rep :=
              exprStatic_storeIrrel env store st rep hstatic
          _ = evalExpr (Env.set env lv c) st rep :=
              (evalExpr_set_unused lv c env st rep hlvrep).symm)]
      exact ih (Env.set env lv c) st hbody hrepbody hstatic
  | seq s1 s2 ih1 ih2 =>
      intro hb hrep hstatic
      obtain ⟨h1, h2⟩ : loopBinds v s1 = false ∧ loopBinds v s2 = false := by
        simpa [loopBinds] using hb
      have hrep1 : ∀ x, varInExpr x rep = true → loopBinds x s1 = false := by
        intro x hx
        have hx2 : loopBinds x (.seq s1 s2) = false := hrep x hx
        simp only [loopBinds, Bool.or_eq_false_iff] at hx2
        exact hx2.1
      have hrep2 : ∀ x, varInExpr x rep = true → loopBinds x s2 = false := by
        intro x hx
        have hx2 : loopBinds x (.seq s1 s2) = false := hrep x hx
        simp only [loopBinds, Bool.or_eq_false_iff] at hx2
        exact hx2.2
      simp only [execStmt, substStmtVar]
      rw [ih1 env store h1 hrep1 hstatic]
      congr 1
      funext st
      rw [show evalExpr env store rep = evalExpr env st rep from
        exprStatic_storeIrrel env store st rep hstatic]
      exact ih2 env st h2 hrep2 hstatic
  | alloc buf shape body ih =>
      intro hb hrep hstatic
      simp only [execStmt, substStmtVar]
      exact ih env store hb hrep hstatic

/-- Substituting an expression does not change which variables are loop-bound
    (an expression replacement never introduces a binder). -/
theorem loopBinds_subst (x v : VarId) (rep : SExpr) (s : Stmt) :
    loopBinds x (substStmtVar v rep s) = loopBinds x s := by
  induction s with
  | skip => rfl
  | bufWrite buf indices val => rfl
  | loop lv lo hi ann body ih =>
      simp only [loopBinds, substStmtVar]
      by_cases hlv : lv = v
      · subst lv; rw [if_pos rfl]; rfl
      · rw [if_neg hlv, loopBinds, ih]
  | seq s1 s2 ih1 ih2 =>
      simp only [loopBinds, substStmtVar]
      rw [ih1, ih2]
  | alloc buf shape body ih =>
      simp only [loopBinds, substStmtVar]
      exact ih

/-- Apply tile to a single loop already identified as the target.
    Soundness requirement: the loop runs over a literal range `[0, hiVal)` with
    `tileSize > 0` and `tileSize` *dividing* `hiVal`. Tiling a range that is not an
    exact multiple of the tile size would make the final block overrun `[0, hiVal)`,
    so we refuse those cases (`none`). Under the divisibility guard the outer
    loop's upper bound is exactly `hiVal / tileSize`. -/
private def tileAt (v : VarId) (lo hi : SExpr) (ann : Annotation) (body : Stmt)
    (tileSize : Nat) : Option Stmt :=
  match lo, hi with
  | .lit loVal, .lit hiVal =>
    if tileSize > 0 && loVal == 0 && hiVal >= 0 && (hiVal % (tileSize : Int)) == 0 then
      let outerVar := v ++ "_outer"
      let innerVar := v ++ "_inner"
      let ts := SExpr.lit tileSize
      let outerHi := SExpr.lit (hiVal / (tileSize : Int))
      let replacement := SExpr.add (.mul (.var outerVar) ts) (.var innerVar)
      let newBody := substStmtVar v replacement body
      some (.loop outerVar lo outerHi ann
        (.loop innerVar (.lit 0) (.lit tileSize) .none newBody))
    else none
  | _, _ => none

/-- Apply the tile transformation to a loop statement.
    Soundness requirement: `tileSize > 0` and, for the target loop over a literal
    range starting at 0, `tileSize` must divide the extent. Otherwise `none`. -/
def tile (stmt : Stmt) (targetVar : VarId) (tileSize : Nat) : Option Stmt :=
  match stmt with
  | .loop v lo hi ann body =>
    if v = targetVar then tileAt v lo hi ann body tileSize
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

/-- Core tiling correctness: the flat loop equals the explicit tiled nest. -/
theorem tile_correct' (v : VarId) (N ts : Nat) (body : Stmt) (ann : Annotation)
    (hts : ts > 0) (hdiv : (N : Int) % (ts : Int) = 0)
    (hb : loopBinds v body = false)
    (hb1 : loopBinds (v ++ "_outer") body = false)
    (hb2 : loopBinds (v ++ "_inner") body = false)
    (hf1 : varFreeStmt (v ++ "_outer") body = false)
    (hf2 : varFreeStmt (v ++ "_inner") body = false)
    (fuel : Nat) (env : Env) (store : Store) :
    execStmt fuel env store (.loop v (.lit 0) (.lit N) ann body) =
    execStmt fuel env store
      (.loop (v ++ "_outer") (.lit 0) (.lit ((N : Int) / (ts : Int))) ann
        (.loop (v ++ "_inner") (.lit 0) (.lit ts) .none
          (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts))
            (.var (v ++ "_inner"))) body))) := by
  -- arithmetic facts
  have hNatMod : N % ts = 0 := by
    have h2 : ((N % ts : Nat) : Int) = 0 := by rw [Int.natCast_mod]; exact hdiv
    exact (Int.natCast_inj.mp (by rw [Int.natCast_zero]; exact h2))
  have hmulN : ts * (N / ts) = N := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hNatMod)
  -- naming facts
  have hso : ("_outer" : String) ≠ "_inner" := by decide
  have hovi : v ++ "_outer" ≠ v ++ "_inner" := fun h => hso (String.append_right_inj v |>.mp h)
  have h6o : ("_outer" : String).length = 6 := by decide
  have h6i : ("_inner" : String).length = 6 := by decide
  have hovv : v ++ "_outer" ≠ v := by
    intro h
    have hl : String.length (v ++ "_outer") = String.length v := by rw [h]
    rw [String.length_append, h6o] at hl
    omega
  have hivv : v ++ "_inner" ≠ v := by
    intro h
    have hl : String.length (v ++ "_inner") = String.length v := by rw [h]
    rw [String.length_append, h6i] at hl
    omega
  -- the replacement expression and its freshness obligations
  have hrepBody : ∀ x : VarId,
      varInExpr x (.add (.mul (.var (v ++ "_outer")) (.lit ts)) (.var (v ++ "_inner"))) = true →
      loopBinds x body = false := by
    intro x hx
    by_cases hx1 : x = v ++ "_outer"
    · rw [hx1] at hx ⊢
      exact hb1
    · by_cases hx2 : x = v ++ "_inner"
      · rw [hx2] at hx ⊢
        exact hb2
      · exfalso
        have h0 : varInExpr x (.add (.mul (.var (v ++ "_outer")) (.lit ts))
              (.var (v ++ "_inner"))) = false := by
          simp [varInExpr, eq_comm, hx1, hx2]
        rw [h0] at hx
        simp at hx
  have hstatic : exprStatic
        (.add (.mul (.var (v ++ "_outer")) (.lit ts)) (.var (v ++ "_inner"))) = true := by
    simp [exprStatic]
  -- count normalizations
  have hIL : (((N : Int) - 0).toNat) = N := by
    rw [show ((N : Int) - 0) = ((N : Int)) from by ring, Int.toNat_natCast]
  have hIO : ((((N : Int)) / (ts : Int)) - 0).toNat = N / ts := by
    rw [show (((N : Int)) / (ts : Int) - 0) = (((N : Int)) / (ts : Int)) from by ring,
      ← Int.natCast_div, Int.toNat_natCast]
  have hIt : (((ts : Int) - 0).toNat) = ts := by
    rw [show ((ts : Int) - 0) = ((ts : Int)) from by ring, Int.toNat_natCast]
  -- unfold both sides to folds
  have hL : execStmt fuel env store (.loop v (.lit 0) (.lit N) ann body)
      = execLoopIters (fun e' st' => execStmt fuel e' st' body) env v 0 N store := by
    simp only [execStmt, evalExpr, hIL]
  have hR : execStmt fuel env store
        (.loop (v ++ "_outer") (.lit 0) (.lit ((N : Int) / (ts : Int))) ann
          (.loop (v ++ "_inner") (.lit 0) (.lit ts) .none
            (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts))
              (.var (v ++ "_inner"))) body)))
      = execLoopIters
          (fun e' st' => execLoopIters
            (fun e'' st'' => execStmt fuel e'' st''
              (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts))
                (.var (v ++ "_inner"))) body)) e' (v ++ "_inner") 0 ts st')
          env (v ++ "_outer") 0 (N / ts) store := by
    simp only [execStmt, evalExpr, hIO, hIt]
  rw [hL, hR]
  -- per-iteration agreement: inner iteration j of outer block q is flat iteration q*ts + j
  have hstep : ∀ (q j : Int) (st : Store),
      execStmt fuel (Env.set (Env.set env (v ++ "_outer") q) (v ++ "_inner") j) st
          (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts))
            (.var (v ++ "_inner"))) body)
      = execStmt fuel (Env.set env v (q * (ts : Int) + j)) st body := by
    intro q j st
    have hEv : evalExpr (Env.set (Env.set env (v ++ "_outer") q) (v ++ "_inner") j) st
          (.add (.mul (.var (v ++ "_outer")) (.lit ts)) (.var (v ++ "_inner")))
        = q * (ts : Int) + j := by
      simp [evalExpr, Env.set, hovi, hovv, hivv, Ne.symm hovi, Ne.symm hovv, Ne.symm hivv]
    rw [← substStmtVar_correct v _ fuel (Env.set (Env.set env (v ++ "_outer") q)
      (v ++ "_inner") j) st body hb hrepBody hstatic, hEv]
    -- peel the generated bindings: first `_outer`, then `_inner`
    have hpeel1 : execStmt fuel
          (Env.set (Env.set (Env.set env (v ++ "_outer") q) (v ++ "_inner") j) v (q * (ts : Int) + j)) st body
        = execStmt fuel (Env.set (Env.set env (v ++ "_inner") j) v (q * (ts : Int) + j)) st body := by
      refine execStmt_agree_except fuel (v ++ "_outer") body hf1
        (Env.set (Env.set (Env.set env (v ++ "_outer") q) (v ++ "_inner") j) v (q * (ts : Int) + j))
        (Env.set (Env.set env (v ++ "_inner") j) v (q * (ts : Int) + j)) st ?_
      intro z hz
      by_cases hzv : z = v
      · subst z; simp [Env.set]
      · by_cases hzi : z = v ++ "_inner"
        · subst z; simp [Env.set, hzv]
        · simp [Env.set, hzv, hzi, hz]
    have hpeel2 : execStmt fuel (Env.set (Env.set env (v ++ "_inner") j) v (q * (ts : Int) + j)) st body
        = execStmt fuel (Env.set env v (q * (ts : Int) + j)) st body := by
      refine execStmt_agree_except fuel (v ++ "_inner") body hf2
        (Env.set (Env.set env (v ++ "_inner") j) v (q * (ts : Int) + j))
        (Env.set env v (q * (ts : Int) + j)) st ?_
      intro z hz
      by_cases hzv : z = v
      · subst z; simp [Env.set]
      · simp [Env.set, hzv, hz]
    rw [hpeel1, hpeel2]
  -- partition the flat fold into blocks
  rw [execLoopIters_partition
    (F := fun (e' : Env) (st' : Store) => execStmt fuel e' st'
      (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts)) (.var (v ++ "_inner"))) body))
    (f := fun (e' : Env) (st' : Store) => execStmt fuel e' st' body)
    env v (v ++ "_outer") (v ++ "_inner") ts hstep (N / ts) 0 store]
  rw [show ((0 : Int) * (ts : Int)) = 0 from by ring, hmulN]

/-- Tile correctness theorem: tiling a loop `[0, N)` by a divisor `ts` of `N`
    preserves semantics, provided the generated names are fresh for the body. -/
theorem tile_correct (v : VarId) (N ts : Nat) (hts : ts > 0) (body : Stmt)
    (ann : Annotation)
    (hdiv : (N : Int) % (ts : Int) = 0)
    (hb : loopBinds v body = false)
    (hb1 : loopBinds (v ++ "_outer") body = false)
    (hb2 : loopBinds (v ++ "_inner") body = false)
    (hf1 : varFreeStmt (v ++ "_outer") body = false)
    (hf2 : varFreeStmt (v ++ "_inner") body = false) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      execStmt fuel env store (.loop v (.lit 0) (.lit N) ann body) =
      execStmt fuel env store
        (match tile (.loop v (.lit 0) (.lit N) ann body) v ts with
         | some s => s
         | none => .loop v (.lit 0) (.lit N) ann body) := by
  intro fuel env store
  have hg : (ts > 0 && (0 : Int) == 0 && (N : Int) >= 0 && ((N : Int) % (ts : Int)) == 0) = true := by
    simp [hts, hdiv]
  have htile : tile (.loop v (.lit 0) (.lit N) ann body) v ts =
      some (.loop (v ++ "_outer") (.lit 0) (.lit ((N : Int) / (ts : Int))) ann
        (.loop (v ++ "_inner") (.lit 0) (.lit ts) .none
          (substStmtVar v (.add (.mul (.var (v ++ "_outer")) (.lit ts))
            (.var (v ++ "_inner"))) body))) := by
    simp only [tile, tileAt, hg, if_true, eq_self_iff_true, if_true]
  rw [htile]
  exact tile_correct' v N ts body ann hts hdiv hb hb1 hb2 hf1 hf2 fuel env store

end VeriTac.Tactic
