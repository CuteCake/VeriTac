/-
  VeriTac.Schedule.ExecLemmas
  Execution-level lemmas: totality of `execStmt`, invariance under unused
  environment bindings, agreement-based congruence, read-only store invariance,
  write-set discipline, and blindness to untouched buffers.
-/
import Mathlib.Tactic
import VeriTac.Schedule.LoopComposition
import VeriTac.Schedule.FreeVars

namespace VeriTac

/-! ## Totality -/

theorem execLoopIters_total (f : Env → Store → Option Store)
    (hf : ∀ (e : Env) (st : Store), ∃ st', f e st = some st') :
    ∀ (env : Env) (v : VarId) (lo : Int) (n : Nat) (store : Store),
      ∃ store', execLoopIters f env v lo n store = some store' := by
  intro env v lo n store
  induction n generalizing lo store with
  | zero => exact ⟨store, rfl⟩
  | succ n ih =>
      obtain ⟨st', hf'⟩ := hf (Env.set env v lo) store
      obtain ⟨st'', hres⟩ := ih (lo + 1) st'
      refine ⟨st'', ?_⟩
      simp [execLoopIters, hf', hres]

theorem execStmt_total (fuel : Nat) :
    ∀ (env : Env) (store : Store) (s : Stmt),
      ∃ store', execStmt fuel env store s = some store' := by
  intro env store s
  induction s generalizing env store with
  | skip => exact ⟨store, rfl⟩
  | bufWrite _ _ _ => exact ⟨_, rfl⟩
  | loop v lo hi _ body ih =>
      simp only [execStmt]
      exact execLoopIters_total _ (fun e st => ih e st) env v _ _ store
  | seq s1 s2 ih1 ih2 =>
      obtain ⟨st1, h1⟩ := ih1 env store
      obtain ⟨st2, h2⟩ := ih2 env st1
      refine ⟨st2, ?_⟩
      simp [execStmt, h1, h2]
  | alloc _ _ body ih => exact ih env store

/-! ## Invariance under unused environment bindings -/

/-- The loop body's execution is insensitive to an ambient `w`-binding, so the
    fold over `lv` is too. -/
theorem execLoopIters_set_unused (fuel : Nat) (w : VarId) (x : Int) (body : Stmt)
    (hb : ∀ (env' : Env) (store' : Store),
      execStmt fuel (Env.set env' w x) store' body = execStmt fuel env' store' body)
    (lv : VarId) (env : Env) :
    ∀ (n : Nat) (lo : Int) (st : Store),
      execLoopIters (fun e' st' => execStmt fuel e' st' body) (Env.set env w x) lv lo n st =
      execLoopIters (fun e' st' => execStmt fuel e' st' body) env lv lo n st := by
  intro n lo st
  induction n generalizing lo st with
  | zero => rfl
  | succ n ih =>
      by_cases hlv : lv = w
      · subst lv
        exact execLoopIters_self_bind _ env w x (n + 1) lo st
      · simp only [execLoopIters]
        rw [Env.set_set_comm (Ne.symm hlv), hb (Env.set env lv lo) st]
        congr 1
        funext st'
        exact ih (lo + 1) st'

/-- Executing a statement is insensitive to environment bindings that the
    statement cannot observe (see `varFreeStmt`). -/
theorem execStmt_set_unused (fuel : Nat) (w : VarId) (x : Int) :
    ∀ (env : Env) (store : Store) (s : Stmt), varFreeStmt w s = false →
      execStmt fuel (Env.set env w x) store s = execStmt fuel env store s := by
  intro env store s
  induction s generalizing env store with
  | skip => intro _; rfl
  | bufWrite _ indices val =>
      intro h
      simp only [varFreeStmt, Bool.or_eq_false_iff] at h
      simp only [execStmt]
      congr 1
      funext b i
      rw [evalExpr_set_unused w x env store val h.2,
        flatIndex_set_unused w x env store indices (varInExprs_false w indices h.1)]
  | loop lv lo hi _ body ih =>
      intro h
      obtain ⟨⟨hlo, hhi⟩, hbody⟩ :
          (varInExpr w lo = false ∧ varInExpr w hi = false) ∧
            (¬(lv = w) → varFreeStmt w body = false) := by
        simpa [varFreeStmt] using h
      simp only [execStmt]
      rw [evalExpr_set_unused w x env store lo hlo,
        evalExpr_set_unused w x env store hi hhi]
      by_cases hlv : lv = w
      · subst lv
        exact execLoopIters_self_bind _ env w x _ _ _
      · refine execLoopIters_set_unused fuel w x body ?_ lv env _ _ _
        intro env' store'
        exact ih env' store' (hbody hlv)
  | seq s1 s2 ih1 ih2 =>
      intro h
      simp only [varFreeStmt, Bool.or_eq_false_iff] at h
      simp only [execStmt]
      rw [ih1 env store h.1]
      congr 1
      funext st
      exact ih2 env st h.2
  | alloc _ _ body ih =>
      intro h
      simp only [varFreeStmt, Bool.or_eq_false_iff] at h
      exact ih env store h.2

/-- Two environments that agree away from `w` give the same execution. -/
theorem execStmt_agree_except (fuel : Nat) (w : VarId) (s : Stmt)
    (h : varFreeStmt w s = false) :
    ∀ (env1 env2 : Env) (store : Store), (∀ z, z ≠ w → env1 z = env2 z) →
      execStmt fuel env1 store s = execStmt fuel env2 store s := by
  intro env1 env2 store hagree
  calc execStmt fuel env1 store s
      = execStmt fuel (Env.set env1 w (env2 w)) store s :=
        (execStmt_set_unused fuel w (env2 w) env1 store s h).symm
    _ = execStmt fuel env2 store s := by
        rw [show Env.set env1 w (env2 w) = env2 from by
          funext z
          by_cases hz : z = w
          · subst z; simp [Env.set]
          · simp [Env.set, hz, hagree z hz]]

/-! ## Read-only store invariance -/

theorem execLoopIters_ro (f : Env → Store → Option Store) (b : BufId)
    (hf : ∀ (e : Env) (st st' : Store), f e st = some st' → st' b = st b) :
    ∀ (env : Env) (v : VarId) (lo : Int) (n : Nat) (store store' : Store),
      execLoopIters f env v lo n store = some store' → store' b = store b := by
  intro env v lo n store
  induction n generalizing lo store with
  | zero => intro store' h; simp only [execLoopIters, Option.some.injEq] at h; subst h; rfl
  | succ n ih =>
      intro store' h
      obtain ⟨st1, h1, h2⟩ : ∃ st1, f (Env.set env v lo) store = some st1 ∧
          execLoopIters f env v (lo + 1) n st1 = some store' := by
        simpa [execLoopIters, Option.bind_eq_some_iff] using h
      rw [ih (lo + 1) st1 store' h2, hf _ _ _ h1]

/-- A read-only statement never changes buffer `b`. -/
theorem execStmt_ro (fuel : Nat) (b : BufId) :
    ∀ (env : Env) (store : Store) (s : Stmt), isReadOnly b s = true →
      ∀ store' : Store, execStmt fuel env store s = some store' → store' b = store b := by
  intro env store s
  induction s generalizing env store with
  | skip => intro _ store' h; simp only [execStmt, Option.some.injEq] at h; subst h; rfl
  | bufWrite c _ _ =>
      intro h store' hres
      simp only [isReadOnly, bne_iff_ne] at h
      simp only [execStmt, Option.some.injEq] at hres
      subst hres
      have hcb : (b == c) = false := beq_eq_false_iff_ne.mpr (Ne.symm h)
      funext i
      simp [hcb]
  | loop v lo hi _ body ih =>
      intro h store' hres
      simp only [isReadOnly] at h
      simp only [execStmt] at hres
      exact execLoopIters_ro _ b (fun e st st' hst' => ih e st h st' hst') env v _ _ store store' hres
  | seq s1 s2 ih1 ih2 =>
      intro h store' hres
      simp only [isReadOnly, Bool.and_eq_true] at h
      obtain ⟨st1, h1, h2⟩ : ∃ st1, execStmt fuel env store s1 = some st1 ∧
          execStmt fuel env st1 s2 = some store' := by
        simpa [execStmt, Option.bind_eq_some_iff] using hres
      rw [ih2 env st1 h.2 store' h2, ih1 env store h.1 st1 h1]
  | alloc _ _ body ih =>
      intro h store' hres
      exact ih env store (by simpa [isReadOnly] using h) store' hres

/-! ## Write-set discipline -/

/-- Every write inside the statement targets buffer `b`. -/
def writesOnlyBuf (b : BufId) : Stmt → Bool
  | .skip => true
  | .bufWrite c _ _ => c = b
  | .loop _ _ _ _ body => writesOnlyBuf b body
  | .seq s1 s2 => writesOnlyBuf b s1 && writesOnlyBuf b s2
  | .alloc _ _ body => writesOnlyBuf b body

theorem execLoopIters_writes_only (f : Env → Store → Option Store) (b : BufId)
    (hf : ∀ (e : Env) (st st' : Store), f e st = some st' →
      ∀ (c : BufId), c ≠ b → ∀ i, st' c i = st c i) :
    ∀ (env : Env) (v : VarId) (lo : Int) (n : Nat) (store store' : Store),
      execLoopIters f env v lo n store = some store' →
      ∀ (c : BufId), c ≠ b → ∀ i, store' c i = store c i := by
  intro env v lo n store
  induction n generalizing lo store with
  | zero =>
      intro store' h c hc i
      simp only [execLoopIters, Option.some.injEq] at h
      subst h
      rfl
  | succ n ih =>
      intro store' h c hc i
      obtain ⟨st1, h1, h2⟩ : ∃ st1, f (Env.set env v lo) store = some st1 ∧
          execLoopIters f env v (lo + 1) n st1 = some store' := by
        simpa [execLoopIters, Option.bind_eq_some_iff] using h
      rw [ih (lo + 1) st1 store' h2 c hc i, hf _ _ _ h1 c hc i]

/-- A statement whose writes all target `b` never changes any other buffer. -/
theorem execStmt_writes_only (fuel : Nat) (b : BufId) :
    ∀ (env : Env) (store : Store) (s : Stmt), writesOnlyBuf b s = true →
      ∀ store' : Store, execStmt fuel env store s = some store' →
        ∀ (c : BufId), c ≠ b → ∀ i, store' c i = store c i := by
  intro env store s
  induction s generalizing env store with
  | skip => intro _ store' h; simp only [execStmt, Option.some.injEq] at h; subst h; intro c _ i; rfl
  | bufWrite d _ _ =>
      intro h store' hres
      simp only [writesOnlyBuf] at h
      simp only [execStmt, Option.some.injEq] at hres
      subst hres
      intro c hc i
      have hcdb : c ≠ d := by
        intro h'
        exact hc (h'.trans (of_decide_eq_true h))
      have hcb : (c == d) = false := beq_eq_false_iff_ne.mpr hcdb
      simp [hcb]
  | loop v lo hi _ body ih =>
      intro h store' hres
      simp only [writesOnlyBuf] at h
      simp only [execStmt] at hres
      exact execLoopIters_writes_only _ b
        (fun e st st' hst' => ih e st h st' hst') env v _ _ store store' hres
  | seq s1 s2 ih1 ih2 =>
      intro h store' hres
      simp only [writesOnlyBuf, Bool.and_eq_true] at h
      obtain ⟨st1, h1, h2⟩ : ∃ st1, execStmt fuel env store s1 = some st1 ∧
          execStmt fuel env st1 s2 = some store' := by
        simpa [execStmt, Option.bind_eq_some_iff] using hres
      intro c hc i
      rw [ih2 env st1 h.2 store' h2 c hc i, ih1 env store h.1 st1 h1 c hc i]
  | alloc _ _ body ih =>
      intro h store' hres
      exact ih env store (by simpa [writesOnlyBuf] using h) store' hres

/-! ## Buffer reads -/

/-- Does the expression read buffer `w`? -/
def readsBuf (w : BufId) : SExpr → Bool
  | .lit _ => false
  | .var _ => false
  | .add a b => readsBuf w a || readsBuf w b
  | .mul a b => readsBuf w a || readsBuf w b
  | .div a b => readsBuf w a || readsBuf w b
  | .mod a b => readsBuf w a || readsBuf w b
  | .bufRead c indices => c = w || readsBufs w indices
where
  readsBufs (w : BufId) : List SExpr → Bool
    | [] => false
    | e :: es => readsBuf w e || readsBufs w es

theorem readsBufs_false (w : BufId) :
    ∀ (l : List SExpr), readsBuf.readsBufs w l = false → ∀ e, e ∈ l → readsBuf w e = false := by
  intro l
  induction l with
  | nil => intro _ e he; exact absurd he (by simp)
  | cons i is ih =>
      intro h e he
      simp only [readsBuf.readsBufs, Bool.or_eq_false_iff] at h
      rcases List.mem_cons.mp he with rfl | he'
      · exact h.1
      · exact ih (by simp [h.2]) e he'

/-- Does the statement read buffer `w`? -/
def stmtReadsBuf (w : BufId) : Stmt → Bool
  | .skip => false
  | .bufWrite _ indices val => readsBuf.readsBufs w indices || readsBuf w val
  | .loop _ lo hi _ body => readsBuf w lo || readsBuf w hi || stmtReadsBuf w body
  | .seq s1 s2 => stmtReadsBuf w s1 || stmtReadsBuf w s2
  | .alloc _ shape body => readsBuf.readsBufs w shape || stmtReadsBuf w body

/-- An expression that never reads `w` evaluates the same on two stores that
    agree away from `w`. -/
theorem evalExpr_blind (w : BufId) (env : Env) (s1 s2 : Store)
    (hst : ∀ (c : BufId) (i : Int), c ≠ w → s1 c i = s2 c i) :
    ∀ e : SExpr, readsBuf w e = false → evalExpr env s1 e = evalExpr env s2 e := by
  apply SExpr_induct
  · intro n; simp [evalExpr]
  · intro v; simp [evalExpr]
  · intro a b ha hb h
    simp only [readsBuf, Bool.or_eq_false_iff] at h
    simp only [evalExpr]
    rw [ha h.1, hb h.2]
  · intro a b ha hb h
    simp only [readsBuf, Bool.or_eq_false_iff] at h
    simp only [evalExpr]
    rw [ha h.1, hb h.2]
  · intro a b ha hb h
    simp only [readsBuf, Bool.or_eq_false_iff] at h
    simp only [evalExpr]
    rw [ha h.1, hb h.2]
  · intro a b ha hb h
    simp only [readsBuf, Bool.or_eq_false_iff] at h
    simp only [evalExpr]
    rw [ha h.1, hb h.2]
  · intro buf indices hIdx h
    simp only [readsBuf, Bool.or_eq_false_iff] at h
    have hbuf : buf ≠ w := by simpa using h.1
    simp only [evalExpr]
    rw [foldl_eval_congr_store env s1 s2 indices 0
      (fun e he => hIdx e he (readsBufs_false w indices h.2 e he))]
    exact hst buf _ hbuf

/-- The fold bodies produce results agreeing away from `w`, so the folds do. -/
theorem execLoopIters_blind (f : Env → Store → Option Store) (w : BufId)
    (hf : ∀ (e : Env) (st1 st2 : Store),
      (∀ (c : BufId) (i : Int), c ≠ w → st1 c i = st2 c i) →
      ∀ r1 r2, f e st1 = some r1 → f e st2 = some r2 →
        ∀ (c : BufId) (i : Int), c ≠ w → r1 c i = r2 c i)
    (env : Env) (v : VarId) :
    ∀ (n : Nat) (lo : Int) (st1 st2 : Store),
      (∀ (c : BufId) (i : Int), c ≠ w → st1 c i = st2 c i) →
      ∀ store1 store2, execLoopIters f env v lo n st1 = some store1 →
        execLoopIters f env v lo n st2 = some store2 →
        ∀ (c : BufId) (i : Int), c ≠ w → store1 c i = store2 c i := by
  intro n lo
  induction n generalizing lo with
  | zero =>
      intro st1 st2 h store1 store2 h1 h2
      simp only [execLoopIters, Option.some.injEq] at h1 h2
      subst h1; subst h2
      exact h
  | succ n ih =>
      intro st1 st2 h store1 store2 h1 h2
      obtain ⟨r1, hr1, hrest1⟩ : ∃ r1, f (Env.set env v lo) st1 = some r1 ∧
          execLoopIters f env v (lo + 1) n r1 = some store1 := by
        simpa [execLoopIters, Option.bind_eq_some_iff] using h1
      obtain ⟨r2, hr2, hrest2⟩ : ∃ r2, f (Env.set env v lo) st2 = some r2 ∧
          execLoopIters f env v (lo + 1) n r2 = some store2 := by
        simpa [execLoopIters, Option.bind_eq_some_iff] using h2
      intro c hci i
      exact ih (lo + 1) r1 r2 (fun c' hci' i' => hf _ _ _ h r1 r2 hr1 hr2 c' hci' i')
        store1 store2 hrest1 hrest2 c hci i

/-- Executions of a statement that never reads or writes `w` produce results
    that agree away from `w`, provided the starting stores do. -/
theorem execStmt_blind (fuel : Nat) (w : BufId) :
    ∀ (env : Env) (st1 st2 : Store) (s : Stmt),
      stmtReadsBuf w s = false → isReadOnly w s = true →
      (∀ (c : BufId) (i : Int), c ≠ w → st1 c i = st2 c i) →
      ∀ r1 r2, execStmt fuel env st1 s = some r1 → execStmt fuel env st2 s = some r2 →
        ∀ (c : BufId) (i : Int), c ≠ w → r1 c i = r2 c i := by
  intro env st1 st2 s
  induction s generalizing env st1 st2 with
  | skip =>
      intro _ _ h r1 r2 h1 h2
      simp only [execStmt, Option.some.injEq] at h1 h2
      subst h1; subst h2
      exact h
  | bufWrite buf indices val =>
      intro hr hw hst r1 r2 h1 h2
      simp only [stmtReadsBuf, Bool.or_eq_false_iff] at hr
      simp only [isReadOnly, bne_iff_ne] at hw
      simp only [execStmt, Option.some.injEq] at h1 h2
      subst h1; subst h2
      have hbuf : buf ≠ w := hw
      have hval : evalExpr env st1 val = evalExpr env st2 val :=
        evalExpr_blind w env st1 st2 hst val hr.2
      have hflat : flatIndex env st1 indices = flatIndex env st2 indices :=
        foldl_eval_congr_store env st1 st2 indices 0
          (fun e he => evalExpr_blind w env st1 st2 hst e (readsBufs_false w indices hr.1 e he))
      intro c i hc
      by_cases hcb : c = buf
      · subst c
        by_cases hci : i = flatIndex env st2 indices
        · simp [hci, hflat, hval]
        · simp [hci, hflat, hst buf i hbuf,
            show i ≠ flatIndex env st1 indices from fun h => hci (h.trans hflat)]
      · have hcb : (c == buf) = false := beq_eq_false_iff_ne.mpr hcb
        simp [hcb, hst c i hc]
  | loop v lo hi _ body ih =>
      intro hr hw hst r1 r2 h1 h2
      obtain ⟨⟨hlo', hhi'⟩, hbody⟩ :
          (readsBuf w lo = false ∧ readsBuf w hi = false) ∧ stmtReadsBuf w body = false := by
        simpa [stmtReadsBuf] using hr
      have hlo : evalExpr env st1 lo = evalExpr env st2 lo :=
        evalExpr_blind w env st1 st2 hst lo hlo'
      have hhi : evalExpr env st1 hi = evalExpr env st2 hi :=
        evalExpr_blind w env st1 st2 hst hi hhi'
      have e1 : execLoopIters (fun e st => execStmt fuel e st body) env v
          (evalExpr env st1 lo) ((evalExpr env st1 hi - evalExpr env st1 lo).toNat) st1 = some r1 :=
        by simpa [execStmt] using h1
      have e2 : execLoopIters (fun e st => execStmt fuel e st body) env v
          (evalExpr env st2 lo) ((evalExpr env st2 hi - evalExpr env st2 lo).toNat) st2 = some r2 :=
        by simpa [execStmt] using h2
      rw [hlo, hhi] at e1
      exact execLoopIters_blind _ w
        (fun e a b hag r1' r2' hr1' hr2' => ih e a b hbody hw hag r1' r2' hr1' hr2')
        env v _ _ _ _ hst r1 r2 e1 e2
  | seq s1 s2 ih1 ih2 =>
      intro hr hw hst r1 r2 h1 h2
      simp only [stmtReadsBuf, Bool.or_eq_false_iff] at hr
      simp only [isReadOnly, Bool.and_eq_true] at hw
      obtain ⟨mid1, hm1, hrest1⟩ : ∃ mid1, execStmt fuel env st1 s1 = some mid1 ∧
          execStmt fuel env mid1 s2 = some r1 := by
        simpa [execStmt, Option.bind_eq_some_iff] using h1
      obtain ⟨mid2, hm2, hrest2⟩ : ∃ mid2, execStmt fuel env st2 s1 = some mid2 ∧
          execStmt fuel env mid2 s2 = some r2 := by
        simpa [execStmt, Option.bind_eq_some_iff] using h2
      exact ih2 env mid1 mid2 hr.2 hw.2
        (ih1 env st1 st2 hr.1 hw.1 hst mid1 mid2 hm1 hm2) r1 r2 hrest1 hrest2
  | alloc _ _ body ih =>
      intro hr hw hst r1 r2 h1 h2
      simp only [stmtReadsBuf, Bool.or_eq_false_iff] at hr
      exact ih env st1 st2 hr.2 (by simpa [isReadOnly] using hw) hst r1 r2 h1 h2

/-! ## Agreement-based congruence -/

/-- If the two fold bodies agree whenever a predicate `P` holds of the store,
    and `P` is preserved by the right-hand body, the folds agree. -/
theorem execLoopIters_congr_under (fL fR : Env → Store → Option Store) (env : Env) (v : VarId)
    (P : Store → Prop)
    (htot : ∀ (e : Env) (st : Store), ∃ st', fR e st = some st')
    (hpt : ∀ (e : Env) (st : Store), P st → fL e st = fR e st)
    (hpres : ∀ (e : Env) (st st' : Store), fR e st = some st' → P st → P st') :
    ∀ (n : Nat) (lo : Int) (st : Store), P st →
      execLoopIters fL env v lo n st = execLoopIters fR env v lo n st := by
  intro n lo st
  induction n generalizing lo st with
  | zero => intro _; rfl
  | succ n ih =>
      intro hP
      simp only [execLoopIters]
      obtain ⟨st1, h1⟩ := htot _ _
      rw [hpt _ _ hP, h1]
      exact ih (lo + 1) st1 (hpres _ _ _ h1 hP)

end VeriTac
