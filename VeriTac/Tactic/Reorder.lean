/-
  VeriTac.Tactic.Reorder
  Reorder transformation: swap two adjacent independent loops.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Schedule.FreeVars
import VeriTac.Schedule.ExecLemmas
import VeriTac.Schedule.LoopComposition

namespace VeriTac.Tactic

/-- Check if the bounds of loop2 depend on loop1's variable.

    ⚠️ This is a *heuristic*, not a sound dependence analysis. Swapping two nested
    loops is only semantics-preserving when the body has no loop-carried dependence
    between the two axes (no read-after-write / write-after-write / write-after-read
    conflicts across iterations). This check only verifies that the inner loop's
    *bounds* do not use the outer variable; it does not analyze the body. The
    machine-checked `reorder_correct` theorem therefore additionally requires
    `swapSafe` (bounds independent of the other axis and of the store) and a
    semantic pairwise-commutation hypothesis on the body. A full affine dependence
    check is a known future item (see `veritac_design.md` §7.4). -/
def loopsIndependent (var1 : VarId) (_var2 : VarId) (lo2 hi2 : SExpr) (_body : Stmt) : Bool :=
  !varInExpr var1 lo2 && !varInExpr var1 hi2

/-- Sound dependence criterion for reordering: every loop bound is independent of
    the other axis' variable and does not read the store, so that both loop nests
    traverse exactly the same iterate rectangle. The *body* relation is captured
    separately by the commutation hypothesis of `reorder_correct`. -/
def swapSafe (v1 v2 : VarId) (lo1 hi1 lo2 hi2 : SExpr) : Bool :=
  !varInExpr v2 lo1 && !varInExpr v2 hi1 && exprStatic lo1 && exprStatic hi1 &&
    !varInExpr v1 lo2 && !varInExpr v1 hi2 && exprStatic lo2 && exprStatic hi2

/-- Swap two adjacent nested loops if they are (heuristically) independent. -/
def reorder (stmt : Stmt) (var1 var2 : VarId) : Option Stmt :=
  match stmt with
  | .loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body) =>
    if v1 == var1 && v2 == var2 then
      if swapSafe v1 v2 lo1 hi1 lo2 hi2 then
        some (.loop v2 lo2 hi2 ann2 (.loop v1 lo1 hi1 ann1 body))
      else none
    else
      -- recurse on the *whole* inner loop (v2 ... body), not just `body`:
      -- the target pair may be adjacent deeper down (e.g. loop a { loop b { loop c } }
      -- reordering b,c requires recursing into `loop b { loop c }`).
      match reorder (.loop v2 lo2 hi2 ann2 body) var1 var2 with
      | some body' => some (.loop v1 lo1 hi1 ann1 body')
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

/-- Reorder correctness. Swapping two nested loops preserves semantics provided
    the bounds are independent (no cross-axis or store dependence) and the body
    executes commute pairwise across the two axes — the genuine dependence
    obligation that `loopsIndependent` can only approximate syntactically. -/
theorem reorder_correct (v1 v2 : VarId) (lo1 hi1 lo2 hi2 : SExpr) (ann1 ann2 : Annotation)
    (body : Stmt) (h12 : v1 ≠ v2) (hsafe : swapSafe v1 v2 lo1 hi1 lo2 hi2 = true) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      (∀ (i j i' j' : Int) (st : Store),
        (kcomp (fun st' => execStmt fuel (Env.set (Env.set env v1 i) v2 j) st' body)
               (fun st' => execStmt fuel (Env.set (Env.set env v1 i') v2 j') st' body)) st =
        (kcomp (fun st' => execStmt fuel (Env.set (Env.set env v1 i') v2 j') st' body)
               (fun st' => execStmt fuel (Env.set (Env.set env v1 i) v2 j) st' body)) st) →
      execStmt fuel env store (.loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body)) =
      execStmt fuel env store (.loop v2 lo2 hi2 ann2 (.loop v1 lo1 hi1 ann1 body)) := by
  intro fuel env store hcomm
  -- extract the bound-independence facts (left-nested conjunction, peel from right)
  simp only [swapSafe, Bool.and_eq_true] at hsafe
  have hb1l : varInExpr v2 lo1 = false := by simpa using hsafe.1.1.1.1.1.1.1
  have hb1h : varInExpr v2 hi1 = false := by simpa using hsafe.1.1.1.1.1.1.2
  have hs1l : exprStatic lo1 = true := hsafe.1.1.1.1.1.2
  have hs1h : exprStatic hi1 = true := hsafe.1.1.1.1.2
  have hb2l : varInExpr v1 lo2 = false := by simpa using hsafe.1.1.1.2
  have hb2h : varInExpr v1 hi2 = false := by simpa using hsafe.1.1.2
  have hs2l : exprStatic lo2 = true := hsafe.1.2
  have hs2h : exprStatic hi2 = true := hsafe.2
  -- constant bounds (independent of the other axis' variable and of the store)
  have hlo1c : ∀ (c : Int) (st : Store),
      evalExpr (Env.set env v2 c) st lo1 = evalExpr env store lo1 := by
    intro c st
    calc evalExpr (Env.set env v2 c) st lo1 = evalExpr env st lo1 :=
        evalExpr_set_unused v2 c env st lo1 hb1l
      _ = evalExpr env store lo1 := exprStatic_storeIrrel env st store lo1 hs1l
  have hhi1c : ∀ (c : Int) (st : Store),
      evalExpr (Env.set env v2 c) st hi1 = evalExpr env store hi1 := by
    intro c st
    calc evalExpr (Env.set env v2 c) st hi1 = evalExpr env st hi1 :=
        evalExpr_set_unused v2 c env st hi1 hb1h
      _ = evalExpr env store hi1 := exprStatic_storeIrrel env st store hi1 hs1h
  have hlo2c : ∀ (c : Int) (st : Store),
      evalExpr (Env.set env v1 c) st lo2 = evalExpr env store lo2 := by
    intro c st
    calc evalExpr (Env.set env v1 c) st lo2 = evalExpr env st lo2 :=
        evalExpr_set_unused v1 c env st lo2 hb2l
      _ = evalExpr env store lo2 := exprStatic_storeIrrel env st store lo2 hs2l
  have hhi2c : ∀ (c : Int) (st : Store),
      evalExpr (Env.set env v1 c) st hi2 = evalExpr env store hi2 := by
    intro c st
    calc evalExpr (Env.set env v1 c) st hi2 = evalExpr env st hi2 :=
        evalExpr_set_unused v1 c env st hi2 hb2h
      _ = evalExpr env store hi2 := exprStatic_storeIrrel env st store hi2 hs2h
  -- the four constant iteration counts / starts
  set loV1 : Int := evalExpr env store lo1 with hloV1
  set loV2 : Int := evalExpr env store lo2 with hloV2
  set it1 : Nat := (evalExpr env store hi1 - loV1).toNat with hit1
  set it2 : Nat := (evalExpr env store hi2 - loV2).toNat with hit2
  let fbody : Env → Store → Option Store := fun e st => execStmt fuel e st body
  -- inner folds use the constant bounds
  have hinner1 : ∀ (c : Int) (st : Store),
      execStmt fuel (Env.set env v1 c) st (.loop v2 lo2 hi2 ann2 body)
      = execLoopIters fbody (Env.set env v1 c) v2 loV2 it2 st := by
    intro c st
    dsimp [fbody]
    simp only [execStmt]
    rw [hlo2c c st, hhi2c c st]
  have hinner2 : ∀ (c : Int) (st : Store),
      execStmt fuel (Env.set env v2 c) st (.loop v1 lo1 hi1 ann1 body)
      = execLoopIters fbody (Env.set env v2 c) v1 loV1 it1 st := by
    intro c st
    dsimp [fbody]
    simp only [execStmt]
    rw [hlo1c c st, hhi1c c st]
  -- unfold LHS into the canonical fold shape
  have hL : execStmt fuel env store (.loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 ann2 body))
      = execLoopIters (fun e st => execLoopIters fbody e v2 loV2 it2 st) env v1 loV1 it1 store := by
    simp only [execStmt]
    rw [show evalExpr env store lo1 = loV1 from rfl,
      show ((evalExpr env store hi1 - evalExpr env store lo1).toNat) = it1 from by
        rw [show evalExpr env store lo1 = loV1 from rfl, hit1]]
    exact execLoopIters_congr
      (fun e st => execStmt fuel e st (.loop v2 lo2 hi2 ann2 body))
      (fun e st => execLoopIters fbody e v2 loV2 it2 st)
      env env v1 (fun c st => hinner1 c st) it1 loV1 store
  -- unfold RHS into the canonical fold shape
  have hR : execStmt fuel env store (.loop v2 lo2 hi2 ann2 (.loop v1 lo1 hi1 ann1 body))
      = execLoopIters (fun e st => execLoopIters fbody e v1 loV1 it1 st) env v2 loV2 it2 store := by
    simp only [execStmt]
    rw [show evalExpr env store lo2 = loV2 from rfl,
      show ((evalExpr env store hi2 - evalExpr env store lo2).toNat) = it2 from by
        rw [show evalExpr env store lo2 = loV2 from rfl, hit2]]
    exact execLoopIters_congr
      (fun e st => execStmt fuel e st (.loop v1 lo1 hi1 ann1 body))
      (fun e st => execLoopIters fbody e v1 loV1 it1 st)
      env env v2 (fun c st => hinner2 c st) it2 loV2 store
  -- the two folds are equal by the swap (commutation) lemma
  rw [hL, hR]
  exact congrFun (execLoopIters_swap fbody env v1 v2 h12 it2 hcomm it1 loV1 loV2) store

end VeriTac.Tactic
