/-
  VeriTac.Tactic.Fuse
  Fuse transformation: merge two adjacent *nested* loops into one.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Schedule.LoopComposition
import VeriTac.Schedule.ExecLemmas
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Fuse two adjacent *nested* loops into a single loop over the merged index.
    `loop i 0 N { loop j 0 M { body } }` becomes
    `loop ij 0 (N*M) { body[i := ij / M, j := ij % M] }`.

    This is the standard, semantics-preserving fusion (each `(i, j)` pair is visited
    exactly once, in the same order), as opposed to fusing two *sequential* loops,
    which would change how many times each body runs. The proof obligation is the
    `(i, j) ↔ i*M + j` bijection for `0 ≤ i < N`, `0 ≤ j < M`. -/
def fuse (stmt : Stmt) (var1 var2 : VarId) : Option Stmt :=
  match stmt with
  | .loop v1 lo1 hi1 ann1 (.loop v2 lo2 hi2 _ann2 body) =>
    if v1 = var1 && v2 = var2 then
      let fusedVar := v1 ++ "_" ++ v2 ++ "_fused"
      -- v1 = fusedVar / hi2, v2 = fusedVar % hi2
      let body' := substStmtVar v1 (.div (.var fusedVar) hi2)
        (substStmtVar v2 (.mod (.var fusedVar) hi2) body)
      some (.loop fusedVar (.mul lo1 hi2) (.mul hi1 hi2) ann1 body')
    else
      match fuse body var1 var2 with
      | some body' => some (.loop v1 lo1 hi1 ann1 body')
      | none => none
  | .seq s1 s2 =>
    match fuse s1 var1 var2 with
    | some s1' => some (.seq s1' s2)
    | none => match fuse s2 var1 var2 with
      | some s2' => some (.seq s1 s2')
      | none => none
  | .loop v lo hi ann body =>
    match fuse body var1 var2 with
    | some body' => some (.loop v lo hi ann body')
    | none => none
  | _ => none

/-- Fuse correctness: merging two nested loops via the `(i, j) ↔ i*M+j` bijection
    preserves semantics. Requires `M > 0` (so that `i = ij / M`, `j = ij % M` is a
    well-defined split of `ij ∈ [0, N*M)`), the generated fused-name is fresh for the
    body, and the body re-binds neither `v1` nor `v2`. -/
theorem fuse_correct (v1 v2 : VarId) (N M : Nat) (hM : M > 0)
    (body : Stmt) (ann1 : Annotation)
    (hfv1 : loopBinds v1 body = false)
    (hfv2 : loopBinds v2 body = false)
    (hlw : loopBinds (v1 ++ "_" ++ v2 ++ "_fused") body = false)
    (hfw : varFreeStmt (v1 ++ "_" ++ v2 ++ "_fused") body = false) :
    .loop v1 (.lit 0) (.lit N) ann1 (.loop v2 (.lit 0) (.lit M) .none body) ≈ₛ
    (match fuse (.loop v1 (.lit 0) (.lit N) ann1
                  (.loop v2 (.lit 0) (.lit M) .none body)) v1 v2 with
     | some s => s
     | none => .loop v1 (.lit 0) (.lit N) ann1
                   (.loop v2 (.lit 0) (.lit M) .none body)) := by
  set w : VarId := v1 ++ "_" ++ v2 ++ "_fused" with hw
  -- distinctness of the generated fused-name from the loop variables
  have hlenu : String.length ("_" : String) = 1 := by decide
  have hlenf : String.length ("_fused" : String) = 6 := by decide
  have hwv1 : v1 ++ "_" ++ v2 ++ "_fused" ≠ v1 := by
    intro h
    have hl := congrArg String.length h
    simp [String.length_append, hlenu, hlenf] at hl
    omega
  have hwv2 : v1 ++ "_" ++ v2 ++ "_fused" ≠ v2 := by
    intro h
    have hl := congrArg String.length h
    simp [String.length_append, hlenu, hlenf] at hl
    omega
  have hwv1' : w ≠ v1 := hwv1
  have hwv2' : w ≠ v2 := hwv2
  -- static / freshness facts for the two substitutions
  have hstatic1 : exprStatic (.div (.var w) (.lit (M : Int))) = true := by
    simp [exprStatic]
  have hstatic2 : exprStatic (.mod (.var w) (.lit (M : Int))) = true := by
    simp [exprStatic]
  have hb2' : loopBinds v2 body = false := hfv2
  have hrep2 : ∀ x, varInExpr x (.mod (.var w) (.lit (M : Int))) = true
        → loopBinds x body = false := by
    intro x hx
    have hxw : w = x := by
      simpa [varInExpr, Bool.or_false, decide_eq_true_iff] using hx
    rw [← hxw]
    exact hlw
  have hrep1 : ∀ x, varInExpr x (.div (.var w) (.lit (M : Int))) = true
        → loopBinds x (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body) = false := by
    intro x hx
    have hxw : w = x := by
      simpa [varInExpr, Bool.or_false, decide_eq_true_iff] using hx
    rw [loopBinds_subst, ← hxw]
    exact hlw
  have hb1' : loopBinds v1 (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body) = false := by
    rw [loopBinds_subst]
    exact hfv1
  -- the fused outcome
  have hfuse : fuse (.loop v1 (.lit 0) (.lit N) ann1
        (.loop v2 (.lit 0) (.lit M) .none body)) v1 v2
      = some (.loop (v1 ++ "_" ++ v2 ++ "_fused") (.mul (.lit 0) (.lit M))
            (.mul (.lit N) (.lit M)) ann1
            (substStmtVar v1 (.div (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int)))
              (substStmtVar v2 (.mod (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int))) body))) := by
    rw [fuse, if_pos (by simp : (v1 = v1 && v2 = v2) = true)]
  rw [hfuse]
  intro fuel env store
  -- arithmetic
  have hMne : (M : Int) ≠ 0 := by omega
  have hMpos : (0 : Int) < (M : Int) := by omega
  have hnm : ((N * M : Nat) : Int) - 0 = (N * M : Nat) := by omega
  -- count normalization for the fused loop
  have hIter : (((N : Int) * (M : Int)) - 0).toNat = N * M := by
    rw [show (((N : Int) * (M : Int)) - (0 : Int)) = (((N : Int) * (M : Int))) from by ring,
      ← Int.natCast_mul, Int.toNat_natCast]
  -- per (q,j)-step: fused iteration k = q*M + j corresponds to (v1,v2) = (q,j)
  have hstep : ∀ (q j : Int) (st : Store), 0 ≤ j → j < (M : Int) →
      execStmt fuel (Env.set env w (q * (M : Int) + j)) st
          (substStmtVar v1 (.div (.var w) (.lit (M : Int)))
            (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body))
      = execStmt fuel (Env.set (Env.set env v1 q) v2 j) st body := by
    intro q j st hj0 hjlt
    have hd := int_mul_add_div_mod q M j hj0 hjlt hMpos
    -- inner substitution first (v2 := w mod M), then outer (v1 := w div M)
    have hsub2 : execStmt fuel (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) st
          (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body)
        = execStmt fuel (Env.set (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) v2 j) st body := by
      have hE : (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) w = q * (M : Int) + j := by
        simp [Env.set, hwv1']
      have hev : evalExpr (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) st
            (.mod (.var w) (.lit (M : Int)))
          = j := by
        simp only [evalExpr]
        rw [if_neg hMne, hE, hd.2]
      rw [← substStmtVar_correct v2 (.mod (.var w) (.lit (M : Int))) fuel
          (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) st body hfv2 hrep2 hstatic2, hev]
    have hsub1 : execStmt fuel (Env.set env w (q * (M : Int) + j)) st
          (substStmtVar v1 (.div (.var w) (.lit (M : Int)))
            (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body))
        = execStmt fuel (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) st
            (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body) := by
      have hE1 : (Env.set env w (q * (M : Int) + j)) w = q * (M : Int) + j := by
        simp [Env.set]
      have hev1 : evalExpr (Env.set env w (q * (M : Int) + j)) st
            (.div (.var w) (.lit (M : Int)))
          = q := by
        simp only [evalExpr]
        rw [if_neg hMne, hE1, hd.1]
      rw [← substStmtVar_correct v1 (.div (.var w) (.lit (M : Int))) fuel
          (Env.set env w (q * (M : Int) + j)) st
          (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body) hb1' hrep1 hstatic1, hev1]
    -- glue and peel the generated name from the environment
    rw [hsub1, hsub2]
    refine execStmt_agree_except fuel w body hfw
      (Env.set (Env.set (Env.set env w (q * (M : Int) + j)) v1 q) v2 j)
      (Env.set (Env.set env v1 q) v2 j) st ?_
    intro z hz
    by_cases hzv1 : z = v1
    · subst z; simp [Env.set, hwv1']
    · by_cases hzv2 : z = v2
      · subst z; simp [Env.set, hwv2']
      · simp [Env.set, hz, hzv1, hzv2]
  -- unfold both sides
  have hL : execStmt fuel env store (.loop v1 (.lit 0) (.lit N) ann1
        (.loop v2 (.lit 0) (.lit M) .none body))
      = execLoopIters (fun e st => execLoopIters
          (fun e' st' => execStmt fuel e' st' body) e v2 0 M st) env v1 0 N store := by
    simp only [execStmt, evalExpr, Int.toNat_natCast, Int.natCast_zero, sub_zero]
  have hR : execStmt fuel env store (.loop (v1 ++ "_" ++ v2 ++ "_fused")
        (.mul (.lit 0) (.lit M)) (.mul (.lit N) (.lit M)) ann1
          (substStmtVar v1 (.div (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int)))
            (substStmtVar v2 (.mod (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int))) body)))
      = execLoopIters (fun e' st' => execStmt fuel e' st'
          (substStmtVar v1 (.div (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int)))
            (substStmtVar v2 (.mod (.var (v1 ++ "_" ++ v2 ++ "_fused")) (.lit (M : Int))) body)))
          env (v1 ++ "_" ++ v2 ++ "_fused") 0 (N * M) store := by
    simp only [execStmt, evalExpr, Int.natCast_zero, Int.zero_mul]
    rw [hIter]
  rw [hL, hR]
  rw [execLoopIters_fuse
    (fun e' st' => execStmt fuel e' st'
      (substStmtVar v1 (.div (.var w) (.lit (M : Int)))
        (substStmtVar v2 (.mod (.var w) (.lit (M : Int))) body)))
    (fun e' st' => execStmt fuel e' st' body)
    env w v1 v2 M hstep N 0 store]
  rw [show ((0 : Int) * (M : Int)) = 0 from by ring, Nat.mul_comm M N]

end VeriTac.Tactic
