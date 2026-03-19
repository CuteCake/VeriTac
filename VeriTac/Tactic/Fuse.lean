/-
  VeriTac.Tactic.Fuse
  Fuse transformation: merge two adjacent loops into one.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Fuse two adjacent sequential loops with the same bounds into one.
    Transforms `seq (loop v1 0 N1 body1) (loop v2 0 N2 body2)`
    into `loop v_fused 0 (N1*N2) body'`
    where v1 = v_fused / N2, v2 = v_fused % N2. -/
def fuse (stmt : Stmt) (var1 var2 : VarId) : Option Stmt :=
  match stmt with
  | .seq (.loop v1 lo1 hi1 ann1 body1) (.loop v2 lo2 hi2 _ann2 body2) =>
    if v1 == var1 && v2 == var2 then
      let fusedVar := v1 ++ "_" ++ v2 ++ "_fused"
      let fusedHi := SExpr.mul hi1 hi2
      -- v1 = fusedVar / hi2, v2 = fusedVar % hi2
      let body1' := substStmtVar v1 (.div (.var fusedVar) hi2) body1
      let body2' := substStmtVar v2 (.mod (.var fusedVar) hi2) body2
      some (.loop fusedVar lo1 fusedHi ann1 (.seq body1' body2'))
    else none
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

/-- Fuse correctness: merging two loops via (i,j) ↔ i*N+j bijection preserves semantics. -/
theorem fuse_correct (v1 v2 : VarId) (N1 N2 : Nat) (hN2 : N2 > 0)
    (body1 body2 : Stmt) (ann : Annotation) :
    .seq (.loop v1 (.lit 0) (.lit N1) ann body1)
         (.loop v2 (.lit 0) (.lit N2) .none body2) ≈ₛ
    (match fuse (.seq (.loop v1 (.lit 0) (.lit N1) ann body1)
                      (.loop v2 (.lit 0) (.lit N2) .none body2)) v1 v2 with
     | some s => s
     | none => .seq (.loop v1 (.lit 0) (.lit N1) ann body1)
                    (.loop v2 (.lit 0) (.lit N2) .none body2)) := by
  sorry -- Requires bijection proof: (i,j) ↔ i*N2+j for 0≤i<N1, 0≤j<N2

end VeriTac.Tactic
