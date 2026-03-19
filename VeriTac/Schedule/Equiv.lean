/-
  VeriTac.Schedule.Equiv
  Schedule equivalence relation.
-/
import VeriTac.Schedule.LoopNest

namespace VeriTac

/-- Two statements are schedule-equivalent if they produce the same store
    for all environments, stores, and sufficient fuel. -/
def ScheduleEquiv (s1 s2 : Stmt) : Prop :=
  ∀ (fuel : Nat) (env : Env) (store : Store),
    execStmt fuel env store s1 = execStmt fuel env store s2

notation:50 s1 " ≈ₛ " s2 => ScheduleEquiv s1 s2

namespace ScheduleEquiv

theorem refl (s : Stmt) : s ≈ₛ s :=
  fun _ _ _ => rfl

theorem symm {s1 s2 : Stmt} (h : s1 ≈ₛ s2) : s2 ≈ₛ s1 :=
  fun fuel env store => (h fuel env store).symm

theorem trans {s1 s2 s3 : Stmt} (h1 : s1 ≈ₛ s2) (h2 : s2 ≈ₛ s3) : s1 ≈ₛ s3 :=
  fun fuel env store => (h1 fuel env store).trans (h2 fuel env store)

end ScheduleEquiv

end VeriTac
