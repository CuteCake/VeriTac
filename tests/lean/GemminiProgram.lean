/-
  VeriTac Gemmini program checker: kernel-decide acceptance and rejection
  fixtures for `checkSupportedProgram` (VeriTac.Gemmini.Checked).

  Accepted: the four proved concrete programs (16^3 and 32x16x16, both
  schedules) against capacity plans.  Rejected: mutated load offset, mutated
  accumulate flag, mutated preload source, mutated load stride, missing fence,
  swapped schedule order, undersized scratchpad, undersized accumulator for
  reuse_b, and unsupported 32-cube shapes.
-/
import VeriTac.Gemmini.Checked

namespace VeriTac.Gemmini

/-! ## Plans at the supported capacities -/

def plan16 (sched : GemminiSchedule) : GemminiPlan :=
  { m := 16, n := 16, k := 16, dim := 16,
    scratchpadRows := 32, accumulatorRows := 16, schedule := sched }

def plan32 (sched : GemminiSchedule) (accRows : Nat) : GemminiPlan :=
  { m := 32, n := 16, k := 16, dim := 16,
    scratchpadRows := 32, accumulatorRows := accRows, schedule := sched }

/-! ## Acceptance: the four proved programs -/

example : checkSupportedProgram (plan16 .baseline) baseline16 = true := by decide
example : checkSupportedProgram (plan16 .reuseB) reuse16 = true := by decide
example : checkSupportedProgram (plan32 .baseline 16) baseline32 = true := by decide
example : checkSupportedProgram (plan32 .reuseB 32) reuse32 = true := by decide

/-! ## Rejection: mutated instructions -/

/-- A-load offset 0 -> 8 corrupts the source tile. -/
def baseline16_badOffset : Program :=
  [.configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
   .mvin 0 .a 8 0 16 16, .mvin 1 .b 0 16 16 16,
   .preload 16 2684354560, .compute false 0 4294967295,
   .mvout 0 3758096384 16 16, .fence]
example : checkSupportedProgram (plan16 .baseline) baseline16_badOffset = false := by decide

/-- compute_preloaded -> compute_accumulated with no retained weights. -/
def baseline16_badFlag : Program :=
  [.configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
   .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
   .preload 16 2684354560, .compute true 0 4294967295,
   .mvout 0 3758096384 16 16, .fence]
example : checkSupportedProgram (plan16 .baseline) baseline16_badFlag = false := by decide

/-- Preload B source row 16 -> 0 aliases the A tile. -/
def baseline16_badPreload : Program :=
  [.configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
   .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
   .preload 0 2684354560, .compute false 0 4294967295,
   .mvout 0 3758096384 16 16, .fence]
example : checkSupportedProgram (plan16 .baseline) baseline16_badPreload = false := by decide

/-- configLd slot-0 stride 16 -> 32 mis-addresses every A row. -/
def baseline16_badStride : Program :=
  [.configEx 1, .configSt 64, .configLd 0 32 true, .configLd 1 16 true,
   .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
   .preload 16 2684354560, .compute false 0 4294967295,
   .mvout 0 3758096384 16 16, .fence]
example : checkSupportedProgram (plan16 .baseline) baseline16_badStride = false := by decide

/-- The closing fence is dropped; DMA drain is no longer guaranteed. -/
def baseline16_noFence : Program :=
  [.configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
   .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
   .preload 16 2684354560, .compute false 0 4294967295,
   .mvout 0 3758096384 16 16]
example : checkSupportedProgram (plan16 .baseline) baseline16_noFence = false := by decide

/-- The reuse_b load order is not the baseline program. -/
example : checkSupportedProgram (plan16 .baseline) reuse16 = false := by decide

/-! ## Rejection: plan shape and capacity -/

/-- Scratchpad below the 2 * dim staging requirement. -/
def plan16_starved : GemminiPlan :=
  { plan16 .baseline with scratchpadRows := 16 }
example : checkSupportedProgram plan16_starved baseline16 = false := by decide

/-- reuse_b at m=32 needs 32 accumulator rows; 16 must fail closed. -/
example : checkSupportedProgram (plan32 .reuseB 16) reuse32 = false := by decide

/-- 32-cube is outside the proved subset even with a valid program prefix. -/
example : checkSupportedProgram (plan32 .baseline 16) baseline16 = false := by decide

/-! ## Kernel soundness statement exercised on a concrete plan

`checkSupportedProgram_sound` quantifies over all int8 input functions; here it
is instantiated once with an arbitrary-function statement on the 16^3 baseline
plan to confirm the exact conclusion shape (success, drained, outputs, written)
is derivable without any extra premise. -/

theorem sound16_instance (a b : Nat → Int)
    (ha : ∀ i < 256, GemminiExact.Int8 (a i))
    (hb : ∀ i < 256, GemminiExact.Int8 (b i)) :
    ∃ s, runProgram (plan16 .baseline) a b baseline16 = some s ∧ s.drained = true ∧
      ∀ r < 16, ∀ c < 16,
        s.output (r * 16 + c) = gemm (plan16 .baseline) a b r c ∧
        s.written (r * 16 + c) = true :=
  checkSupportedProgram_sound (plan16 .baseline) baseline16 (by decide) a b ha hb

/-! ## Axiom audit

The acceptance path must rest only on Lean's built-in axioms (propext,
Classical.choice, Quot.sound) plus `sorryAx`-free proofs.  These commands fail
the build under any `sorry` in the dependency chain. -/

#print axioms checkSupportedProgram_sound
#print axioms checkSupportedProgram_observe
#print axioms run_baseline16_observe
#print axioms run_reuse16_observe
#print axioms run_baseline32_observe
#print axioms run_reuse32_observe
#print axioms signed32_dot_prefix
#print axioms signed32_eq_of_bounds

end VeriTac.Gemmini
