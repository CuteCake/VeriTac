import VeriTac.Gemmini.LoweringSound
open VeriTac.Gemmini
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedRewrite
def plan : GemminiPlan := { m := 48, n := 16, k := 32, dim := 16, scratchpadRows := 32, accumulatorRows := 32, schedule := .baseline }
def source : Program := [
  .configEx 1,
  .configSt 64,
  .configLd 0 32 true,
  .configLd 1 16 true,
  .mvin 0 .a 0 0 16 16,
  .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560,
  .compute false 0 4294967295,
  .mvin 0 .a 16 0 16 16,
  .mvin 1 .b 256 16 16 16,
  .preload 16 3758096384,
  .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 512 0 16 16,
  .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560,
  .compute false 0 4294967295,
  .mvin 0 .a 528 0 16 16,
  .mvin 1 .b 256 16 16 16,
  .preload 16 3758096384,
  .compute false 0 4294967295,
  .mvout 256 3758096384 16 16,
  .mvin 0 .a 1024 0 16 16,
  .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560,
  .compute false 0 4294967295,
  .mvin 0 .a 1040 0 16 16,
  .mvin 1 .b 256 16 16 16,
  .preload 16 3758096384,
  .compute false 0 4294967295,
  .mvout 512 3758096384 16 16,
  .fence
]
def destination : Program := [
  .configEx 1,
  .configSt 64,
  .configLd 0 32 true,
  .configLd 1 16 true,
  .mvin 1 .b 0 16 16 16,
  .mvin 0 .a 0 0 16 16,
  .preload 16 2684354560,
  .compute false 0 4294967295,
  .mvin 0 .a 512 0 16 16,
  .preload 16 2684354576,
  .compute false 0 4294967295,
  .mvin 1 .b 256 16 16 16,
  .mvin 0 .a 16 0 16 16,
  .preload 16 3758096384,
  .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 528 0 16 16,
  .preload 16 3758096400,
  .compute false 0 4294967295,
  .mvout 256 3758096400 16 16,
  .mvin 1 .b 0 16 16 16,
  .mvin 0 .a 1024 0 16 16,
  .preload 16 2684354560,
  .compute false 0 4294967295,
  .mvin 1 .b 256 16 16 16,
  .mvin 0 .a 1040 0 16 16,
  .preload 16 3758096384,
  .compute false 0 4294967295,
  .mvout 512 3758096384 16 16,
  .fence
]

theorem sourceAccepted : Symbolic.check plan source = true := by decide +kernel
theorem destinationAccepted : Symbolic.check plan destination = true := by decide +kernel
def edge : CheckedRewrite plan source := ⟨destination, sourceAccepted, destinationAccepted⟩

theorem equivalent (a b : Nat → Int)
    (ha : ∀ i < plan.m * plan.k, VeriTac.GemminiExact.Int8 (a i))
    (hb : ∀ i < plan.k * plan.n, VeriTac.GemminiExact.Int8 (b i)) :
    ∃ before after, runProgram plan a b source = some before ∧
      runProgram plan a b destination = some after ∧
      before.drained = true ∧ after.drained = true ∧
      ∀ r < plan.m, ∀ c < plan.n,
        before.output (r * plan.n + c) = after.output (r * plan.n + c) :=
  checkedRewrite_sound plan source edge a b ha hb
#print axioms equivalent
end SubmittedRewrite
