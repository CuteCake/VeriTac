import VeriTac.Gemmini.LoweringSound
import VeriTac.Gemmini.Executable
import VeriTac.Gemmini.K32Baseline
import VeriTac.Gemmini.K32Reuse

open VeriTac.Gemmini
set_option maxRecDepth 100000
set_option maxHeartbeats 0

def p32 : GemminiPlan := {
  m := 32, n := 16, k := 32, dim := 16
  scratchpadRows := 32, accumulatorRows := 32, schedule := .baseline
}
def p48 : GemminiPlan := { p32 with m := 48 }
def rectangular : GemminiPlan := { p32 with m := 16, n := 32, k := 48 }

example : baselineProgram p32 = baselineK32 := by decide +kernel
example : batchedReuseProgram p32 2 = reuseK32 := by decide +kernel
example : (lowerBaseline p32).isSome = true := by decide +kernel
example : (lowerBReuse p32 2).isSome = true := by decide +kernel
example : (lowerBReuse p48 2).isSome = true := by decide +kernel
example : (lowerBReuse p48 3).isSome = false := by decide +kernel
example : (lowerBaseline rectangular).isSome = true := by decide +kernel
example : (lowerBReuse rectangular 1).isSome = true := by decide +kernel
example : (reuseOperand p48 (baselineProgram p48) 2).isSome = true := by decide +kernel

#print axioms Symbolic.check_sound
#print axioms checkedLowering_sound
#print axioms checkedRewrite_sound
#print axioms checkExecutable_sound
#print axioms k32Result_exact
#print axioms run_baselineK32_observe
#print axioms run_reuseK32_observe
