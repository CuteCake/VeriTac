import VeriTac.Gemmini.Program

namespace VeriTac.Gemmini

/-- Two output-row tiles and one K tile: the smallest nontrivial B-reuse case. -/
def baseline32 : Program := [
  .configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
  .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 256 0 16 16, .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvout 256 3758096384 16 16, .fence]

def reuse32 : Program := [
  .configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
  .mvin 1 .b 0 16 16 16, .mvin 0 .a 0 0 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 256 0 16 16,
  .preload 4294967295 2684354576, .compute true 0 4294967295,
  .mvout 256 3758096400 16 16, .fence]

def checkProgram32 (p : GemminiPlan) (program : Program) : Bool :=
  checkGemminiPlan p && p.m == 32 && p.n == 16 && p.k == 16 && p.dim == 16 &&
  (match p.schedule with
   | .baseline => decide (program = baseline32)
   | .reuseB => decide (program = reuse32))

def checkSupportedProgram (p : GemminiPlan) (program : Program) : Bool :=
  checkProgram p program || checkProgram32 p program

end VeriTac.Gemmini
