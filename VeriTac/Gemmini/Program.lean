import VeriTac.Gemmini.Plan

namespace VeriTac.Gemmini

inductive InputBuffer where
  | a | b
  deriving Repr, DecidableEq, BEq

/-- Full-tile, weight-stationary command subset. Addresses are byte offsets for
host buffers and row addresses for on-chip storage, just as in the emitter. -/
inductive Instr where
  | configEx (dataflow : Nat)
  | configLd (slot strideBytes : Nat) (identityScale : Bool)
  | configSt (strideBytes : Nat)
  | mvin (slot : Nat) (buf : InputBuffer) (offset spadAddr cols rows : Nat)
  | preload (bdSpadAddr outAddr : Nat)
  | compute (accumulated : Bool) (aSpadAddr bdSpadAddr : Nat)
  | mvout (bufOffset accAddr cols rows : Nat)
  | fence
  deriving Repr, DecidableEq, BEq

abbrev Program := List Instr

def baseline16 : Program := [
  .configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
  .mvin 0 .a 0 0 16 16, .mvin 1 .b 0 16 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16, .fence]

/-- At one output-row tile, the reuse schedule differs only in load order. -/
def reuse16 : Program := [
  .configEx 1, .configSt 64, .configLd 0 16 true, .configLd 1 16 true,
  .mvin 1 .b 0 16 16 16, .mvin 0 .a 0 0 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16, .fence]

/-- The first certificate covers a fixed shape, all int8 input values, and both
one-tile command orders. Broader shapes must fail closed until proved. -/
def checkProgram (p : GemminiPlan) (program : Program) : Bool :=
  checkGemminiPlan p && p.m == 16 && p.n == 16 && p.k == 16 && p.dim == 16 &&
  (match p.schedule with
   | .baseline => decide (program = baseline16)
   | .reuseB => decide (program = reuse16))

end VeriTac.Gemmini
