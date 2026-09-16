import VeriTac.Gemmini.Symbolic

/-! Parameterized command proposals with proof-carrying validation.
Changing dimensions or row-batch size never requires a new template theorem. -/
namespace VeriTac.Gemmini

def gemmPreamble (p : GemminiPlan) : Program :=
  [.configEx 1, .configSt (4 * p.n), .configLd 0 p.k true, .configLd 1 p.n true]

def outputAddress (kBlock row : Nat) : Nat :=
  (if kBlock = 0 then 2684354560 else 3758096384) + row

def baselineProgram (p : GemminiPlan) : Program :=
  gemmPreamble p ++
  ((List.range (p.m / 16)).flatMap fun i =>
    (List.range (p.n / 16)).flatMap fun j =>
      ((List.range (p.k / 16)).flatMap fun k => [
        .mvin 0 .a ((i * p.k + k) * 16) 0 16 16,
        .mvin 1 .b ((k * p.n + j) * 16) 16 16 16,
        .preload 16 (outputAddress k 0),
        .compute false 0 4294967295]) ++
      [.mvout ((i * p.n + j) * 16) 3758096384 16 16]) ++ [.fence]

/-- Reuse each B tile across a batch of output-row tiles. The final batch may
be shorter. Its live partial sums use local accumulator rows, reused per batch. -/
def batchedReuseProgram (p : GemminiPlan) (batchRows : Nat) : Program :=
  if batchRows = 0 then [] else
  gemmPreamble p ++
  ((List.range (p.n / 16)).flatMap fun j =>
    (List.range ((p.m / 16 + batchRows - 1) / batchRows)).flatMap fun batch =>
      let first := batch * batchRows
      let count := min batchRows (p.m / 16 - first)
      (List.range (p.k / 16)).flatMap fun k =>
        [.mvin 1 .b ((k * p.n + j) * 16) 16 16 16] ++
        ((List.range count).flatMap fun localRow =>
          let i := first + localRow
          [.mvin 0 .a ((i * p.k + k) * 16) 0 16 16,
           .preload (if localRow = 0 then 16 else 4294967295) (outputAddress k (localRow * 16)),
           .compute (localRow != 0) 0 4294967295] ++
          if k + 1 = p.k / 16 then
            [.mvout ((i * p.n + j) * 16) (3758096384 + localRow * 16) 16 16]
          else [])) ++ [.fence]

abbrev CheckedProgram (p : GemminiPlan) := { program : Program // Symbolic.check p program = true }

def validateProgram (p : GemminiPlan) (program : Program) : Option (CheckedProgram p) :=
  if h : Symbolic.check p program = true then some ⟨program, h⟩ else none

def lowerBaseline (p : GemminiPlan) : Option (CheckedProgram p) :=
  validateProgram p (baselineProgram p)

def lowerBReuse (p : GemminiPlan) (batchRows : Nat) : Option (CheckedProgram p) :=
  validateProgram p (batchedReuseProgram p batchRows)

structure CheckedRewrite (p : GemminiPlan) (source : Program) where
  destination : Program
  sourceAccepted : Symbolic.check p source = true
  destinationAccepted : Symbolic.check p destination = true

/-- A reusable optimization interface: arbitrary proposals are accepted only
after checking both endpoints under the same workload and target contract. -/
def validateRewrite (p : GemminiPlan) (source destination : Program) :
    Option (CheckedRewrite p source) :=
  if hs : Symbolic.check p source = true then
    if hd : Symbolic.check p destination = true then some ⟨destination, hs, hd⟩
    else none
  else none

def reuseOperand (p : GemminiPlan) (source : Program) (batchRows : Nat) :
    Option (CheckedRewrite p source) :=
  validateRewrite p source (batchedReuseProgram p batchRows)

end VeriTac.Gemmini
