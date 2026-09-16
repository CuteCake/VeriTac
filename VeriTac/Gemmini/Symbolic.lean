import VeriTac.Gemmini.Engine

/-! A shape-parameterized translation validator. A loaded value names an input
element. An accumulator carries an ordered list of products modulo int32.
Acceptance compares EVERY written output with the GEMM product list. No
catalogue of accepted programs or fixed shapes appears in this checker. -/
namespace VeriTac.Gemmini.Symbolic

inductive InputRef where
  | a (index : Nat)
  | b (index : Nat)
  deriving Repr, DecidableEq, BEq

abbrev Term := InputRef × InputRef
abbrev Terms := List Term
abbrev State := GState InputRef Terms

def ops : EvalOps InputRef Terms := {
  zero := []
  mac := fun old a b => old ++ (List.range 16).map (fun t => (a t, b t))
}

def run (p : GemminiPlan) (program : Program) : Option State :=
  gRunProgram ops p InputRef.a InputRef.b program

def expected (p : GemminiPlan) (r c : Nat) : Terms :=
  (List.range p.k).map (fun t => (InputRef.a (r * p.k + t), InputRef.b (t * p.n + c)))

def cellsCorrect (p : GemminiPlan) (s : State) : Bool :=
  (List.range p.m).all fun r => (List.range p.n).all fun c =>
    s.written (r * p.n + c) && decide (s.output (r * p.n + c) = expected p r c)

def check (p : GemminiPlan) (program : Program) : Bool :=
  checkGemminiPlan p && p.dim == 16 &&
  match run p program with
  | none => false
  | some s => s.drained && cellsCorrect p s

def diagnostic (p : GemminiPlan) (program : Program) : String :=
  if !checkGemminiPlan p then planDiagnostic p
  else if p.dim != 16 then "instruction model requires DIM=16"
  else match run p program with
    | none => "instruction execution rejected: check addresses, strides, initialization, preload and retained weights"
    | some s =>
      if !s.drained then "completion obligation failed: missing final fence"
      else if cellsCorrect p s then "accepted"
      else "GEMM output obligation failed: a cell is unwritten or its input-product sequence differs from the specification"

end VeriTac.Gemmini.Symbolic
