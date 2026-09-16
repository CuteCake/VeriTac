import VeriTac.Gemmini.Engine
import VeriTac.Gemmini.Word

namespace VeriTac.Gemmini
open scoped BigOperators

abbrev Tile := Nat → Nat → Int

/-- Aligned full tiles are the supported storage granularity. Distinct aligned
row addresses cannot overlap. Host A/B/C buffers are separate regions; the byte
layer must validate their concrete bases and sizes. Loads/stores complete in
issue order in this declared sequential target model. -/
abbrev MachineState := GState Int Int

def initialState : MachineState := initialGState 0

def tileProduct (a b : Tile) (r c : Nat) : Int :=
  ∑ t ∈ Finset.range 16, a r t * b t c

/-- One actual instruction transition. Integer-width behavior is explicit in
compute through signed32. An absent tile cannot be loaded/accumulated/stored. -/
def step (p : GemminiPlan) (a b : Nat → Int)
    (s : MachineState) : Instr → Option MachineState
  | .configEx flow =>
      if flow == 1 then some { s with ws := true } else none
  | .configLd slot stride identity =>
      if !identity then none
      else if slot == 0 then some { s with ld0 := stride }
      else if slot == 1 then some { s with ld1 := stride } else none
  | .configSt stride =>
      if stride > 0 && stride % 4 == 0 then some { s with stStride := stride } else none
  | .mvin slot buf offset row cols rows => do
      let stride ← if slot == 0 then some s.ld0 else if slot == 1 then some s.ld1 else none
      let size := match buf with | .a => p.m * p.k | .b => p.k * p.n
      let src := match buf with | .a => a | .b => b
      if stride > 0 && cols == 16 && rows == 16 &&
          offset + 15 * stride + 16 ≤ size && tileFits row p.scratchpadRows then
        let tile : Tile := fun r c => src (offset + r * stride + c)
        some { s with spad := (fun x => if x = row then some tile else s.spad x), drained := false }
      else none
  | .preload bd out =>
      if (bd == 4294967295 || tileFits bd p.scratchpadRows) &&
          accAddress out p.accumulatorRows then
        some { s with pending := some (bd, out) } else none
  | .compute retained aRow bd => do
      if !s.ws || bd != 4294967295 then none else do
        let (bRow, out) ← s.pending
        let aTile ← s.spad aRow
        let bTile ← if retained then
          if bRow == 4294967295 then s.weights else none
          else s.spad bRow
        let old ← if accumulates out then s.acc (accRow out) else some (fun _ _ => 0)
        let result : Tile := fun r c => signed32 (old r c + tileProduct aTile bTile r c)
        some { s with acc := (fun row => if row = accRow out then some result else s.acc row), weights := some bTile, pending := none, drained := false }
  | .mvout offset addr cols rows => do
      if cols != 16 || rows != 16 || s.stStride == 0 || s.stStride % 4 != 0 ||
          !(accAddress addr p.accumulatorRows) then none else do
        let stride := s.stStride / 4
        if stride < 16 || offset + 15 * stride + 16 > p.m * p.n then none else do
          let tile ← s.acc (accRow addr)
          let covers := fun idx => decide (offset ≤ idx ∧
            (idx - offset) / stride < 16 ∧ (idx - offset) % stride < 16)
          let output := fun idx => if covers idx then
            tile ((idx - offset) / stride) ((idx - offset) % stride) else s.output idx
          some { s with output := output, written := (fun idx => covers idx || s.written idx), drained := false }
  | .fence => some { s with drained := true }

def runFrom (p : GemminiPlan) (a b : Nat → Int) : Program → MachineState → Option MachineState
  | [], s => some s
  | i :: rest, s => (step p a b s i).bind (runFrom p a b rest)

def runProgram (p : GemminiPlan) (a b : Nat → Int) (program : Program) : Option MachineState :=
  runFrom p a b program initialState

/-- Row-major mathematical GEMM, independent of any instruction program. -/
def gemm (p : GemminiPlan) (a b : Nat → Int) (r c : Nat) : Int :=
  GemminiExact.dot (fun t => a (r * p.k + t)) (fun t => b (t * p.n + c)) p.k

def intOps : EvalOps Int Int := {
  zero := 0
  mac := fun old a b => signed32 (old + ∑ t ∈ Finset.range 16, a t * b t)
}

theorem gstep_int (p : GemminiPlan) (a b : Nat → Int) (s : MachineState) (i : Instr) :
    gstep intOps p a b s i = step p a b s i := by
  rfl

theorem gRunFrom_int (p : GemminiPlan) (a b : Nat → Int) (program : Program) (s : MachineState) :
    gRunFrom intOps p a b program s = runFrom p a b program s := by
  induction program generalizing s with
  | nil => rfl
  | cons i rest ih => simp [gRunFrom, runFrom, gstep_int, ih]

theorem gRunProgram_int (p : GemminiPlan) (a b : Nat → Int) (program : Program) :
    gRunProgram intOps p a b program = runProgram p a b program := by
  exact gRunFrom_int p a b program initialState

end VeriTac.Gemmini
