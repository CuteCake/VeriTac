import VeriTac.Gemmini.Program

/-! Shared instruction control/address semantics. Arithmetic is a parameter;
numeric and symbolic instances use these exact same transitions. -/
namespace VeriTac.Gemmini

abbrev ValueTile (α : Type) := Nat → Nat → α

structure GState (α β : Type) where
  ws : Bool := false
  ld0 : Nat := 0
  ld1 : Nat := 0
  stStride : Nat := 0
  spad : Nat → Option (ValueTile α) := fun _ => none
  acc : Nat → Option (ValueTile β) := fun _ => none
  pending : Option (Nat × Nat) := none
  weights : Option (ValueTile α) := none
  output : Nat → β
  written : Nat → Bool := fun _ => false
  drained : Bool := true

def initialGState {α β : Type} (zero : β) : GState α β := { output := fun _ => zero }

def tileFits (row capacity : Nat) : Bool := row % 16 == 0 && row + 16 ≤ capacity

def accRow (addr : Nat) : Nat := addr % 536870912

def accAddress (addr capacity : Nat) : Bool :=
  addr < 4294967296 && addr / 2147483648 == 1 &&
  (addr / 536870912) % 2 == 1 && tileFits (accRow addr) capacity

def accumulates (addr : Nat) : Bool := (addr / 1073741824) % 2 == 1

structure EvalOps (α β : Type) where
  zero : β
  mac : β → (Nat → α) → (Nat → α) → β

def gstep {α β : Type} (ops : EvalOps α β) (p : GemminiPlan) (a b : Nat → α)
    (s : GState α β) : Instr → Option (GState α β)
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
        let tile : ValueTile α := fun r c => src (offset + r * stride + c)
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
        let old ← if accumulates out then s.acc (accRow out) else some (fun _ _ => ops.zero)
        let result : ValueTile β := fun r c => ops.mac (old r c) (fun t => aTile r t) (fun t => bTile t c)
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

def gRunFrom {α β : Type} (ops : EvalOps α β) (p : GemminiPlan) (a b : Nat → α) :
    Program → GState α β → Option (GState α β)
  | [], s => some s
  | i :: rest, s => (gstep ops p a b s i).bind (gRunFrom ops p a b rest)

def gRunProgram {α β : Type} (ops : EvalOps α β) (p : GemminiPlan)
    (a b : Nat → α) (program : Program) : Option (GState α β) :=
  gRunFrom ops p a b program (initialGState ops.zero)

end VeriTac.Gemmini
