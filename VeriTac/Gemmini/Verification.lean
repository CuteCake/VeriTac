import VeriTac.Gemmini.Program32
import VeriTac.Gemmini.Symbolic
import VeriTac.Gemmini.Encoding

/-! Executable acceptance definitions. Proof modules import these exact
constants; the native checker does not need to link the proof library. -/
namespace VeriTac.Gemmini
open Encoding

def packedTile (addr : Nat) : Nat := 4503668346847232 + addr

def packetOf (bases : Bases) : Instr → Option OpPacket
  | .configEx flow => some ⟨0, 4575657221408489472 + flow * 4, 281474976710656⟩
  | .configLd slot stride _ => some ⟨0, 4575657221409472769 + slot * 8, stride⟩
  | .configSt stride => some ⟨0, 2, 4575657221408423936 + stride⟩
  | .mvin slot buf offset row _ _ =>
      some ⟨if slot = 0 then 2 else 1,
        (match buf with | .a => bases.a | .b => bases.b) + offset, packedTile row⟩
  | .preload bd out => some ⟨6, packedTile bd, packedTile out⟩
  | .compute retained a bd => some ⟨if retained then 5 else 4, packedTile a, packedTile bd⟩
  | .mvout offset addr _ _ => some ⟨3, bases.c + 4 * offset, packedTile addr⟩
  | .fence => none

def packetize (bases : Bases) (program : Program) : List OpPacket := program.filterMap (packetOf bases)

/-- The full-tile packing is decoded, rather than recovering commands from the
request's claimed program. Other shapes and scaling modes fail closed. -/
def unpackTile (operand : Nat) : Option Nat :=
  if operand / 4294967296 = 1048592 then some (operand % 4294967296) else none

def commandOf (bases : Bases) (p : OpPacket) : Option Instr := do
  if p.funct = 0 then
    if p.rs1 = 4575657221408489476 && p.rs2 = 281474976710656 then some (.configEx 1)
    else if p.rs1 = 4575657221409472769 then some (.configLd 0 p.rs2 true)
    else if p.rs1 = 4575657221409472777 then some (.configLd 1 p.rs2 true)
    else if p.rs1 = 2 && p.rs2 / 4294967296 = 1065353216 then
      some (.configSt (p.rs2 % 4294967296))
    else none
  else if p.funct = 1 || p.funct = 2 then do
    let row ← unpackTile p.rs2
    let base := if p.funct = 2 then bases.a else bases.b
    if base ≤ p.rs1 then
      some (.mvin (if p.funct = 2 then 0 else 1) (if p.funct = 2 then .a else .b)
        (p.rs1 - base) row 16 16)
    else none
  else if p.funct = 3 then do
    let row ← unpackTile p.rs2
    if bases.c ≤ p.rs1 && (p.rs1 - bases.c) % 4 = 0 then
      some (.mvout ((p.rs1 - bases.c) / 4) row 16 16)
    else none
  else if p.funct = 4 || p.funct = 5 then do
    let a ← unpackTile p.rs1
    let bd ← unpackTile p.rs2
    some (.compute (p.funct == 5) a bd)
  else if p.funct = 6 then do
    let bd ← unpackTile p.rs1
    let out ← unpackTile p.rs2
    some (.preload bd out)
  else none

def commandsOf (bases : Bases) (packets : List OpPacket) : Option Program := do
  let commands ← packets.mapM (commandOf bases)
  pure (commands ++ [.fence])

def bufferSizes (p : GemminiPlan) : Nat × Nat × Nat := (p.m * p.k, p.k * p.n, 4 * p.m * p.n)

/-- Executable validation is shape-parameterized: symbolic execution proves
the submitted command program, then actual byte decoding must recover it. -/
def checkExecutable (p : GemminiPlan) (program : Program) (bases : Bases)
    (bytes : List UInt8) : Bool :=
  (p.scratchpadRows ≤ 16384 && p.accumulatorRows ≤ 1024) &&
  Symbolic.check p program &&
  match Encoding.decode bases (bufferSizes p) bytes with
  | .error _ => false
  | .ok packets =>
    match commandsOf bases packets with
    | none => false
    | some decoded => decide (decoded = program)


end VeriTac.Gemmini
