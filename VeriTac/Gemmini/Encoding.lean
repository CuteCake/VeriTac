import Lean

/-!
Operational semantics for the restricted RV64 kernel body. Words are decoded
from the submitted little-endian bytes. Register arithmetic wraps modulo 2^64;
LUI and ADDI sign extension follows RV64. The custom instruction emits the
operand VALUES it reads. Initial scratch registers are unknown, not assumed
zero. The caller supplies x1 for the final return, and mapped A/B/C buffers.
The trace ends only at fence iorw,iorw followed by ret. Hardware correspondence
and caller/loader behavior remain declared assumptions.
-/
namespace VeriTac.Gemmini.Encoding

structure OpPacket where
  funct : Nat
  rs1 : Nat
  rs2 : Nat
  deriving Repr, DecidableEq, BEq

structure Bases where
  a : Nat
  b : Nat
  c : Nat
  deriving Repr, DecidableEq, BEq

def modulus : Nat := 18446744073709551616

def disjoint (a sizeA b sizeB : Nat) : Bool := a + sizeA ≤ b || b + sizeB ≤ a

def checkBases (b : Bases) (sizes : Nat × Nat × Nat) : Bool :=
  let (sa, sb, sc) := sizes
  sa > 0 && sb > 0 && sc > 0 && b.a % 4 == 0 && b.b % 4 == 0 && b.c % 4 == 0 &&
  b.a + sa ≤ modulus && b.b + sb ≤ modulus && b.c + sc ≤ modulus &&
  disjoint b.a sa b.b sb && disjoint b.a sa b.c sc && disjoint b.b sb b.c sc

/-- Only these caller-clobbered registers exist in the restricted body.
A finite record avoids a growing chain of functional updates during kernel replay. -/
structure Registers where
  r5 : Option Nat := none
  r6 : Option Nat := none
  r31 : Option Nat := none
  deriving Repr

def initialRegisters : Registers := {}

def readable (r : Nat) : Bool := r == 0 || r == 5 || r == 6 || r == 31

def writable (r : Nat) : Bool := r == 5 || r == 6 || r == 31

def readRegister (regs : Registers) (r : Nat) : Except String Nat :=
  if !readable r then .error "register outside supported calling convention"
  else
    let value := if r = 0 then some 0 else if r = 5 then regs.r5
      else if r = 6 then regs.r6 else regs.r31
    match value with
    | some v => .ok v
    | none => .error "read of uninitialized scratch register"

def writeRegister (regs : Registers) (r value : Nat) : Except String Registers :=
  if r = 5 then .ok { regs with r5 := some (value % modulus) }
  else if r = 6 then .ok { regs with r6 := some (value % modulus) }
  else if r = 31 then .ok { regs with r31 := some (value % modulus) }
  else .error "write outside caller-clobbered scratch registers"

/-- Sign extend a bits-wide value into the RV64 unsigned representation. -/
def sext (value bits : Nat) : Nat :=
  if value < 2 ^ (bits - 1) then value else value + modulus - 2 ^ bits

/-- Decode and execute one host/custom word. Only instructions represented here
belong to the accepted byte subset. A custom packet contains live register values. -/
def executeWord (regs : Registers) (w : Nat) : Except String (Registers × Option OpPacket) := do
  if w ≥ 4294967296 then .error "instruction word exceeds 32 bits" else do
    let opcode := w % 128
    let rd := (w / 128) % 32
    let f3 := (w / 4096) % 8
    let rs1 := (w / 32768) % 32
    let rs2 := (w / 1048576) % 32
    if opcode == 123 then
      if rd != 0 || f3 != 3 || rs1 != 5 || rs2 != 6 || w / 33554432 > 6 then
        .error "unsupported custom instruction fields"
      else do
        let a ← readRegister regs rs1
        let b ← readRegister regs rs2
        pure (regs, some { funct := w / 33554432, rs1 := a, rs2 := b })
    else do
      let value ←
        if opcode == 55 then
          pure (sext ((w / 4096) * 4096) 32)
        else if opcode == 19 then do
          let a ← readRegister regs rs1
          if f3 == 0 then pure ((a + sext (w / 1048576) 12) % modulus)
          else if (f3 == 1 || f3 == 5) && w / 67108864 == 0 then
            let shift := (w / 1048576) % 64
            if f3 == 1 then pure ((a * 2 ^ shift) % modulus) else pure (a / 2 ^ shift)
          else .error "unsupported immediate instruction"
        else if opcode == 51 && f3 == 0 && w / 33554432 == 0 then do
          let a ← readRegister regs rs1
          let b ← readRegister regs rs2
          pure ((a + b) % modulus)
        else .error "unsupported RV64 instruction"
      let regs' ← writeRegister regs rd value
      pure (regs', none)

/-- Full fence and return are checked as the exact final two words. Neither can
occur earlier. Return preserves caller x1; all writes above target x5/x6/x31. -/
def executeWords : List Nat → Registers → List OpPacket → Except String (List OpPacket)
  | [], _, _ => .error "missing final fence and return"
  | [_], _, _ => .error "missing final fence and return"
  | [fence, ret], _, trace =>
      if fence == 267386895 && ret == 32871 then .ok trace.reverse
      else .error "missing final fence and return"
  | w :: rest, regs, trace => do
      let (regs', packet) ← executeWord regs w
      executeWords rest regs' (match packet with | none => trace | some p => p :: trace)

def bytesToWords : List UInt8 → Except String (List Nat)
  | [] => .ok []
  | a :: b :: c :: d :: rest => do
      let tail ← bytesToWords rest
      pure ((a.toNat + b.toNat * 256 + c.toNat * 65536 + d.toNat * 16777216) :: tail)
  | _ => .error "truncated little-endian instruction word"

def executeBytes (bytes : List UInt8) : Except String (List OpPacket) := do
  let words ← bytesToWords bytes
  executeWords words initialRegisters []

/-- Range and alias assumptions are validated separately from execution. Full
DMA spans are established by the checked command semantics and exact packet
comparison, rather than checking only the first byte of each DMA here. -/
def decode (bases : Bases) (sizes : Nat × Nat × Nat) (bytes : List UInt8) :
    Except String (List OpPacket) :=
  if checkBases bases sizes then executeBytes bytes else .error "invalid buffer ranges"

def checkEncoding (bases : Bases) (sizes : Nat × Nat × Nat) (bytes : List UInt8)
    (expected : List OpPacket) : Bool :=
  match decode bases sizes bytes with
  | .error _ => false
  | .ok actual => decide (actual = expected)

/-- True acceptance means actual execution of the submitted bytes reaches the
expected custom operand trace and the required final fence/return. -/
theorem checkEncoding_sound (bases : Bases) (sizes : Nat × Nat × Nat) (bytes : List UInt8)
    (expected : List OpPacket) (h : checkEncoding bases sizes bytes expected = true) :
    checkBases bases sizes = true ∧ executeBytes bytes = .ok expected := by
  have hd : decode bases sizes bytes = .ok expected := by
    unfold checkEncoding at h
    cases he : decode bases sizes bytes with
    | error message => simp [he] at h
    | ok actual =>
      have hsame : actual = expected := of_decide_eq_true (by simpa [he] using h)
      simpa [hsame] using he
  unfold decode at hd
  split at hd
  · exact ⟨by assumption, hd⟩
  · contradiction

end VeriTac.Gemmini.Encoding
