/-
  GemminiProgramMain: JSON CLI for CONCRETE Gemmini program acceptance.

  Unlike `gemmini_check` (plan legality only), this binary accepts one JSON
  object describing a full concrete generated program:

    {"schema":   "veritac_program_request_v1",
     "plan":     { ...GemminiPlan fields... },
     "commands": [ ...exact command stream emitted by CodeGen/gemmini.py... ],
     "encoding": { ...executable-byte artifact (decoder-defined schema)... }}

  and emits exactly one JSON object:

    {"accepted": bool, "reason": string,
     "program": {"plan": ..., "commands": ..., "encoding": ...},  -- accepted only
     "certificate": {...}}                                        -- accepted only

  Fail-closed contract: malformed JSON, missing/unknown fields, noninteger
  numbers, unknown instruction kinds, absent checkers, decoder rejection, or
  any mismatch between the supplied commands and the decoded executable bytes
  is rejected with `accepted: false` (exit code 0).  Acceptance NEVER rests on
  plan legality or hash equality alone: the proved program checker and the
  proved byte decoder must both accept the concrete artifact.  While those
  proved modules are unavailable this CLI rejects every request (fail closed);
  it never substitutes a weaker check.
-/
import Lean.Data.Json
import VeriTac.Gemmini.Verification

open Lean (Json JsonNumber)
open Lean.ToJson
open VeriTac.Gemmini
open VeriTac.Gemmini.Encoding

/-! ## Strict JSON scalar parsing -/

/-- A JSON number is a strict integer exactly when `mantissa / 10^exponent`
    is exact; large exponents are rejected fail-closed. -/
def progJsonStrictInt? (n : JsonNumber) : Option Int :=
  if n.exponent > 30 then none
  else
    let scale : Int := (10 : Int) ^ n.exponent
    if n.mantissa % scale == 0 then some (n.mantissa / scale) else none

/-- Read field `name` as a strict integer. -/
def progIntField? (j : Json) (name : String) : Option Int := do
  let v ← j.getObjVal? name |>.toOption
  match v with
  | .num n => progJsonStrictInt? n
  | _ => none

/-- Strict nonnegative integer field. -/
def progNatField? (j : Json) (name : String) : Option Nat := do
  let v ← progIntField? j name
  if v < 0 then none else some v.toNat

/-- The object has exactly the given field names. -/
def objFieldsExactly (j : Json) (names : List String) : Bool :=
  match j with
  | .obj o => o.toList.length == names.length ∧ names.all (o.contains ·)
  | _ => false

/-- Convert an `Option` into an `Except` with a failure message. -/
def optToExcept {α : Type} (o : Option α) (e : String) : Except String α :=
  match o with
  | some a => .ok a
  | none => .error e

/-! ## Plan parsing (same strictness as GemminiMain) -/

def progParseSchedule (s : String) : Option GemminiSchedule :=
  match s with
  | "baseline" => some .baseline
  | "reuse_b" => some .reuseB
  | _ => none

def progParsePlan (j : Json) : Option GemminiPlan := do
  guard (objFieldsExactly j ["m", "n", "k", "dim", "scratchpad_rows",
    "accumulator_rows", "schedule"])
  let m ← progNatField? j "m"
  let n ← progNatField? j "n"
  let k ← progNatField? j "k"
  let dim ← progNatField? j "dim"
  let sp ← progNatField? j "scratchpad_rows"
  let acc ← progNatField? j "accumulator_rows"
  let sched ← match j.getObjValAs? String "schedule" with
    | .ok s => progParseSchedule s | .error _ => none
  pure { m, n, k, dim, scratchpadRows := sp, accumulatorRows := acc,
         schedule := sched }

def progPlanToJson (p : GemminiPlan) : Json :=
  .mkObj [("m", toJson p.m), ("n", toJson p.n), ("k", toJson p.k),
    ("dim", toJson p.dim),
    ("scratchpad_rows", toJson p.scratchpadRows),
    ("accumulator_rows", toJson p.accumulatorRows),
    ("schedule", match p.schedule with
      | .baseline => .str "baseline" | .reuseB => .str "reuse_b")]

/-! ## Command parsing (strict, exhaustive, unknown kinds rejected) -/

/-- The exact instruction kinds of the supported subset. -/
inductive ProgCmdKind where
  | configEx | configLd | configSt | mvin | preload | compute | mvout | fence
  deriving Repr, DecidableEq, Inhabited

def progCmdKindName : ProgCmdKind → String
  | .configEx => "config_ex"
  | .configLd => "config_ld"
  | .configSt => "config_st"
  | .mvin => "mvin"
  | .preload => "preload"
  | .compute => "compute"
  | .mvout => "mvout"
  | .fence => "fence"

def progCmdKindFromString (s : String) : Option ProgCmdKind :=
  match s with
  | "config_ex" => some .configEx
  | "config_ld" => some .configLd
  | "config_st" => some .configSt
  | "mvin" => some .mvin
  | "preload" => some .preload
  | "compute" => some .compute
  | "mvout" => some .mvout
  | "fence" => some .fence
  | _ => none

/-- Parsed concrete command.  Fields mirror the Python generator's JSON
    exactly; strictness of every field is enforced at parse time. -/
inductive ProgCmd where
  | configEx : (dataflow : Nat) → ProgCmd
  | configLd : (slot : Nat) → (strideBytes : Nat) → ProgCmd
  | configSt : (strideBytes : Nat) → ProgCmd
  | mvin : (slot : Nat) → (buf : Bool) → (offset : Nat) → (spadAddr : Nat) →
      (cols : Nat) → (rows : Nat) → ProgCmd
  | preload : (bdSpadAddr : Nat) → (outAddr : Nat) → ProgCmd
  | compute : (accumulated : Bool) → (aSpadAddr : Nat) → (bdSpadAddr : Nat) → ProgCmd
  | mvout : (bufOffset : Nat) → (accAddr : Nat) → (cols : Nat) → (rows : Nat) → ProgCmd
  | fence : ProgCmd
  deriving Repr, DecidableEq, Inhabited

/-- Strict uint32 field (addrs may carry bit31/30/29 flags). -/
def progUint32Field? (j : Json) (name : String) : Option Nat := do
  let v ← progNatField? j name
  if v ≤ 0xFFFFFFFF then some v else none

/-- Strict field that must be the JSON number exactly 1 (scale 1.0 or 1). -/
def progScaleOneField? (j : Json) (name : String) : Option Unit := do
  let v ← j.getObjVal? name |>.toOption
  match v with
  | .num n => guard (progJsonStrictInt? n == some 1)
  | _ => failure

/-- Parse one command object strictly: exact field set per kind, no extras,
    no unknown kinds. -/
def progParseCmd (j : Json) : Option ProgCmd := do
  let kindStr ← match j.getObjValAs? String "kind" with
    | .ok s => some s | .error _ => none
  let kind ← progCmdKindFromString kindStr
  match kind with
  | .configEx =>
    guard (objFieldsExactly j ["kind", "dataflow"])
    let d ← progNatField? j "dataflow"
    pure (.configEx d)
  | .configLd =>
    guard (objFieldsExactly j ["kind", "slot", "stride_bytes", "scale"])
    let slot ← progNatField? j "slot"
    guard (slot ≤ 2)
    let stride ← progNatField? j "stride_bytes"
    let _ ← progScaleOneField? j "scale"
    pure (.configLd slot stride)
  | .configSt =>
    guard (objFieldsExactly j ["kind", "stride_bytes"])
    let stride ← progNatField? j "stride_bytes"
    pure (.configSt stride)
  | .mvin =>
    guard (objFieldsExactly j ["kind", "slot", "buf", "offset", "spad_addr",
      "cols", "rows"])
    let slot ← progNatField? j "slot"
    guard (slot ≤ 1)
    let buf ← match j.getObjValAs? String "buf" with
      | .ok "A" => some true | .ok "B" => some false | _ => none
    let off ← progNatField? j "offset"
    let spad ← progNatField? j "spad_addr"
    let cols ← progNatField? j "cols"
    let rows ← progNatField? j "rows"
    pure (.mvin slot buf off spad cols rows)
  | .preload =>
    guard (objFieldsExactly j ["kind", "bd_spad_addr", "out_addr"])
    let bd ← progUint32Field? j "bd_spad_addr"
    let out ← progUint32Field? j "out_addr"
    pure (.preload bd out)
  | .compute =>
    guard (objFieldsExactly j ["kind", "accumulated", "a_spad_addr",
      "bd_spad_addr"])
    let accumulated ← match j.getObjVal? "accumulated" |>.toOption with
      | some (.bool b) => some b | _ => none
    let a ← progNatField? j "a_spad_addr"
    let bd ← progUint32Field? j "bd_spad_addr"
    pure (.compute accumulated a bd)
  | .mvout =>
    guard (objFieldsExactly j ["kind", "buf_offset", "acc_addr", "cols",
      "rows"])
    let off ← progNatField? j "buf_offset"
    let acc ← progUint32Field? j "acc_addr"
    let cols ← progNatField? j "cols"
    let rows ← progNatField? j "rows"
    pure (.mvout off acc cols rows)
  | .fence =>
    guard (objFieldsExactly j ["kind"])
    pure .fence

def progParseCommands (j : Json) : Option (List ProgCmd) := do
  match j with
  | .arr xs => xs.toList.mapM progParseCmd
  | _ => none

/-! ## Encoding artifact parsing (schema per encoding-contract.md, worker2) -/

/-- Parsed encoding artifact. -/
structure ProgEncoding where
  bases : Nat × Nat × Nat          -- (A, B, C) byte addresses
  sizes : Nat × Nat × Nat          -- (A, B, C) byte extents
  bytes : List UInt8               -- full executable image, little-endian words
  bytesHex : String
  words : Nat

/-- Hex digit value; uppercase is rejected (emitter contract is lowercase). -/
def hexCharVal? (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else none

/-- Parse eight hexadecimal characters as four bytes in memory order.
    bytes_hex is byte serialization, not a printed numeric instruction word. -/
def hexWordBytes? (s : String) : Option (List UInt8) := do
  let ds ← s.toList.mapM hexCharVal?
  match ds with
  | [a, b, c, d, e, f, g, h] =>
      some [(a * 16 + b).toUInt8, (c * 16 + d).toUInt8,
            (e * 16 + f).toUInt8, (g * 16 + h).toUInt8]
  | _ => none

/-- Split a hex string into consecutive 8-char groups (nonrecursive:
    positional slicing). -/
def hexGroups (s : String) : List String :=
  let cs := s.toList
  (List.range (cs.length / 8)).map
    (fun i => String.ofList ((cs.drop (i * 8)).take 8))

def progParseEncoding (j : Json) : Except String ProgEncoding := do
  unless objFieldsExactly j ["format", "word_size", "endianness", "regs",
      "bases", "sizes", "bytes_hex"] do
    .error "encoding must have exactly the fields format/word_size/endianness/regs/bases/sizes/bytes_hex"
  let format ← match j.getObjValAs? String "format" with
    | .ok s => .ok s | .error _ => .error "encoding: missing or non-string 'format'"
  if format != "veritac_gemmini_bytes_v2" then
    .error s!"encoding: unknown format {format}"
  let ws ← optToExcept (progNatField? j "word_size") "encoding: bad word_size"
  if ws != 4 then .error "encoding: word_size must be 4"
  let endi ← match j.getObjValAs? String "endianness" with
    | .ok s => .ok s | .error _ => .error "encoding: missing or non-string 'endianness'"
  if endi != "little" then .error "encoding: endianness must be little"
  let regs ← optToExcept (j.getObjVal? "regs" |>.toOption) "encoding: missing 'regs'"
  unless objFieldsExactly regs ["rs1", "rs2", "temp"] do
    .error "encoding.regs must have exactly rs1/rs2/temp"
  let rs1 ← optToExcept (progNatField? regs "rs1") "encoding.regs: bad rs1"
  let rs2 ← optToExcept (progNatField? regs "rs2") "encoding.regs: bad rs2"
  let temp ← optToExcept (progNatField? regs "temp") "encoding.regs: bad temp"
  if rs1 != 5 ∨ rs2 != 6 ∨ temp != 31 then
    .error "encoding.regs must be rs1=5 rs2=6 temp=31"
  let parseTriple (jn : Json) (name : String) : Except String (Nat × Nat × Nat) := do
    unless objFieldsExactly jn ["A", "B", "C"] do
      .error s!"encoding.{name} must have exactly A/B/C"
    let a ← optToExcept (progNatField? jn "A") s!"encoding.{name}.A bad"
    let b ← optToExcept (progNatField? jn "B") s!"encoding.{name}.B bad"
    let c ← optToExcept (progNatField? jn "C") s!"encoding.{name}.C bad"
    pure (a, b, c)
  let bases ← optToExcept (j.getObjVal? "bases" |>.toOption) "encoding: missing 'bases'"
  let (ba, bb, bc) ← parseTriple bases "bases"
  let sizesJson ← optToExcept (j.getObjVal? "sizes" |>.toOption) "encoding: missing 'sizes'"
  let (sa, sb, sc) ← parseTriple sizesJson "sizes"
  if sa = 0 ∨ sb = 0 ∨ sc = 0 then
    .error "encoding.sizes must all be positive"
  let hex ← match j.getObjValAs? String "bytes_hex" with
    | .ok s => .ok s | .error _ => .error "encoding: missing or non-string 'bytes_hex'"
  if hex.isEmpty then .error "encoding: empty bytes_hex"
  if hex.length % 8 != 0 then
    .error "encoding: bytes_hex length must be a multiple of 8"
  if !hex.all (fun c => (hexCharVal? c).isSome) then
    .error "encoding: bytes_hex has non-hex characters"
  let groups := hexGroups hex
  if groups.length = 0 then .error "encoding: empty bytes_hex"
  let byteLists ← optToExcept (groups.mapM hexWordBytes?) "encoding: bad hex group"
  let bytes := byteLists.flatten
  let words := groups.length
  pure { bases := (ba, bb, bc), sizes := (sa, sb, sc), bytes,
         bytesHex := hex, words }

/-! ## Request parsing -/

structure ProgRequest where
  plan : GemminiPlan
  commands : List ProgCmd
  commandsJson : Json
  encoding : ProgEncoding
  encodingJson : Json
  planJson : Json

def progParseRequest (j : Json) : Except String ProgRequest := do
  unless objFieldsExactly j ["schema", "plan", "commands", "encoding"] do
    .error "request must have exactly the fields schema/plan/commands/encoding"
  let schema ← match j.getObjValAs? String "schema" with
    | .ok s => .ok s | .error _ => .error "missing or non-string field 'schema'"
  if schema != "veritac_program_request_v1" then
    .error s!"unknown schema {schema}"
  let planJson ← optToExcept (j.getObjVal? "plan" |>.toOption) "missing field 'plan'"
  let plan ← optToExcept (progParsePlan planJson) "malformed plan"
  let commandsJson ← optToExcept (j.getObjVal? "commands" |>.toOption) "missing field 'commands'"
  let commands ← optToExcept (progParseCommands commandsJson) "malformed commands"
  let encodingJson ← optToExcept (j.getObjVal? "encoding" |>.toOption) "missing field 'encoding'"
  let encoding ← optToExcept (progParseEncoding encodingJson |>.toOption)
    "malformed encoding artifact"
  pure { plan, commands, commandsJson, encoding, encodingJson, planJson }

/-! ## Command → instruction AST conversion (worker1 VeriTac.Gemmini.Instr) -/

/-- The strict JSON parser already enforces `scale == 1` for `config_ld`, so
    every converted instruction carries `identityScale = true`. -/
def progCmdToInstr : ProgCmd → Instr
  | .configEx d => .configEx d
  | .configLd slot stride => .configLd slot stride true
  | .configSt s => .configSt s
  | .mvin slot buf off sp cols rows =>
      .mvin slot (if buf then .a else .b) off sp cols rows
  | .preload bd out => .preload bd out
  | .compute acc a bd => .compute acc a bd
  | .mvout off addr cols rows => .mvout off addr cols rows
  | .fence => .fence

/-! ## The exact runtime predicate proved by checkExecutable_sound -/

def acceptProgram (req : ProgRequest) : Except String Json := do
  if !checkGemminiPlan req.plan then .error (planDiagnostic req.plan) else do
    let (ba, bb, bc) := req.encoding.bases
    let bases : Bases := ⟨ba, bb, bc⟩
    let program := req.commands.map progCmdToInstr
    if req.encoding.sizes != bufferSizes req.plan then
      .error "buffer size metadata does not match plan"
    else if !Symbolic.check req.plan program then
      .error ("program rejected: " ++ Symbolic.diagnostic req.plan program)
    else if !checkExecutable req.plan program bases req.encoding.bytes then
      .error "executable rejected: illegal capacity, decoded command mismatch, or invalid bytes/buffer ranges"
    else
      .ok <| .mkObj [
        ("program_checker", .str "VeriTac.Gemmini.checkExecutable"),
        ("soundness", .str "VeriTac.Gemmini.checkExecutable_sound"),
        ("claim", .str "submitted bytes execute to GEMM for all int8 inputs under the declared sequential target and loader contract"),
        ("bytes_hex", .str req.encoding.bytesHex),
        ("instruction_words", toJson req.encoding.words),
        ("decoded_packets", toJson (packetize bases program).length)]

def processProgram (input : String) : String :=
  match Json.parse input with
  | .error e =>
    Json.compress <| .mkObj [("accepted", toJson false),
      ("reason", .str s!"JSON parse error: {e}")]
  | .ok j =>
    match progParseRequest j with
    | .error e =>
      Json.compress <| .mkObj [("accepted", toJson false), ("reason", .str e)]
    | .ok req =>
      match acceptProgram req with
      | .error e =>
        Json.compress <| .mkObj [("accepted", toJson false), ("reason", .str e)]
      | .ok cert =>
        Json.compress <| .mkObj [("accepted", toJson true),
          ("reason", .str "accepted"),
          ("program", .mkObj [("plan", progPlanToJson req.plan),
            ("commands", req.commandsJson), ("encoding", req.encodingJson)]),
          ("certificate", cert)]

def main (args : List String) : IO Unit := do
  let input ←
    match args with
    | ["--json", jsonStr] => pure jsonStr
    | _ => IO.getStdin >>= (·.readToEnd)
  let input := input.trimAscii.toString
  if input.isEmpty then
    IO.println <| Json.compress <| .mkObj [("accepted", toJson false),
      ("reason", .str "no input provided")]
  else
    IO.println (processProgram input)
