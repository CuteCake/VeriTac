/-
  GemminiMain: JSON CLI for Gemmini GEMM plan checking.

  Reads one JSON object on stdin (or via `--json '<json>'`), validates a GEMM
  plan, and emits exactly one JSON object:

    {"accepted": bool, "reason": string, "plan": <validated plan>}

  where `"plan"` is present only on acceptance.  Every failure mode — malformed
  JSON, missing fields, noninteger values, negative values, unknown schedules,
  illegal plans — exits normally with `accepted: false` (fail closed).
-/
import Lean.Data.Json
import VeriTac.Gemmini.Plan

open Lean (Json JsonNumber ToJson FromJson)
open Lean.ToJson
open VeriTac.Gemmini

/-! ## Strict integer field parsing -/

/-- A JSON number `mantissa * 10^-exponent` is a strict integer exactly when
    the mantissa is divisible by `10^exponent`.  Exponents above 30 are
    rejected fail-closed: they cannot arise from realistic plan dimensions and
    would make the scale computation needlessly large. -/
def jsonStrictInt? (n : JsonNumber) : Option Int :=
  if n.exponent > 30 then none
  else
    let scale : Int := (10 : Int) ^ n.exponent
    if n.mantissa % scale == 0 then some (n.mantissa / scale) else none

/-- Read field `name` of a JSON object as a strict integer.  Missing fields,
    non-numbers, and noninteger numbers all fail. -/
def intField? (j : Json) (name : String) : Option Int := do
  let v ← j.getObjVal? name |>.toOption
  match v with
  | .num n => jsonStrictInt? n
  | _ => none

/-- Reject negative values; the plan structure carries `Nat`s. -/
def requireNonneg (name : String) (v : Int) : Except String Nat :=
  if v < 0 then .error s!"field '{name}' is negative"
  else .ok v.toNat

/-! ## Plan parsing -/

/-- Parse the schedule string; unknown schedules fail closed. -/
def parseSchedule (s : String) : Except String GemminiSchedule :=
  match s with
  | "baseline" => .ok .baseline
  | "reuse_b" => .ok .reuseB
  | _ => .error s!"unknown schedule: {s}"

/-- Parse a full Gemmini plan.  Every integer field must be present, a strict
    JSON integer, and nonnegative; `m n k dim` must additionally be positive
    (checked by `checkGemminiPlan`). -/
def parseGemminiPlan (j : Json) : Except String GemminiPlan := do
  let mI ← match intField? j "m" with
    | some v => .ok v | none => .error "missing or noninteger field 'm'"
  let nI ← match intField? j "n" with
    | some v => .ok v | none => .error "missing or noninteger field 'n'"
  let kI ← match intField? j "k" with
    | some v => .ok v | none => .error "missing or noninteger field 'k'"
  let dimI ← match intField? j "dim" with
    | some v => .ok v | none => .error "missing or noninteger field 'dim'"
  let spI ← match intField? j "scratchpad_rows" with
    | some v => .ok v | none => .error "missing or noninteger field 'scratchpad_rows'"
  let accI ← match intField? j "accumulator_rows" with
    | some v => .ok v | none => .error "missing or noninteger field 'accumulator_rows'"
  let schedStr ← match j.getObjValAs? String "schedule" with
    | .ok s => .ok s | .error _ => .error "missing or non-string field 'schedule'"
  let schedule ← parseSchedule schedStr
  let m ← requireNonneg "m" mI
  let n ← requireNonneg "n" nI
  let k ← requireNonneg "k" kI
  let dim ← requireNonneg "dim" dimI
  let scratchpadRows ← requireNonneg "scratchpad_rows" spI
  let accumulatorRows ← requireNonneg "accumulator_rows" accI
  pure { m, n, k, dim, scratchpadRows, accumulatorRows, schedule }

/-! ## Serialization -/

/-- Serialize the schedule back to its JSON name. -/
def scheduleToJson : GemminiSchedule → Json
  | .baseline => .str "baseline"
  | .reuseB => .str "reuse_b"

/-- Serialize a validated plan: the original fields, canonically. -/
def gemminiPlanToJson (p : GemminiPlan) : Json :=
  .mkObj [
    ("m", toJson p.m),
    ("n", toJson p.n),
    ("k", toJson p.k),
    ("dim", toJson p.dim),
    ("scratchpad_rows", toJson p.scratchpadRows),
    ("accumulator_rows", toJson p.accumulatorRows),
    ("schedule", scheduleToJson p.schedule)
  ]

/-! ## Request processing -/

/-- Handle one request.  Rejections (parse errors and illegal plans alike)
    produce `accepted: false` with a reason and no `plan` field; acceptances
    echo the validated plan. -/
def processGemmini (input : String) : String :=
  match Json.parse input with
  | .error e =>
    Json.compress <| .mkObj [("accepted", toJson false),
      ("reason", .str s!"JSON parse error: {e}")]
  | .ok j =>
    match parseGemminiPlan j with
    | .error e =>
      Json.compress <| .mkObj [("accepted", toJson false), ("reason", .str e)]
    | .ok p =>
      if checkGemminiPlan p then
        Json.compress <| .mkObj [("accepted", toJson true),
          ("reason", .str "accepted"), ("plan", gemminiPlanToJson p)]
      else
        Json.compress <| .mkObj [("accepted", toJson false),
          ("reason", .str (planDiagnostic p))]

/-! ## Entry point -/

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
    IO.println (processGemmini input)
