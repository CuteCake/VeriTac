/-
  VeriTac CLI
  Accepts JSON tactic sequences, applies them, emits JSON loop nest.
-/
import VeriTac.Basic
import Lean.Data.Json

open VeriTac
open VeriTac.Tactic
open VeriTac.Compose
open Lean (Json ToJson toJson FromJson)

/-! ## JSON Serialization -/

/-- Serialize an SExpr to JSON. -/
partial def sexprToJson : SExpr → Json
  | .lit n => .mkObj [("tag", "lit"), ("val", toJson n)]
  | .var v => .mkObj [("tag", "var"), ("name", .str v)]
  | .add a b => .mkObj [("tag", "add"), ("left", sexprToJson a), ("right", sexprToJson b)]
  | .mul a b => .mkObj [("tag", "mul"), ("left", sexprToJson a), ("right", sexprToJson b)]
  | .div a b => .mkObj [("tag", "div"), ("left", sexprToJson a), ("right", sexprToJson b)]
  | .mod a b => .mkObj [("tag", "mod"), ("left", sexprToJson a), ("right", sexprToJson b)]
  | .bufRead buf indices =>
    .mkObj [("tag", "bufRead"), ("buf", .str buf),
            ("indices", .arr (indices.map sexprToJson).toArray)]

/-- Serialize an Annotation to JSON. -/
def annToJson : Annotation → Json
  | .none => .str "none"
  | .parallel => .str "parallel"
  | .vectorize => .str "vectorize"
  | .unrolled => .str "unrolled"

/-- Serialize a Stmt to JSON. -/
partial def stmtToJson : Stmt → Json
  | .skip => .mkObj [("tag", "skip")]
  | .bufWrite buf indices val =>
    .mkObj [("tag", "bufWrite"), ("buf", .str buf),
            ("indices", .arr (indices.map sexprToJson).toArray),
            ("val", sexprToJson val)]
  | .loop var lo hi ann body =>
    .mkObj [("tag", "loop"), ("var", .str var),
            ("lo", sexprToJson lo), ("hi", sexprToJson hi),
            ("ann", annToJson ann), ("body", stmtToJson body)]
  | .seq s1 s2 =>
    .mkObj [("tag", "seq"), ("s1", stmtToJson s1), ("s2", stmtToJson s2)]
  | .alloc buf shape body =>
    .mkObj [("tag", "alloc"), ("buf", .str buf),
            ("shape", .arr (shape.map sexprToJson).toArray),
            ("body", stmtToJson body)]

/-! ## JSON Deserialization -/

/-- Parse a TacticKind from a string. -/
def parseTacticKind (s : String) : Except String TacticKind :=
  match s with
  | "tile" => .ok .tile
  | "split" => .ok .split
  | "fuse" => .ok .fuse
  | "reorder" => .ok .reorder
  | "unroll" => .ok .unroll
  | "vectorize" => .ok .vectorize
  | "parallel" => .ok .parallel
  | "cache_read" => .ok .cacheRead
  | _ => .error s!"Unknown tactic: {s}"

/-- Parse a TacticApp from JSON. -/
def parseTacticApp (j : Json) : Except String TacticApp := do
  let kindStr ← j.getObjValAs? String "kind"
  let kind ← parseTacticKind kindStr
  let vars ← j.getObjValAs? (List String) "vars"
  let intParams := (j.getObjValAs? (List Nat) "int_params").toOption.getD []
  let strParams := (j.getObjValAs? (List String) "str_params").toOption.getD []
  pure { kind, vars, intParams, strParams }

/-- Parse an SExpr from JSON. -/
partial def parseSExpr (j : Json) : Except String SExpr := do
  let tag ← j.getObjValAs? String "tag"
  match tag with
  | "lit" => do
    let n ← j.getObjValAs? Int "val"
    pure (.lit n)
  | "var" => do
    let name ← j.getObjValAs? String "name"
    pure (.var name)
  | "add" => do
    let left ← parseSExpr (← j.getObjVal? "left")
    let right ← parseSExpr (← j.getObjVal? "right")
    pure (.add left right)
  | "mul" => do
    let left ← parseSExpr (← j.getObjVal? "left")
    let right ← parseSExpr (← j.getObjVal? "right")
    pure (.mul left right)
  | "div" => do
    let left ← parseSExpr (← j.getObjVal? "left")
    let right ← parseSExpr (← j.getObjVal? "right")
    pure (.div left right)
  | "mod" => do
    let left ← parseSExpr (← j.getObjVal? "left")
    let right ← parseSExpr (← j.getObjVal? "right")
    pure (.mod left right)
  | "bufRead" => do
    let buf ← j.getObjValAs? String "buf"
    let indicesJson ← j.getObjValAs? (Array Json) "indices"
    let indices ← indicesJson.toList.mapM parseSExpr
    pure (.bufRead buf indices)
  | _ => .error s!"Unknown expression tag: {tag}"

/-- Parse a Stmt from JSON. -/
partial def parseStmt (j : Json) : Except String Stmt := do
  let tag ← j.getObjValAs? String "tag"
  match tag with
  | "skip" => pure .skip
  | "bufWrite" => do
    let buf ← j.getObjValAs? String "buf"
    let indicesJson ← j.getObjValAs? (Array Json) "indices"
    let indices ← indicesJson.toList.mapM parseSExpr
    let valJson ← j.getObjVal? "val"
    let val ← parseSExpr valJson
    pure (.bufWrite buf indices val)
  | "loop" => do
    let var ← j.getObjValAs? String "var"
    let lo ← parseSExpr (← j.getObjVal? "lo")
    let hi ← parseSExpr (← j.getObjVal? "hi")
    let annStr := (j.getObjValAs? String "ann").toOption.getD "none"
    let ann := match annStr with
      | "parallel" => Annotation.parallel
      | "vectorize" => Annotation.vectorize
      | "unrolled" => Annotation.unrolled
      | _ => Annotation.none
    let body ← parseStmt (← j.getObjVal? "body")
    pure (.loop var lo hi ann body)
  | "seq" => do
    let s1 ← parseStmt (← j.getObjVal? "s1")
    let s2 ← parseStmt (← j.getObjVal? "s2")
    pure (.seq s1 s2)
  | "alloc" => do
    let buf ← j.getObjValAs? String "buf"
    let shapeJson ← j.getObjValAs? (Array Json) "shape"
    let shape ← shapeJson.toList.mapM parseSExpr
    let body ← parseStmt (← j.getObjVal? "body")
    pure (.alloc buf shape body)
  | _ => .error s!"Unknown statement tag: {tag}"

/-! ## Main CLI -/

def processInput (input : String) : String :=
  match Json.parse input with
  | .error e => Json.compress <| .mkObj [("error", .str s!"JSON parse error: {e}")]
  | .ok json =>
    match json.getObjVal? "stmt" with
    | .error _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'stmt' field")]
    | .ok stmtJson =>
      match parseStmt stmtJson with
      | .error e =>
        Json.compress <| .mkObj [("error", .str s!"Stmt parse error: {e}")]
      | .ok stmt =>
        match json.getObjValAs? (Array Json) "tactics" with
        | .error _ =>
          Json.compress <| .mkObj [("stmt", stmtToJson stmt)]
        | .ok tacticsJson =>
          match tacticsJson.toList.mapM parseTacticApp with
          | .error e =>
            Json.compress <| .mkObj [("error", .str s!"Tactic parse error: {e}")]
          | .ok tactics =>
            match applySchedule tactics stmt with
            | .error e =>
              Json.compress <| .mkObj [("error", .str e)]
            | .ok result =>
              Json.compress <| .mkObj [
                ("stmt", stmtToJson result.stmt),
                ("applied", toJson result.appliedCount)
              ]

def main (args : List String) : IO Unit := do
  match args with
  | ["--json", jsonStr] =>
    IO.println (processInput jsonStr)
  | _ =>
    let input ← IO.getStdin >>= (·.readToEnd)
    if input.trim.isEmpty then
      IO.println "{\"error\": \"No input provided. Use --json '<json>' or pipe to stdin.\"}"
    else
      IO.println (processInput input.trim)
