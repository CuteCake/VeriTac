/-
  VeriTac CLI
  Accepts JSON tactic sequences, applies them, emits JSON loop nest.
-/
import VeriTac.Basic
import VeriTac.Hardware.Target
import VeriTac.Attention.Plan
import VeriTac.Attention.CudaPlan
import Lean.Data.Json

open VeriTac
open VeriTac.Tactic
open VeriTac.Compose
open VeriTac.Hardware
open MetalAttentionLaunch
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

/-! ## Hardware attention-launch checking -/

/-- Parse an `AttentionMapping` from its JSON string name. -/
def parseAttentionMapping (s : String) : Except String AttentionMapping :=
  match s with
  | "scalar" => .ok .scalar
  | "simdgroup" => .ok .simdgroup
  | _ => .error s!"Unknown mapping: {s}"

/-- Parse a normalized hardware `Target` from snake_case JSON matching
    `Hardware/profile.py` (schema_version 1).  The target must declare
    `schema_version` exactly 1 and carry the backend, device, limits, and SIMD
    fields.  The descriptive `toolchain`/`provenance` objects are ignored and
    stored as empty strings, since the legality checker does not use them. -/
def parseTarget (j : Json) : Except String Target := do
  let sv ← j.getObjValAs? Nat "schema_version"
  if sv ≠ 1 then
    .error "expected target schema_version 1"
  else
    let backend ← j.getObjValAs? String "backend"
    let deviceName ← j.getObjValAs? String "device_name"
    let maxThreads ← j.getObjValAs? Nat "max_threads_per_threadgroup"
    let maxMem ← j.getObjValAs? Nat "max_threadgroup_memory_bytes"
    let simdWidth ← j.getObjValAs? Nat "simd_width"
    pure { backend, deviceName, maxThreadsPerThreadgroup := maxThreads,
           maxThreadgroupMemoryBytes := maxMem, simdWidth,
           toolchain := "", provenance := "" }

/-- Parse a Metal attention launch from snake_case JSON.  `mapping` is optional
    and defaults to `scalar`. -/
def parseLaunch (j : Json) : Except String MetalAttentionLaunch := do
  let mappingStr := (j.getObjValAs? String "mapping").toOption.getD "scalar"
  let mapping ← parseAttentionMapping mappingStr
  let headDim ← j.getObjValAs? Nat "head_dim"
  let queryTile ← j.getObjValAs? Nat "query_tile"
  let keyTile ← j.getObjValAs? Nat "key_tile"
  let threadsPerThreadgroup ← j.getObjValAs? Nat "threads_per_threadgroup"
  let dtypeBytes ← j.getObjValAs? Nat "dtype_bytes"
  let stageK ← j.getObjValAs? Bool "stage_k"
  let stageV ← j.getObjValAs? Bool "stage_v"
  let extraSharedBytes := (j.getObjValAs? Nat "extra_shared_bytes").toOption.getD 0
  pure { mapping, headDim, queryTile, keyTile, threadsPerThreadgroup, dtypeBytes,
         stageK, stageV, extraSharedBytes }

/-- Serialize a `MetalAttentionLaunch` to snake_case JSON, including the
    `mapping` field. -/
def launchToJson (l : MetalAttentionLaunch) : Json :=
  let mappingStr := match l.mapping with | .scalar => "scalar" | .simdgroup => "simdgroup"
  .mkObj [
    ("mapping", .str mappingStr),
    ("head_dim", toJson l.headDim),
    ("query_tile", toJson l.queryTile),
    ("key_tile", toJson l.keyTile),
    ("threads_per_threadgroup", toJson l.threadsPerThreadgroup),
    ("dtype_bytes", toJson l.dtypeBytes),
    ("stage_k", toJson l.stageK),
    ("stage_v", toJson l.stageV),
    ("extra_shared_bytes", toJson l.extraSharedBytes),
  ]

/-- A deterministic, human-readable diagnostic for a rejected launch.  The first
    violated condition wins; the ordering is fixed, so the output is stable.
    Common conditions are checked first, then the mapping-specific query-tile,
    SIMD-width, and thread-count clauses. -/
def checkLaunchDiagnostic (t : Target) (l : MetalAttentionLaunch) : String :=
  if checkLaunch t l then "accepted"
  else if t.backend != "metal" then "target backend is not metal"
  else if t.maxThreadsPerThreadgroup = 0 || t.maxThreadgroupMemoryBytes = 0 then "target limits are not positive"
  else if l.headDim != 192 && l.headDim != 256 then "unsupported head_dim"
  else if l.keyTile != 8 && l.keyTile != 16 then "unsupported key_tile"
  else if l.dtypeBytes != 4 then "unsupported dtype_bytes"
  else if !l.stageK || !l.stageV then "missing staging of K or V"
  else
    match l.mapping with
    | .scalar =>
        if l.queryTile != 8 && l.queryTile != 16 then "unsupported query_tile"
        else if l.threadsPerThreadgroup != l.queryTile then "threads must equal query_tile"
        else if l.threadsPerThreadgroup > t.maxThreadsPerThreadgroup then "threads exceed target limit"
        else if sharedBytes l > t.maxThreadgroupMemoryBytes then "shared memory exceeds target limit"
        else "rejected"
    | .simdgroup =>
        if l.queryTile != 4 && l.queryTile != 8 then "unsupported query_tile"
        else if t.simdWidth != 32 then "unsupported simd_width"
        else if l.threadsPerThreadgroup != l.queryTile * t.simdWidth then "threads must equal query_tile * simd_width"
        else if l.threadsPerThreadgroup > t.maxThreadsPerThreadgroup then "threads exceed target limit"
        else if sharedBytes l > t.maxThreadgroupMemoryBytes then "shared memory exceeds target limit"
        else "rejected"

/-- Handle a `check_attention_launch` request.  The request must declare
    `schema_version` exactly 1 and carry a `target` and `launch` object. -/
def processCheckAttentionLaunch (json : Json) : String :=
  match json.getObjValAs? Nat "schema_version" with
  | .error _ =>
    Json.compress <| .mkObj [("error", .str "Missing or invalid 'schema_version'")]
  | .ok 1 =>
    match json.getObjVal? "target", json.getObjVal? "launch" with
    | .error _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'target' field")]
    | _, .error _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'launch' field")]
    | .ok targetJson, .ok launchJson =>
      match parseTarget targetJson, parseLaunch launchJson with
      | .error e, _ =>
        Json.compress <| .mkObj [("error", .str s!"Target parse error: {e}")]
      | _, .error e =>
        Json.compress <| .mkObj [("error", .str s!"Launch parse error: {e}")]
      | .ok t, .ok l =>
        let accepted := checkLaunch t l
        Json.compress <| .mkObj [
          ("schema_version", toJson 1),
          ("accepted", toJson accepted),
          ("shared_memory_bytes", toJson (sharedBytes l)),
          ("diagnostic", .str (checkLaunchDiagnostic t l))
        ]
  | .ok _ =>
    Json.compress <| .mkObj [("error", .str "Unsupported 'schema_version'; expected 1")]

/-- Parse an `AttentionTactic` from its JSON object.  Each tactic carries
    exactly one kind plus its payload: `set_mapping` + `mapping`,
    `set_query_tile` + `value`, `set_key_tile` + `value`. -/
def parseAttentionTactic (j : Json) : Except String AttentionTactic := do
  let kind ← j.getObjValAs? String "kind"
  match kind with
  | "set_mapping" => do
      let mstr ← j.getObjValAs? String "mapping"
      let m ← parseAttentionMapping mstr
      pure (.setMapping m)
  | "set_query_tile" => do
      let q ← j.getObjValAs? Nat "value"
      pure (.setQueryTile q)
  | "set_key_tile" => do
      let k ← j.getObjValAs? Nat "value"
      pure (.setKeyTile k)
  | _ => .error s!"Unknown attention tactic kind: {kind}"

/-- The result of running a tactic sequence against a target: how many tactics
    were accepted, whether the whole sequence was accepted, the last accepted
    launch (the initial launch when nothing was accepted), and the first
    rejected proposed launch (when a tactic failed), if any. -/
structure TacticRunResult where
  applied : Nat
  accepted : Bool
  final : MetalAttentionLaunch
  rejected : Option MetalAttentionLaunch
  initialIllegal : Bool

/-- Run a tactic sequence against a target, driving the accepted path from the
    proved core `checkTactics` function.  On success the returned `final` is
    exactly the value `checkTactics t initial tactics` returns and `applied` is
    the full tactic count, so CLI acceptance flows through the function covered
    by `checkTactics_legal`.  Only on rejection does a diagnostic walk run, to
    report the applied prefix and the first illegal proposed launch. -/
def runCheckTactics (t : Target) (initial : MetalAttentionLaunch)
    (tactics : List AttentionTactic) : TacticRunResult :=
  match checkTactics t initial tactics with
  | some final =>
      { applied := tactics.length, accepted := true, final := final,
        rejected := none, initialIllegal := false }
  | none =>
      if !checkLaunch t initial then
        { applied := 0, accepted := false, final := initial, rejected := none,
          initialIllegal := true }
      else
        let rec go (acc : Nat) (current : MetalAttentionLaunch)
            (rest : List AttentionTactic) : TacticRunResult :=
          match rest with
          | [] =>
              { applied := acc, accepted := false, final := current, rejected := none,
                initialIllegal := false }
          | tac :: tl =>
              let proposed := applyTactic t current tac
              if checkLaunch t proposed then
                go (acc + 1) proposed tl
              else
                { applied := acc, accepted := false, final := current,
                  rejected := some proposed, initialIllegal := false }
        go 0 initial tactics

/-- Handle a `check_attention_tactics` request: run the attention tactic
    sequence against a target starting from an initial launch. -/
def processCheckAttentionTactics (json : Json) : String :=
  match json.getObjValAs? Nat "schema_version" with
  | .error _ =>
    Json.compress <| .mkObj [("error", .str "Missing or invalid 'schema_version'")]
  | .ok 1 =>
    match json.getObjVal? "target", json.getObjVal? "initial_launch", json.getObjValAs? (Array Json) "tactics" with
    | .error _, _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'target' field")]
    | _, .error _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'initial_launch' field")]
    | _, _, .error _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'tactics' field")]
    | .ok targetJson, .ok launchJson, .ok tacticsJson =>
      match parseTarget targetJson, parseLaunch launchJson, tacticsJson.toList.mapM parseAttentionTactic with
      | .error e, _, _ =>
        Json.compress <| .mkObj [("error", .str s!"Target parse error: {e}")]
      | _, .error e, _ =>
        Json.compress <| .mkObj [("error", .str s!"Initial launch parse error: {e}")]
      | _, _, .error e =>
        Json.compress <| .mkObj [("error", .str s!"Tactic parse error: {e}")]
      | .ok t, .ok l0, .ok tactics =>
        let res := runCheckTactics t l0 tactics
        let coreFinal := checkTactics t l0 tactics
        let consistent := match coreFinal with
          | some cf => res.accepted && Json.compress (launchToJson cf) == Json.compress (launchToJson res.final)
          | none => !res.accepted
        if !consistent then
          Json.compress <| .mkObj [("error", .str "internal checker mismatch")]
        else
          let diagnostic := match res.rejected with
            | some p => checkLaunchDiagnostic t p
            | none => if res.accepted then "accepted" else checkLaunchDiagnostic t res.final
          Json.compress <| .mkObj [
            ("schema_version", toJson 1),
            ("accepted", toJson res.accepted),
            ("applied", toJson res.applied),
            ("final_launch", launchToJson res.final),
            ("shared_memory_bytes", toJson (sharedBytes res.final)),
            ("diagnostic", .str diagnostic)
          ]
  | .ok _ =>
    Json.compress <| .mkObj [("error", .str "Unsupported 'schema_version'; expected 1")]

/-! ## Checked attention tiling plan -/

/-- Parse a `TiledAttentionPlan` from snake_case JSON.  `threads_per_threadgroup`
    is derived from the query tile (`(query_tile/8)*32`) and is not read from
    the input. -/
def parsePlan (j : Json) : Except String VeriTac.Attention.TiledAttentionPlan := do
  let seqLen ← j.getObjValAs? Nat "seq_len"
  let headDim ← j.getObjValAs? Nat "head_dim"
  let queryTile ← j.getObjValAs? Nat "query_tile"
  let keyTile ← j.getObjValAs? Nat "key_tile"
  let aliasKV ← j.getObjValAs? Bool "alias_kv"
  pure { seqLen, headDim, queryTile, keyTile, aliasKV,
         threadsPerThreadgroup := VeriTac.Attention.planThreads queryTile }

/-- Serialize a `TiledAttentionPlan` to snake_case JSON. -/
def planToJson (p : VeriTac.Attention.TiledAttentionPlan) : Json :=
  .mkObj [
    ("seq_len", toJson p.seqLen),
    ("head_dim", toJson p.headDim),
    ("query_tile", toJson p.queryTile),
    ("key_tile", toJson p.keyTile),
    ("alias_kv", toJson p.aliasKV),
    ("threads_per_threadgroup", toJson p.threadsPerThreadgroup),
  ]

/-- Parse a plan tactic from its JSON object: `set_query_tile`/`set_key_tile`
    carry a `value`; `reuse_kv_storage` takes none. -/
def parsePlanTactic (j : Json) : Except String VeriTac.Attention.PlanTactic := do
  let kind ← j.getObjValAs? String "kind"
  match kind with
  | "set_query_tile" => do
      let q ← j.getObjValAs? Nat "value"
      pure (.setQueryTile q)
  | "set_key_tile" => do
      let k ← j.getObjValAs? Nat "value"
      pure (.setKeyTile k)
  | "reuse_kv_storage" =>
      pure .reuseKVStorage
  | _ => .error s!"Unknown plan tactic kind: {kind}"

/-- The result of running a plan tactic sequence against a target. -/
structure PlanRunResult where
  applied : Nat
  accepted : Bool
  final : VeriTac.Attention.TiledAttentionPlan
  rejected : Option VeriTac.Attention.TiledAttentionPlan
  initialIllegal : Bool

/-- Run a plan tactic sequence against a target, driving the accepted path from
    the proved core `checkPlanTactics` function. -/
def runCheckPlanTactics (t : Target) (initial : VeriTac.Attention.TiledAttentionPlan)
    (tactics : List VeriTac.Attention.PlanTactic) : PlanRunResult :=
  match VeriTac.Attention.checkPlanTactics t initial tactics with
  | some final =>
      { applied := tactics.length, accepted := true, final := final,
        rejected := none, initialIllegal := false }
  | none =>
      if !VeriTac.Attention.checkPlan t initial then
        { applied := 0, accepted := false, final := initial, rejected := none,
          initialIllegal := true }
      else
        let rec go (acc : Nat) (current : VeriTac.Attention.TiledAttentionPlan)
            (rest : List VeriTac.Attention.PlanTactic) : PlanRunResult :=
          match rest with
          | [] =>
              { applied := acc, accepted := false, final := current, rejected := none,
                initialIllegal := false }
          | tac :: tl =>
              let proposed := VeriTac.Attention.applyPlanTactic current tac
              if VeriTac.Attention.checkPlan t proposed then
                go (acc + 1) proposed tl
              else
                { applied := acc, accepted := false, final := current,
                  rejected := some proposed, initialIllegal := false }
        go 0 initial tactics

/-- A deterministic, human-readable diagnostic for a rejected plan. -/
def checkPlanDiagnostic (t : Target) (p : VeriTac.Attention.TiledAttentionPlan) : String :=
  if VeriTac.Attention.checkPlan t p then "accepted"
  else if t.backend != "metal" then "target backend is not metal"
  else if t.maxThreadsPerThreadgroup = 0 || t.maxThreadgroupMemoryBytes = 0 then "target limits are not positive"
  else if p.seqLen = 0 then "seq_len must be positive"
  else if p.headDim != 192 && p.headDim != 256 then "unsupported head_dim"
  else if p.queryTile != 8 && p.queryTile != 16 && p.queryTile != 24 && p.queryTile != 32 then "unsupported query_tile"
  else if p.keyTile != 8 && p.keyTile != 16 && p.keyTile != 32 then "unsupported key_tile"
  else if t.simdWidth != 32 then "unsupported simd_width"
  else if p.threadsPerThreadgroup != VeriTac.Attention.planThreads p.queryTile then "threads must equal (query_tile/8)*32"
  else if p.threadsPerThreadgroup > t.maxThreadsPerThreadgroup then "threads exceed target limit"
  else if VeriTac.Attention.sharedBytes p > t.maxThreadgroupMemoryBytes then "shared memory exceeds target limit"
  else if !(VeriTac.Attention.plansLegalPartitions p) then "row prefix partitions are not legal"
  else "rejected"

/-- Handle a `check_attention_plan` request: run a plan tactic sequence against a
    target starting from an initial plan.  Empty tactics simply check the
    initial plan. -/
def processCheckAttentionPlan (json : Json) : String :=
  match json.getObjValAs? Nat "schema_version" with
  | .error _ =>
    Json.compress <| .mkObj [("error", .str "Missing or invalid 'schema_version'")]
  | .ok 1 =>
    match json.getObjVal? "target", json.getObjVal? "initial_plan", json.getObjValAs? (Array Json) "tactics" with
    | .error _, _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'target' field")]
    | _, .error _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'initial_plan' field")]
    | _, _, .error _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'tactics' field")]
    | .ok targetJson, .ok planJson, .ok tacticsJson =>
      match parseTarget targetJson, parsePlan planJson, tacticsJson.toList.mapM parsePlanTactic with
      | .error e, _, _ =>
        Json.compress <| .mkObj [("error", .str s!"Target parse error: {e}")]
      | _, .error e, _ =>
        Json.compress <| .mkObj [("error", .str s!"Initial plan parse error: {e}")]
      | _, _, .error e =>
        Json.compress <| .mkObj [("error", .str s!"Tactic parse error: {e}")]
      | .ok t, .ok p0, .ok tactics =>
        let res := runCheckPlanTactics t p0 tactics
        let coreFinal := VeriTac.Attention.checkPlanTactics t p0 tactics
        let consistent := match coreFinal with
          | some cf => res.accepted && Json.compress (planToJson cf) == Json.compress (planToJson res.final)
          | none => !res.accepted
        if !consistent then
          Json.compress <| .mkObj [("error", .str "internal checker mismatch")]
        else
          let diagnostic := match res.rejected with
            | some p => checkPlanDiagnostic t p
            | none => if res.accepted then "accepted" else checkPlanDiagnostic t res.final
          Json.compress <| .mkObj [
            ("schema_version", toJson 1),
            ("accepted", toJson res.accepted),
            ("applied", toJson res.applied),
            ("final_plan", planToJson res.final),
            ("shared_memory_bytes", toJson (VeriTac.Attention.sharedBytes res.final)),
            ("threads_per_threadgroup", toJson res.final.threadsPerThreadgroup),
            ("executable", toJson res.final.aliasKV),
            ("diagnostic", .str diagnostic)
          ]
  | .ok _ =>
    Json.compress <| .mkObj [("error", .str "Unsupported 'schema_version'; expected 1")]

/-! ## Checked CUDA config-selection plan -/

/-- Parse a raw C++ `config_info` object.  Extra metadata fields (e.g.
    `min_blocks_hint`, `num_regs`) are ignored and permitted. -/
def parseCudaConfig (j : Json) : Except String VeriTac.Attention.CudaConfig := do
  let id ← j.getObjValAs? Nat "id"
  let q ← j.getObjValAs? Nat "queries_per_block"
  let k ← j.getObjValAs? Nat "keys_per_block"
  let maxK ← j.getObjValAs? Nat "max_k"
  let numThreads ← j.getObjValAs? Nat "num_threads"
  let dyn ← j.getObjValAs? Nat "smem_bytes"
  let stat ← j.getObjValAs? Nat "static_shared_bytes"
  let kernelMax ← j.getObjValAs? Nat "kernel_max_threads"
  let supported ← j.getObjValAs? Bool "supported"
  pure { id, queriesPerBlock := q, keysPerBlock := k, maxK, numThreads,
         dynamicSmemBytes := dyn, staticSharedBytes := stat,
         kernelMaxThreads := kernelMax, supported }

/-- Serialize a `CudaConfig` to snake_case JSON. -/
def cudaConfigToJson (c : VeriTac.Attention.CudaConfig) : Json :=
  .mkObj [
    ("id", toJson c.id),
    ("queries_per_block", toJson c.queriesPerBlock),
    ("keys_per_block", toJson c.keysPerBlock),
    ("max_k", toJson c.maxK),
    ("num_threads", toJson c.numThreads),
    ("smem_bytes", toJson c.dynamicSmemBytes),
    ("static_shared_bytes", toJson c.staticSharedBytes),
    ("kernel_max_threads", toJson c.kernelMaxThreads),
    ("supported", toJson c.supported),
  ]

/-- Parse an `initial_plan` for the CUDA mode. -/
def parseCudaPlan (j : Json) : Except String VeriTac.Attention.CudaPlan := do
  let seqLen ← j.getObjValAs? Nat "seq_len"
  let headDim ← j.getObjValAs? Nat "head_dim"
  let configId ← j.getObjValAs? Nat "config_id"
  pure { seqLen, headDim, configId }

/-- Serialize a `CudaPlan` (plus its selected config geometry/resources) to JSON. -/
def cudaPlanToJson (catalog : List VeriTac.Attention.CudaConfig)
    (p : VeriTac.Attention.CudaPlan) : Json :=
  let cfg := VeriTac.Attention.selectedConfig catalog p.configId
  .mkObj [
    ("seq_len", toJson p.seqLen),
    ("head_dim", toJson p.headDim),
    ("config_id", toJson p.configId),
    ("queries_per_block", toJson cfg.queriesPerBlock),
    ("keys_per_block", toJson cfg.keysPerBlock),
    ("max_k", toJson cfg.maxK),
    ("num_threads", toJson cfg.numThreads),
    ("smem_bytes", toJson cfg.dynamicSmemBytes),
    ("static_shared_bytes", toJson cfg.staticSharedBytes),
    ("kernel_max_threads", toJson cfg.kernelMaxThreads),
    ("supported", toJson cfg.supported),
  ]

/-- Parse a CUDA plan tactic: `select_config` + `value`. -/
def parseCudaTactic (j : Json) : Except String VeriTac.Attention.CudaTactic := do
  let kind ← j.getObjValAs? String "kind"
  match kind with
  | "select_config" => do
      let v ← j.getObjValAs? Nat "value"
      pure (.selectConfig v)
  | _ => .error s!"Unknown cuda tactic kind: {kind}"

/-- The result of running a CUDA tactic sequence against a target/catalogue. -/
structure CudaPlanRunResult where
  applied : Nat
  accepted : Bool
  final : VeriTac.Attention.CudaPlan
  rejected : Option VeriTac.Attention.CudaPlan
  initialIllegal : Bool

/-- Run a CUDA tactic sequence, driving the accepted path from the proved core
    `checkCudaTactics`.  Only on rejection does a diagnostic walk run to report
    the applied prefix and the first illegal proposed plan. -/
def runCheckCudaTactics (dev : Target) (catalog : List VeriTac.Attention.CudaConfig)
    (initial : VeriTac.Attention.CudaPlan) (tactics : List VeriTac.Attention.CudaTactic) : CudaPlanRunResult :=
  match VeriTac.Attention.checkCudaTactics dev catalog initial tactics with
  | some final =>
      { applied := tactics.length, accepted := true, final := final,
        rejected := none, initialIllegal := false }
  | none =>
      if !VeriTac.Attention.checkCudaPlan dev catalog initial then
        { applied := 0, accepted := false, final := initial, rejected := none,
          initialIllegal := true }
      else
        let rec go (acc : Nat) (current : VeriTac.Attention.CudaPlan)
            (rest : List VeriTac.Attention.CudaTactic) : CudaPlanRunResult :=
          match rest with
          | [] =>
              { applied := acc, accepted := false, final := current, rejected := none,
                initialIllegal := false }
          | tac :: tl =>
              let proposed := VeriTac.Attention.applyCudaTactic current tac
              if VeriTac.Attention.checkCudaPlan dev catalog proposed then
                go (acc + 1) proposed tl
              else
                { applied := acc, accepted := false, final := current,
                  rejected := some proposed, initialIllegal := false }
        go 0 initial tactics

/-- A deterministic, human-readable diagnostic for a rejected CUDA plan. -/
def checkCudaPlanDiagnostic (dev : Target) (catalog : List VeriTac.Attention.CudaConfig)
    (p : VeriTac.Attention.CudaPlan) : String :=
  if VeriTac.Attention.checkCudaPlan dev catalog p then "accepted"
  else if !(VeriTac.Attention.NoDuplicateIds catalog) then "catalogue has duplicate config ids"
  else if dev.backend != "cuda" then "target backend is not cuda"
  else if dev.simdWidth != 32 then "unsupported simd_width (warp must be 32)"
  else if dev.maxThreadsPerThreadgroup = 0 || dev.maxThreadgroupMemoryBytes = 0 then "target limits are not positive"
  else if p.seqLen = 0 then "seq_len must be positive"
  else if p.headDim != 192 && p.headDim != 256 then "unsupported head_dim"
  else if VeriTac.Attention.lookupCudaConfig catalog p.configId = none then "unknown config id"
  else
    let cfg := VeriTac.Attention.selectedConfig catalog p.configId
    if !cfg.supported then "config is unsupported"
    else if p.headDim > cfg.maxK then "head_dim exceeds config max_k"
    else if cfg.queriesPerBlock = 0 || cfg.keysPerBlock = 0 then "config block geometry is not positive"
    else if cfg.queriesPerBlock % 32 != 0 || cfg.keysPerBlock % 32 != 0 then "block geometry must be a multiple of 32"
    else if cfg.numThreads != VeriTac.Attention.CudaConfig.derivedThreads cfg then "threads must equal queries_per_block*keys_per_block/32"
    else if cfg.numThreads > cfg.kernelMaxThreads then "threads exceed kernel max"
    else if cfg.numThreads > dev.maxThreadsPerThreadgroup then "threads exceed device limit"
    else if VeriTac.Attention.CudaConfig.totalSmem cfg > dev.maxThreadgroupMemoryBytes then "shared memory exceeds device limit"
    else if !(VeriTac.Attention.cudaPlansLegalPartitions p.seqLen cfg.keysPerBlock) then "row prefix partitions are not legal"
    else "rejected"

/-- Handle a `check_cuda_attention_plan` request: run a config-selection tactic
    sequence against a target/catalogue starting from an initial plan.  Empty
    tactics simply check the initial plan. -/
def processCheckCudaAttentionPlan (json : Json) : String :=
  match json.getObjValAs? Nat "schema_version" with
  | .error _ =>
    Json.compress <| .mkObj [("error", .str "Missing or invalid 'schema_version'")]
  | .ok 1 =>
    match json.getObjVal? "target", json.getObjValAs? (Array Json) "catalog",
          json.getObjVal? "initial_plan", json.getObjValAs? (Array Json) "tactics" with
    | .error _, _, _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'target' field")]
    | _, .error _, _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing or invalid 'catalog' field")]
    | _, _, .error _, _ =>
      Json.compress <| .mkObj [("error", .str "Missing 'initial_plan' field")]
    | _, _, _, .error _ =>
      Json.compress <| .mkObj [("error", .str "Missing or invalid 'tactics' field")]
    | .ok targetJson, .ok catalogArr, .ok planJson, .ok tacticsArr =>
      match parseTarget targetJson, catalogArr.toList.mapM parseCudaConfig,
            parseCudaPlan planJson, tacticsArr.toList.mapM parseCudaTactic with
      | .error e, _, _, _ =>
        Json.compress <| .mkObj [("error", .str s!"Target parse error: {e}")]
      | _, .error e, _, _ =>
        Json.compress <| .mkObj [("error", .str s!"Catalog parse error: {e}")]
      | _, _, .error e, _ =>
        Json.compress <| .mkObj [("error", .str s!"Initial plan parse error: {e}")]
      | _, _, _, .error e =>
        Json.compress <| .mkObj [("error", .str s!"Tactic parse error: {e}")]
      | .ok dev, .ok catalog, .ok p0, .ok tactics =>
        let res := runCheckCudaTactics dev catalog p0 tactics
        let coreFinal := VeriTac.Attention.checkCudaTactics dev catalog p0 tactics
        let consistent := match coreFinal with
          | some cf => res.accepted && Json.compress (cudaPlanToJson catalog cf) ==
                                           Json.compress (cudaPlanToJson catalog res.final)
          | none => !res.accepted
        if !consistent then
          Json.compress <| .mkObj [("error", .str "internal checker mismatch")]
        else
          let diagnostic := match res.rejected with
            | some p => checkCudaPlanDiagnostic dev catalog p
            | none => if res.accepted then "accepted" else checkCudaPlanDiagnostic dev catalog res.final
          let cfg := VeriTac.Attention.selectedConfig catalog res.final.configId
          Json.compress <| .mkObj [
            ("schema_version", toJson 1),
            ("accepted", toJson res.accepted),
            ("applied", toJson res.applied),
            ("final_plan", cudaPlanToJson catalog res.final),
            ("shared_memory_bytes", toJson (VeriTac.Attention.CudaConfig.totalSmem cfg)),
            ("threads_per_threadgroup", toJson cfg.numThreads),
            ("diagnostic", .str diagnostic)
          ]
  | .ok _ =>
    Json.compress <| .mkObj [("error", .str "Unsupported 'schema_version'; expected 1")]

/-! ## Main CLI -/

def processLegacy (json : Json) : String :=
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

def processInput (input : String) : String :=
  match Json.parse input with
  | .error e => Json.compress <| .mkObj [("error", .str s!"JSON parse error: {e}")]
  | .ok json =>
    match json.getObjValAs? String "mode" with
    | .ok "check_attention_launch" => processCheckAttentionLaunch json
    | .ok "check_attention_tactics" => processCheckAttentionTactics json
    | .ok "check_attention_plan" => processCheckAttentionPlan json
    | .ok "check_cuda_attention_plan" => processCheckCudaAttentionPlan json
    | .ok other => Json.compress <| .mkObj [("error", .str s!"Unknown mode: {other}")]
    | .error _ => processLegacy json

def main (args : List String) : IO Unit := do
  match args with
  | ["--json", jsonStr] =>
    IO.println (processInput jsonStr)
  | _ =>
    let input ← IO.getStdin >>= (·.readToEnd)
    let input := input.trimAscii.toString
    if input.isEmpty then
      IO.println "{\"error\": \"No input provided. Use --json '<json>' or pipe to stdin.\"}"
    else
      IO.println (processInput input)
