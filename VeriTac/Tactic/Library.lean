/-
  VeriTac.Tactic.Library
  Tactic registry: enumerates all available tactics with their metadata.
-/
import VeriTac.Tactic.Tile
import VeriTac.Tactic.Split
import VeriTac.Tactic.Fuse
import VeriTac.Tactic.Reorder
import VeriTac.Tactic.Unroll
import VeriTac.Tactic.Vectorize
import VeriTac.Tactic.Parallel
import VeriTac.Tactic.CacheRead

namespace VeriTac.Tactic

/-- Enumeration of all available tactics. -/
inductive TacticKind where
  | tile
  | split
  | fuse
  | reorder
  | unroll
  | vectorize
  | parallel
  | cacheRead
  deriving Repr, BEq, Inhabited

/-- A tactic application: a tactic kind with its parameters. -/
structure TacticApp where
  kind : TacticKind
  /-- Target loop variable(s) -/
  vars : List VarId
  /-- Integer parameters (e.g., tile size, factor) -/
  intParams : List Nat
  /-- String parameters (e.g., buffer names) -/
  strParams : List String
  deriving Repr, BEq, Inhabited

/-- Does the statement contain a loop named `v`? (whether nested or top-level) -/
def stmtHasLoopVar (v : VarId) : Stmt → Bool
  | .loop lv _ _ _ body => lv == v || stmtHasLoopVar v body
  | .seq s1 s2 => stmtHasLoopVar v s1 || stmtHasLoopVar v s2
  | .alloc _ _ body => stmtHasLoopVar v body
  | _ => false

private def reorderDiag (v1 v2 : VarId) (s : Stmt) : String :=
  if !stmtHasLoopVar v1 s then s!"reorder: loop '{v1}' not found in the statement"
  else if !stmtHasLoopVar v2 s then s!"reorder: loop '{v2}' not found in the statement"
  else s!"reorder: '{v1}' and '{v2}' are not adjacent nested loops, or the swap is not safe (bounds depend on the other axis)"

private def vectorizeDiag (v : VarId) (s : Stmt) : String :=
  if !stmtHasLoopVar v s then s!"vectorize: loop '{v}' not found in the statement"
  else s!"vectorize: loop '{v}' is not an innermost loop (annotating it with SIMD would be unsound)"

private def tileDiag (v : VarId) (ts : Nat) (s : Stmt) : String :=
  if !stmtHasLoopVar v s then s!"tile: loop '{v}' not found in the statement"
  else if ts == 0 then "tile: tile size must be > 0"
  else s!"tile: loop '{v}' is not a literal [0, hi) loop with hi divisible by {ts}"

private def parallelDiag (v : VarId) (s : Stmt) : String :=
  if !stmtHasLoopVar v s then s!"parallel: loop '{v}' not found in the statement"
  else s!"parallel: loop '{v}' has no writes indexed by it, or is not parallel-safe (see loopLocalWrites)"

private def unrollDiag (v : VarId) (s : Stmt) : String :=
  if !stmtHasLoopVar v s then s!"unroll: loop '{v}' not found in the statement"
  else "unroll: loop bounds must be literal constants"

/-- Apply a single tactic application to a statement, returning a diagnostic
    message on failure instead of a bare `none`. -/
def applyTactic (app : TacticApp) (stmt : Stmt) : Except String Stmt :=
  match app.kind with
  | .tile =>
    match app.vars, app.intParams with
    | [v], [ts] => match tile stmt v ts with
        | some s => .ok s
        | none => .error (tileDiag v ts stmt)
    | _, _ => .error "tile requires [var] and one integer tile size"
  | .split =>
    match app.vars, app.intParams with
    | [v], [f] => match split stmt v f with
        | some s => .ok s
        | none => .error (tileDiag v f stmt)
    | _, _ => .error "split requires [var] and one integer factor"
  | .fuse =>
    match app.vars with
    | [v1, v2] => match fuse stmt v1 v2 with
        | some s => .ok s
        | none => .error s!"fuse: '{v1}' and '{v2}' are not adjacent nested loops"
    | _ => .error "fuse requires two variable names"
  | .reorder =>
    match app.vars with
    | [v1, v2] => match reorder stmt v1 v2 with
        | some s => .ok s
        | none => .error (reorderDiag v1 v2 stmt)
    | _ => .error "reorder requires two variable names"
  | .unroll =>
    match app.vars with
    | [v] => match unroll stmt v with
        | some s => .ok s
        | none => .error (unrollDiag v stmt)
    | _ => .error "unroll requires one variable name"
  | .vectorize =>
    match app.vars with
    | [v] => match vectorize stmt v with
        | some s => .ok s
        | none => .error (vectorizeDiag v stmt)
    | _ => .error "vectorize requires one variable name"
  | .parallel =>
    match app.vars with
    | [v] => match parallel stmt v with
        | some s => .ok s
        | none => .error (parallelDiag v stmt)
    | _ => .error "parallel requires one variable name"
  | .cacheRead =>
    match app.strParams with
    | [orig, cache] => match cacheRead stmt orig cache [] [] with
        | some s => .ok s
        | none => .error "cache_read: not applicable"
    | _ => .error "cache_read requires original and cache buffer names"

end VeriTac.Tactic
