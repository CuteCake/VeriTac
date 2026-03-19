/-
  VeriTac.Compose.Precondition
  Precondition framework for tactic application.
-/
import VeriTac.Tactic.Library

namespace VeriTac.Compose

/-- Result of checking a tactic's preconditions. -/
inductive PrecondResult where
  | ok
  | err (msg : String)
  deriving Repr, BEq, Inhabited

/-- Check preconditions for a tactic application. -/
def checkPrecondition (app : VeriTac.Tactic.TacticApp) (stmt : VeriTac.Stmt) : PrecondResult :=
  match app.kind with
  | .tile =>
    match app.intParams with
    | [ts] => if ts > 0 then .ok else .err "tile size must be > 0"
    | _ => .err "tile requires exactly one integer parameter (tile size)"
  | .split =>
    match app.intParams with
    | [f] => if f > 0 then .ok else .err "split factor must be > 0"
    | _ => .err "split requires exactly one integer parameter (factor)"
  | .fuse =>
    match app.vars with
    | [_, _] => .ok  -- detailed check done in fuse itself
    | _ => .err "fuse requires exactly two variable names"
  | .reorder =>
    match app.vars with
    | [_, _] => .ok  -- independence checked in reorder
    | _ => .err "reorder requires exactly two variable names"
  | .unroll =>
    match app.vars with
    | [_] => .ok
    | _ => .err "unroll requires exactly one variable name"
  | .vectorize =>
    match app.vars with
    | [_] => .ok
    | _ => .err "vectorize requires exactly one variable name"
  | .parallel =>
    match app.vars with
    | [_] => .ok
    | _ => .err "parallel requires exactly one variable name"
  | .cacheRead =>
    match app.strParams with
    | [_, _] => .ok
    | _ => .err "cache_read requires original and cache buffer names"

end VeriTac.Compose
