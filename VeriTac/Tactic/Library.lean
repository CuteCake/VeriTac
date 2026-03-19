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

/-- Apply a single tactic application to a statement. -/
def applyTactic (app : TacticApp) (stmt : Stmt) : Option Stmt :=
  match app.kind with
  | .tile =>
    match app.vars, app.intParams with
    | [v], [ts] => tile stmt v ts
    | _, _ => none
  | .split =>
    match app.vars, app.intParams with
    | [v], [f] => split stmt v f
    | _, _ => none
  | .fuse =>
    match app.vars with
    | [v1, v2] => fuse stmt v1 v2
    | _ => none
  | .reorder =>
    match app.vars with
    | [v1, v2] => reorder stmt v1 v2
    | _ => none
  | .unroll =>
    match app.vars with
    | [v] => unroll stmt v
    | _ => none
  | .vectorize =>
    match app.vars with
    | [v] => vectorize stmt v
    | _ => none
  | .parallel =>
    match app.vars with
    | [v] => parallel stmt v
    | _ => none
  | .cacheRead =>
    -- cache_read requires more complex parameters; simplified here
    none

end VeriTac.Tactic
