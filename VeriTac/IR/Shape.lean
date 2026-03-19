/-
  VeriTac.IR.Shape
  Shape and Index types for the tensor IR.
-/
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic

namespace VeriTac

/-- A tensor shape is a list of dimension sizes. -/
abbrev Shape := List Nat

/-- An index into a tensor of shape `s` is a dependent vector of `Fin` values. -/
def Index : Shape → Type
  | [] => Unit
  | [n] => Fin n
  | n :: rest => Fin n × Index rest

namespace Index

/-- Empty index for scalar shapes. -/
def nil : Index [] := ()

/-- Construct an index by prepending a coordinate. -/
def cons {n : Nat} {s : Shape} (i : Fin n) (idx : Index s) : Index (n :: s) :=
  match s with
  | [] => i
  | _ :: _ => (i, idx)

/-- Get the head coordinate of an index. -/
def head {n : Nat} {s : Shape} (idx : Index (n :: s)) : Fin n :=
  match s with
  | [] => idx
  | _ :: _ => idx.1

/-- Get the tail of an index. -/
def tail {n : Nat} {s : Shape} (idx : Index (n :: s)) : Index s :=
  match s with
  | [] => ()
  | _ :: _ => idx.2

/-- Total number of elements in a shape. -/
def shapeSize : Shape → Nat
  | [] => 1
  | n :: rest => n * shapeSize rest

end Index

end VeriTac
