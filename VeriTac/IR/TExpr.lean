/-
  VeriTac.IR.TExpr
  Semantic IR expression type, parameterized over [AddCommMonoid α].
-/
import VeriTac.IR.Shape

namespace VeriTac

/-- Semantic IR for tensor computations, parameterized over element type α. -/
inductive TExpr (α : Type) : Shape → Type where
  /-- A constant (scalar) value. -/
  | const (val : α) : TExpr α []
  /-- A tensor defined by a function from indices to values. -/
  | tensor {s : Shape} (data : Index s → α) : TExpr α s
  /-- Apply a unary function elementwise. -/
  | map {s : Shape} (f : α → α) (e : TExpr α s) : TExpr α s
  /-- Apply a binary function elementwise to two tensors of the same shape. -/
  | zip {s : Shape} (f : α → α → α) (e1 e2 : TExpr α s) : TExpr α s
  /-- Reduce along the first axis using a binary operation and initial value. -/
  | reduce {n : Nat} {s : Shape} (f : α → α → α) (init : α) (e : TExpr α (n :: s)) : TExpr α s

end VeriTac
