/-
  VeriTac.IR.Operations
  Common tensor operations (matmul, softmax) as TExpr compositions.
-/
import VeriTac.IR.Denote

namespace VeriTac

variable {α : Type} {s : Shape} {n : Nat}

/-- Element-wise addition of two tensors. -/
def tensorAdd [Add α] (a b : TExpr α s) : TExpr α s :=
  .zip (· + ·) a b

/-- Element-wise application of a scalar function. -/
def tensorMap (f : α → α) (a : TExpr α s) : TExpr α s :=
  .map f a

/-- Sum-reduce along the first axis. -/
def tensorSum [Add α] (zero : α) (e : TExpr α (n :: s)) : TExpr α s :=
  .reduce (· + ·) zero e

end VeriTac
