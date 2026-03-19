/-
  VeriTac.Schedule.Lower
  Lowering from TExpr to LoopNest (naive nested loops).
-/
import VeriTac.IR.Denote
import VeriTac.Schedule.LoopNest

namespace VeriTac

/-- Generate a fresh loop variable name for a given depth. -/
def loopVar (depth : Nat) : VarId := s!"i{depth}"

/-- Lower a shape into nested loops writing to an output buffer.
    Produces the naive (unoptimized) loop nest. -/
def lowerLoops (shape : List Nat) (depth : Nat) (innerBody : Stmt) : Stmt :=
  match shape with
  | [] => innerBody
  | n :: rest =>
    let v := loopVar depth
    .loop v (.lit 0) (.lit n) .none (lowerLoops rest (depth + 1) innerBody)

/-- Build index expressions from loop variables for a given shape. -/
def indexExprsAux : Nat → Nat → List SExpr
  | 0, _ => []
  | n + 1, d => .var (loopVar d) :: indexExprsAux n (d + 1)

def indexExprs (shape : List Nat) (startDepth : Nat) : List SExpr :=
  indexExprsAux shape.length startDepth

/-- Lower a map operation: for each index, write f(input[idx]) to output[idx]. -/
def lowerMap (shape : List Nat) (inputBuf outputBuf : BufId) : Stmt :=
  let indices := indexExprs shape 0
  let readVal := SExpr.bufRead inputBuf indices
  let body := Stmt.bufWrite outputBuf indices readVal
  lowerLoops shape 0 body

/-- Lower a reduce operation along the first axis.
    output[rest_idx] = Σ_{i=0}^{n-1} input[i, rest_idx] -/
def lowerReduce (n : Nat) (restShape : List Nat) (inputBuf outputBuf : BufId) : Stmt :=
  let restIndices := indexExprs restShape 1
  let allIndices := .var (loopVar 0) :: restIndices
  let initBody := Stmt.bufWrite outputBuf restIndices (.lit 0)
  let initLoops := lowerLoops restShape 1 initBody
  let accumBody := Stmt.bufWrite outputBuf restIndices
    (.add (.bufRead outputBuf restIndices) (.bufRead inputBuf allIndices))
  let reduceLoop := .loop (loopVar 0) (.lit 0) (.lit n) .none
    (lowerLoops restShape 1 accumBody)
  .seq initLoops reduceLoop

end VeriTac
