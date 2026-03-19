/-
  VeriTac.Tactic.CacheRead
  Cache read transformation: insert local buffer + copy + substitute reads.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Tactic.Tile

namespace VeriTac.Tactic

/-- Replace all reads of `origBuf` with reads from `cacheBuf` in an expression. -/
def substBufReadExpr (origBuf cacheBuf : BufId) : SExpr → SExpr
  | .lit n => .lit n
  | .var v => .var v
  | .add a b => .add (substBufReadExpr origBuf cacheBuf a)
                     (substBufReadExpr origBuf cacheBuf b)
  | .mul a b => .mul (substBufReadExpr origBuf cacheBuf a)
                     (substBufReadExpr origBuf cacheBuf b)
  | .div a b => .div (substBufReadExpr origBuf cacheBuf a)
                     (substBufReadExpr origBuf cacheBuf b)
  | .mod a b => .mod (substBufReadExpr origBuf cacheBuf a)
                     (substBufReadExpr origBuf cacheBuf b)
  | .bufRead buf indices =>
    if buf == origBuf then .bufRead cacheBuf indices
    else .bufRead buf (indices.map (substBufReadExpr origBuf cacheBuf))

/-- Replace all reads of `origBuf` with reads from `cacheBuf` in a statement. -/
def substBufReadStmt (origBuf cacheBuf : BufId) : Stmt → Stmt
  | .skip => .skip
  | .bufWrite buf indices val =>
    .bufWrite buf (indices.map (substBufReadExpr origBuf cacheBuf))
                  (substBufReadExpr origBuf cacheBuf val)
  | .loop v lo hi ann body =>
    .loop v (substBufReadExpr origBuf cacheBuf lo)
           (substBufReadExpr origBuf cacheBuf hi)
           ann (substBufReadStmt origBuf cacheBuf body)
  | .seq s1 s2 => .seq (substBufReadStmt origBuf cacheBuf s1)
                       (substBufReadStmt origBuf cacheBuf s2)
  | .alloc buf shape body =>
    .alloc buf (shape.map (substBufReadExpr origBuf cacheBuf))
               (substBufReadStmt origBuf cacheBuf body)

/-- Check if a buffer is only read (never written) in a statement. -/
def isReadOnly (buf : BufId) : Stmt → Bool
  | .skip => true
  | .bufWrite b _ _ => b != buf
  | .loop _ _ _ _ body => isReadOnly buf body
  | .seq s1 s2 => isReadOnly buf s1 && isReadOnly buf s2
  | .alloc _ _ body => isReadOnly buf body

/-- Apply cache_read: allocate a cache buffer, copy data, and substitute reads.
    `cacheShape` specifies the dimensions of the cache buffer.
    `copyLoopVars` and `copyBounds` define the copy loop nest. -/
def cacheRead (stmt : Stmt) (origBuf : BufId) (cacheBuf : BufId)
    (cacheShape : List SExpr) (copyIndices : List (VarId × SExpr))
    : Option Stmt :=
  if !isReadOnly origBuf stmt then none
  else
    -- Build copy loop: nested loops that copy from origBuf to cacheBuf
    let copyBody := Stmt.bufWrite cacheBuf
      (copyIndices.map fun (v, _) => .var v)
      (.bufRead origBuf (copyIndices.map fun (v, _) => .var v))
    let copyLoops := copyIndices.foldr
      (fun (v, bound) acc => Stmt.loop v (.lit 0) bound .none acc) copyBody
    -- Substitute reads in original statement
    let newBody := substBufReadStmt origBuf cacheBuf stmt
    some (.alloc cacheBuf cacheShape (.seq copyLoops newBody))

/-- Cache read correctness: reading from a correctly-populated cache is equivalent
    to reading from the original buffer. -/
theorem cacheRead_correct (origBuf cacheBuf : BufId) (stmt : Stmt)
    (hro : isReadOnly origBuf stmt = true)
    (cacheShape : List SExpr) (copyIndices : List (VarId × SExpr)) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      execStmt fuel env store stmt =
      execStmt fuel env store
        (match cacheRead stmt origBuf cacheBuf cacheShape copyIndices with
         | some s => s
         | none => stmt) := by
  sorry -- Requires showing: copy loop populates cache correctly, reads are equivalent

end VeriTac.Tactic
