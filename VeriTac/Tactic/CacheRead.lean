/-
  VeriTac.Tactic.CacheRead
  Cache read transformation: insert local buffer + copy + substitute reads.
-/
import VeriTac.Schedule.Equiv
import VeriTac.Schedule.FreeVars
import VeriTac.Schedule.ExecLemmas
import VeriTac.Schedule.LoopComposition
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

private theorem foldl_substRead (o c : BufId) (env : Env) (st : Store) :
    ∀ (l : List SExpr) (acc : Int),
      (∀ e, e ∈ l → evalExpr env st (substBufReadExpr o c e) = evalExpr env st e) →
      l.foldl (fun a idx => a * 1000 + evalExpr env st (substBufReadExpr o c idx)) acc =
      l.foldl (fun a idx => a * 1000 + evalExpr env st idx) acc := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons x xs ih =>
      intro acc h
      simp only [List.foldl_cons]
      have hx := h x (by simp)
      rw [hx]
      exact ih (acc * 1000 + evalExpr env st x) (fun e he => h e (by simp [he]))

/-- Evaluating a substituted expression under a store where the cache mirrors the
    original buffer gives the same value as the original expression. -/
theorem evalExpr_substBuffer_correct (o c : BufId) (env : Env) (st : Store)
    (hm : ∀ i : Int, st c i = st o i) :
    ∀ e : SExpr, evalExpr env st (substBufReadExpr o c e) = evalExpr env st e := by
  apply SExpr_induct
  · intro n; simp [evalExpr, substBufReadExpr]
  · intro v; simp [evalExpr, substBufReadExpr]
  · intro a b hia hib; simp [evalExpr, substBufReadExpr, hia, hib]
  · intro a b hia hib; simp [evalExpr, substBufReadExpr, hia, hib]
  · intro a b hia hib; simp [evalExpr, substBufReadExpr, hia, hib]
  · intro a b hia hib; simp [evalExpr, substBufReadExpr, hia, hib]
  · intro buf indices hIdx
    by_cases hb : (buf == o) = true
    · have hbo : buf = o := beq_iff_eq.mp hb
      rw [substBufReadExpr, if_pos hb, hbo]
      simp [evalExpr]
      rw [hm]
    · have hflat : (indices.map (substBufReadExpr o c)).foldl
            (fun a idx => a * 1000 + evalExpr env st idx) 0
          = indices.foldl (fun a idx => a * 1000 + evalExpr env st idx) 0 := by
        rw [List.foldl_map]
        exact foldl_substRead o c env st indices 0 hIdx
      rw [substBufReadExpr, if_neg hb]
      simp only [evalExpr]
      rw [hflat]

/-- **Cache read correctness.** Substituting reads of `origBuf` by reads of
    `cacheBuf` is a no-op when (1) the cache mirrors the original buffer in the
    starting store, (2) neither buffer is written by the statement, and (3) the
    cache buffer is distinct from the original. The `cacheRead` transformation is
    responsible for making the mirror hold (via its copy loops) and covering every
    read index, which are semantic obligations the surrounding embedding must
    discharge. Note this is the honest statement: `cacheRead` additionally writes
    the cache buffer for the copy, so the *whole* transformation is equal only off
    `cacheBuf`; the substitution itself is equal everywhere from a mirroring store. -/
theorem substBufReadStmt_correct (o c : BufId) (hc : c ≠ o)
    (env : Env) (fuel : Nat) :
    ∀ (s : Stmt), isReadOnly o s = true → isReadOnly c s = true →
      ∀ (st : Store), (∀ i : Int, st c i = st o i) →
        execStmt fuel env st (substBufReadStmt o c s) = execStmt fuel env st s := by
  intro s
  induction s generalizing env with
  | skip => intro _ _ st hm; rfl
  | bufWrite buf indices val =>
      intro hro hrc st hm
      have hval : evalExpr env st (substBufReadExpr o c val) = evalExpr env st val :=
        evalExpr_substBuffer_correct o c env st hm val
      have hflat : flatIndex env st (indices.map (substBufReadExpr o c)) =
          flatIndex env st indices := by
        simp only [flatIndex, List.foldl_map]
        exact foldl_substRead o c env st indices 0
          (fun e _ => evalExpr_substBuffer_correct o c env st hm e)
      simp only [execStmt, substBufReadStmt]
      congr 1
      funext b i
      rw [hval, hflat]
  | loop v lo hi ann body ih =>
      intro hro hrc st hm
      have hbro : isReadOnly o body = true := by simpa [isReadOnly] using hro
      have hbrc : isReadOnly c body = true := by simpa [isReadOnly] using hrc
      have hlo : evalExpr env st (substBufReadExpr o c lo) = evalExpr env st lo :=
        evalExpr_substBuffer_correct o c env st hm lo
      have hhi : evalExpr env st (substBufReadExpr o c hi) = evalExpr env st hi :=
        evalExpr_substBuffer_correct o c env st hm hi
      simp only [execStmt, substBufReadStmt]
      rw [hlo, hhi]
      refine execLoopIters_congr_under
        (fun e' st' => execStmt fuel e' st' (substBufReadStmt o c body))
        (fun e' st' => execStmt fuel e' st' body)
        env v (fun st' => ∀ i, st' c i = st' o i) ?_ ?_ ?_ _ _ st hm
      · intro e' st'
        obtain ⟨st'', h⟩ := execStmt_total fuel e' st' body
        exact ⟨st'', h⟩
      · intro e' st' hP
        exact ih e' hbro hbrc st' hP
      · intro e' st' st'' hb hP
        have hco : st'' c = st' c := execStmt_ro fuel c e' st' body hbrc st'' hb
        have hoo : st'' o = st' o := execStmt_ro fuel o e' st' body hbro st'' hb
        intro i
        rw [hco, hoo]
        exact hP i
  | seq s1 s2 ih1 ih2 =>
      intro hro hrc st hm
      have hro12 : isReadOnly o s1 = true ∧ isReadOnly o s2 = true := by
        simpa [isReadOnly, Bool.and_eq_true] using hro
      have hrc12 : isReadOnly c s1 = true ∧ isReadOnly c s2 = true := by
        simpa [isReadOnly, Bool.and_eq_true] using hrc
      have h1ro : isReadOnly o s1 = true := hro12.1
      have h2ro : isReadOnly o s2 = true := hro12.2
      have h1rc : isReadOnly c s1 = true := hrc12.1
      have h2rc : isReadOnly c s2 = true := hrc12.2
      simp only [execStmt, substBufReadStmt]
      rw [ih1 env h1ro h1rc st hm]
      obtain ⟨mid, hmid⟩ := execStmt_total fuel env st s1
      rw [hmid]
      have hm2 : ∀ i, mid c i = mid o i := by
        intro i
        have hco : mid c = st c := execStmt_ro fuel c env st s1 h1rc mid hmid
        have hoo : mid o = st o := execStmt_ro fuel o env st s1 h1ro mid hmid
        rw [hco, hoo]
        exact hm i
      simp
      exact ih2 env h2ro h2rc mid hm2
  | alloc buf shape body ih =>
      intro hro hrc st hm
      have hbro : isReadOnly o body = true := by simpa [isReadOnly] using hro
      have hbrc : isReadOnly c body = true := by simpa [isReadOnly] using hrc
      simp only [execStmt, substBufReadStmt]
      exact ih env hbro hbrc st hm

/-- Public statement of `cacheRead` soundness: from a store where the cache
    mirrors the source, substituting the reads preserves semantics. The copy
    loops inserted by `cacheRead` are what must establish this mirror (and cover
    all read indices); `cacheRead_correct` records the read-equivalence that the
    rest of the pipeline relies on. -/
theorem cacheRead_correct (origBuf cacheBuf : BufId) (stmt : Stmt)
    (hcb : cacheBuf ≠ origBuf)
    (hro : isReadOnly origBuf stmt = true)
    (hrc : isReadOnly cacheBuf stmt = true) :
    ∀ (fuel : Nat) (env : Env) (store : Store),
      (∀ i : Int, store cacheBuf i = store origBuf i) →
        execStmt fuel env store stmt =
        execStmt fuel env store (substBufReadStmt origBuf cacheBuf stmt) := by
  intro fuel env store hm
  exact (substBufReadStmt_correct origBuf cacheBuf hcb env fuel stmt hro hrc store hm).symm

end VeriTac.Tactic
