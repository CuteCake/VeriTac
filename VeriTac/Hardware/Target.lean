/-
  VeriTac.Hardware.Target
  Checked data structures for a normalized hardware target and a Metal
  attention launch, together with a machine-checked acceptance boundary.

  `LaunchLegal` is a proposition (a structure of proof fields) carrying every
  fact the checker needs to authorize a launch; `checkLaunch` is its decidable
  Boolean shadow.  The projection theorems below let callers extract the
  individual legality facts from a successful check without re-deriving them.

  Launches may be refined by a small tactic language (`AttentionTactic`) that
  sets the mapping, query tile, or key tile and always recomputes the thread
  count.  `checkTactics` runs a tactic sequence against a target and returns
  the final legal launch (or `none` if any intermediate state is illegal);
  `checkTactics_legal` is its machine-checked correctness theorem.

  Lean proves *conditional* legality for the data supplied to it: `checkLaunch`
  establishes backend, thread, tile, dtype, staging, and threadgroup-memory
  legality given the target description.  It does NOT prove that a physical
  device matches a supplied JSON profile; that provenance is the caller's
  responsibility (re-probe before use).
-/

namespace VeriTac.Hardware

/-- A normalized hardware target.  All fields are unit-bearing; unknown limits
    are simply not legal (the checker can only authorize what the data states). -/
structure Target where
  backend : String
  deviceName : String
  maxThreadsPerThreadgroup : Nat
  maxThreadgroupMemoryBytes : Nat
  simdWidth : Nat
  toolchain : String
  provenance : String
deriving Repr

/-- The two attention data-layout mappings supported on Metal: a scalar layout
    with one thread per query row, and a simdgroup layout with `simdWidth`
    threads per query row. -/
inductive AttentionMapping where
  | scalar
  | simdgroup
deriving DecidableEq, Repr

/-- A Metal attention launch configuration for the first (FP32) kernel.

    Shared-memory staging is FP32; `dtypeBytes` is carried explicitly so the
    byte accounting does not hard-code a dtype.  `stageK`/`stageV` say whether
    the K and V tiles are staged in threadgroup memory at all.  `mapping`
    selects the data layout; the thread count is derived from it (see
    `derivedThreads`). -/
structure MetalAttentionLaunch where
  mapping : AttentionMapping := .scalar
  headDim : Nat
  queryTile : Nat
  keyTile : Nat
  threadsPerThreadgroup : Nat
  dtypeBytes : Nat
  stageK : Bool
  stageV : Bool
  extraSharedBytes : Nat
deriving Repr

namespace MetalAttentionLaunch

/-- Requested bytes for the staged K tile (0 when K is not staged). -/
def stagedKBytes (l : MetalAttentionLaunch) : Nat :=
  if l.stageK then l.keyTile * l.headDim * l.dtypeBytes else 0

/-- Requested bytes for the staged V tile (0 when V is not staged). -/
def stagedVBytes (l : MetalAttentionLaunch) : Nat :=
  if l.stageV then l.keyTile * l.headDim * l.dtypeBytes else 0

/-- Total threadgroup-memory bytes requested: the simultaneously live staged K
    and V arrays plus any extra scratch.  The checker requires both K and V to
    be staged, so in every accepted launch this is
    `2 * keyTile * headDim * dtypeBytes + extraSharedBytes`. -/
def sharedBytes (l : MetalAttentionLaunch) : Nat :=
  stagedKBytes l + stagedVBytes l + l.extraSharedBytes

end MetalAttentionLaunch

open MetalAttentionLaunch

/-- The thread count a launch requests for its mapping on a given target:
    scalar maps one thread per query row; simdgroup maps `simdWidth` threads
    per query row. -/
def derivedThreads (t : Target) (l : MetalAttentionLaunch) : Nat :=
  match l.mapping with
  | .scalar => l.queryTile
  | .simdgroup => l.queryTile * t.simdWidth

/-- The mapping-specific legality clause.  A scalar layout must use a query
    tile of 8 or 16 with one thread per query row; a simdgroup layout must use
    a query tile of 4 or 8, a `simdWidth` of 32, and `simdWidth` threads per
    query row. -/
def mappingLegal (t : Target) (l : MetalAttentionLaunch) : Prop :=
  match l.mapping with
  | .scalar => (l.queryTile = 8 ∨ l.queryTile = 16) ∧ l.threadsPerThreadgroup = l.queryTile
  | .simdgroup =>
      (l.queryTile = 4 ∨ l.queryTile = 8) ∧ t.simdWidth = 32 ∧
      l.threadsPerThreadgroup = l.queryTile * t.simdWidth

/-- `mappingLegal` is decidable: it is a finite conjunction/disjunction of
    decidable numeric and constructor propositions. -/
instance instDecidableMappingLegal (t : Target) (l : MetalAttentionLaunch) :
    Decidable (mappingLegal t l) := by
  cases hm : l.mapping with
  | scalar =>
      simp [mappingLegal, hm]
      exact inferInstance
  | simdgroup =>
      simp [mappingLegal, hm]
      exact inferInstance

/-- The full legality boundary for a Metal attention launch on a given target.

    Every field is a proof fact the checker commits to: the target is Metal
    with positive limits, the launch uses a supported head dimension and key
    tile, the dtype is FP32 (4 bytes), both K and V are staged, and the exact
    requested threadgroup-memory bytes and thread count fit the target.  The
    `mapping_legal` field is the mapping-specific clause (see `mappingLegal`). -/
structure LaunchLegal (t : Target) (l : MetalAttentionLaunch) : Prop where
  backend_metal : t.backend = "metal"
  target_limits_positive : t.maxThreadsPerThreadgroup > 0 ∧ t.maxThreadgroupMemoryBytes > 0
  head_supported : l.headDim = 192 ∨ l.headDim = 256
  key_tile_supported : l.keyTile = 8 ∨ l.keyTile = 16
  dtype_bytes : l.dtypeBytes = 4
  stage_k : l.stageK = true
  stage_v : l.stageV = true
  shared_within : sharedBytes l ≤ t.maxThreadgroupMemoryBytes
  threads_within_target : l.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup
  mapping_legal : mappingLegal t l

/-- `LaunchLegal` is decidable: every field is a decidable proposition, so the
    checker can decide it at runtime. -/
instance instDecidableLaunchLegal (t : Target) (l : MetalAttentionLaunch) :
    Decidable (LaunchLegal t l) :=
  if h1 : t.backend = "metal" then
    if h2 : t.maxThreadsPerThreadgroup > 0 ∧ t.maxThreadgroupMemoryBytes > 0 then
      if h3 : l.headDim = 192 ∨ l.headDim = 256 then
        if h4 : l.keyTile = 8 ∨ l.keyTile = 16 then
          if h5 : l.dtypeBytes = 4 then
            if h6 : l.stageK = true then
              if h7 : l.stageV = true then
                if h8 : sharedBytes l ≤ t.maxThreadgroupMemoryBytes then
                  if h9 : l.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup then
                    if hm : mappingLegal t l then
                      isTrue ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, hm⟩
                    else isFalse (fun f => hm f.mapping_legal)
                  else isFalse (fun f => h9 f.threads_within_target)
                else isFalse (fun f => h8 f.shared_within)
              else isFalse (fun f => h7 f.stage_v)
            else isFalse (fun f => h6 f.stage_k)
          else isFalse (fun f => h5 f.dtype_bytes)
        else isFalse (fun f => h4 f.key_tile_supported)
      else isFalse (fun f => h3 f.head_supported)
    else isFalse (fun f => h2 f.target_limits_positive)
  else isFalse (fun f => h1 f.backend_metal)

/-- Boolean acceptance predicate for a Metal attention launch: decides whether
    the launch satisfies the `LaunchLegal` boundary. -/
def checkLaunch (t : Target) (l : MetalAttentionLaunch) : Bool :=
  decide (LaunchLegal t l)

/-- A successful check witnesses the full legality boundary.  This is the
    single point that turns `decide` into a proof; the projection theorems
    below are stable, machine-checked projections of that boundary. -/
theorem checkLaunch_legal (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) : LaunchLegal t l :=
  of_decide_eq_true h

/-- Checker success implies the thread count fits the target and equals the
    value derived from the launch's mapping (thread legality). -/
theorem checkLaunch_thread_legal (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.threadsPerThreadgroup ≤ t.maxThreadsPerThreadgroup ∧
    l.threadsPerThreadgroup = derivedThreads t l := by
  have hl := checkLaunch_legal t l h
  cases hm : l.mapping with
  | scalar =>
      simp [derivedThreads, hm]
      have hmap := hl.mapping_legal
      simp [mappingLegal, hm] at hmap
      exact ⟨hl.threads_within_target, hmap.2⟩
  | simdgroup =>
      simp [derivedThreads, hm]
      have hmap := hl.mapping_legal
      simp [mappingLegal, hm] at hmap
      exact ⟨hl.threads_within_target, hmap.2.2⟩

/-- Checker success implies the exact requested threadgroup-memory bytes fit
    within the target's threadgroup-memory limit (shared-memory legality). -/
theorem checkLaunch_shared_legal (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    sharedBytes l ≤ t.maxThreadgroupMemoryBytes := by
  have hl := checkLaunch_legal t l h
  exact hl.shared_within

/-- Checker success implies the launch's thread count equals the value derived
    from its mapping (structural invariant of the attention layout). -/
theorem checkLaunch_threads_derived (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.threadsPerThreadgroup = derivedThreads t l := by
  have hl := checkLaunch_legal t l h
  cases hm : l.mapping with
  | scalar =>
      simp [derivedThreads, hm]
      have hmap := hl.mapping_legal
      simp [mappingLegal, hm] at hmap
      exact hmap.2
  | simdgroup =>
      simp [derivedThreads, hm]
      have hmap := hl.mapping_legal
      simp [mappingLegal, hm] at hmap
      exact hmap.2.2

/-- Checker success implies the head dimension is one of the supported values. -/
theorem checkLaunch_head_supported (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.headDim = 192 ∨ l.headDim = 256 := by
  have hl := checkLaunch_legal t l h
  exact hl.head_supported

/-- Checker success implies the key tile is one of the supported shapes. -/
theorem checkLaunch_key_tile_supported (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.keyTile = 8 ∨ l.keyTile = 16 := by
  have hl := checkLaunch_legal t l h
  exact hl.key_tile_supported

/-- Checker success implies the dtype is FP32 (4 bytes). -/
theorem checkLaunch_dtype (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.dtypeBytes = 4 := by
  have hl := checkLaunch_legal t l h
  exact hl.dtype_bytes

/-- Checker success implies both K and V are staged (no partial staging). -/
theorem checkLaunch_stages_both (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    l.stageK = true ∧ l.stageV = true := by
  have hl := checkLaunch_legal t l h
  exact ⟨hl.stage_k, hl.stage_v⟩

/-- Checker success implies the launch targets the Metal backend. -/
theorem checkLaunch_backend (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    t.backend = "metal" := by
  have hl := checkLaunch_legal t l h
  exact hl.backend_metal

/-- Checker success implies the target limits are positive (no degenerate,
    zero-limit target can authorize a launch). -/
theorem checkLaunch_target_limits_positive (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) :
    t.maxThreadsPerThreadgroup > 0 ∧ t.maxThreadgroupMemoryBytes > 0 := by
  have hl := checkLaunch_legal t l h
  exact hl.target_limits_positive

/-- Checker success for a scalar launch implies its query tile is 8 or 16 with
    one thread per query row (scalar mapping clause). -/
theorem checkLaunch_mapping_scalar (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) (hm : l.mapping = .scalar) :
    (l.queryTile = 8 ∨ l.queryTile = 16) ∧ l.threadsPerThreadgroup = l.queryTile := by
  have hl := checkLaunch_legal t l h
  simpa [mappingLegal, hm] using hl.mapping_legal

/-- Checker success for a simdgroup launch implies its query tile is 4 or 8, a
    simdWidth of 32, and `simdWidth` threads per query row (simdgroup clause). -/
theorem checkLaunch_mapping_simd (t : Target) (l : MetalAttentionLaunch)
    (h : checkLaunch t l = true) (hm : l.mapping = .simdgroup) :
    (l.queryTile = 4 ∨ l.queryTile = 8) ∧ t.simdWidth = 32 ∧
    l.threadsPerThreadgroup = l.queryTile * t.simdWidth := by
  have hl := checkLaunch_legal t l h
  simpa [mappingLegal, hm] using hl.mapping_legal

/-- A tactic that refines an attention launch configuration: it sets one
    requested field (mapping, query tile, or key tile).  Thread count is always
    recomputed afterwards (see `applyTactic`). -/
inductive AttentionTactic where
  | setMapping (m : AttentionMapping)
  | setQueryTile (q : Nat)
  | setKeyTile (k : Nat)

/-- Apply a single tactic to a launch: update the requested field and always
    recompute the thread count from the (possibly new) mapping and query tile
    on the given target. -/
def applyTactic (t : Target) (l : MetalAttentionLaunch) (tac : AttentionTactic) : MetalAttentionLaunch :=
  match tac with
  | .setMapping m => { l with mapping := m, threadsPerThreadgroup := derivedThreads t { l with mapping := m } }
  | .setQueryTile q => { l with queryTile := q, threadsPerThreadgroup := derivedThreads t { l with queryTile := q } }
  | .setKeyTile k => { l with keyTile := k, threadsPerThreadgroup := derivedThreads t { l with keyTile := k } }

def checkTacticsAux (t : Target) (l : MetalAttentionLaunch) :
    List AttentionTactic → Option MetalAttentionLaunch
  | [] => some l
  | tac :: rest =>
      if checkLaunch t (applyTactic t l tac) then
        checkTacticsAux t (applyTactic t l tac) rest
      else none

/-- Run a tactic sequence against a target: the `initial` launch and every
    state after a tactic must be legal, else `none`; otherwise the final
    launch. -/
def checkTactics (t : Target) (initial : MetalAttentionLaunch)
    (tactics : List AttentionTactic) : Option MetalAttentionLaunch :=
  if checkLaunch t initial then checkTacticsAux t initial tactics else none

/-- Every state accepted along a tactic sequence is legal: if `checkTacticsAux`
    returns a launch, that launch satisfies the `LaunchLegal` boundary. -/
theorem checkTacticsAux_legal (t : Target) (l : MetalAttentionLaunch) :
    ∀ (tactics : List AttentionTactic) (final : MetalAttentionLaunch),
      LaunchLegal t l → checkTacticsAux t l tactics = some final → LaunchLegal t final := by
  intro tactics
  induction tactics generalizing l with
  | nil =>
      intro final hl h
      simp [checkTacticsAux] at h
      subst final
      exact hl
  | cons tac rest ih =>
      intro final hl h
      simp [checkTacticsAux] at h
      cases hc : checkLaunch t (applyTactic t l tac) with
      | false =>
          simp [hc] at h
      | true =>
          simp [hc] at h
          have hl' : LaunchLegal t (applyTactic t l tac) :=
            checkLaunch_legal t (applyTactic t l tac) hc
          exact ih (applyTactic t l tac) final hl' h

/-- Correctness of `checkTactics`: if it returns a final launch, that launch is
    legal on the given target. -/
theorem checkTactics_legal (t : Target) (initial : MetalAttentionLaunch)
    (tactics : List AttentionTactic) (final : MetalAttentionLaunch)
    (h : checkTactics t initial tactics = some final) :
    LaunchLegal t final := by
  simp [checkTactics] at h
  cases hc : checkLaunch t initial with
  | false =>
      simp [hc] at h
  | true =>
      simp [hc] at h
      have hl : LaunchLegal t initial := checkLaunch_legal t initial hc
      exact checkTacticsAux_legal t initial tactics final hl h

end VeriTac.Hardware
