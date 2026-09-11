/-
  VeriTac.Attention.CudaPlan
  Checked CUDA config-selection tactics with the same real partition semantics
  as the Metal tiling plan, but driven by a COMPILED resource catalogue rather
  than a padded shared-memory formula.

  A `CudaConfig` records the raw C++ `config_info` observations emitted by the
  CUDA runtime manifest: id, queries_per_block, keys_per_block, max_k,
  num_threads, dynamic smem_bytes, static_shared_bytes, kernel_max_threads and a
  `supported` flag.  Total shared memory is the SUM of the compiled dynamic and
  static bytes — we never borrow the Metal K/V-aliasing memory model.

  Hardware and compiler facts (device limits, warp width, the catalogue itself)
  are TRUSTED OBSERVATIONS, not proved physical truth: Lean only checks that the
  plan is resource-legal and semantically partition-correct GIVEN those
  observations.  Binding a selected config to a specific executable hash is the
  caller's (controller's) responsibility and is explicitly outside the theorems
  below.

  Resource legality of a selected config:
    - backend cuda, warp (simd_width) 32, positive device limits / seq_len
    - head_dim 192 or 256 and <= config.max_k
    - queries_per_block, keys_per_block positive multiples of 32
    - num_threads = Q * K / 32, num_threads <= kernel_max_threads and device max
    - dynamic_smem + static_shared <= device max threadgroup memory
    - config.supported = true
  We do NOT hardcode a shared-memory formula, and we do NOT assume occupancy
  from any `min_blocks_hint` / `num_regs` metadata.

  For every causal row prefix 1..seq_len the key-tile range partition
  `partitionRanges r keys_per_block` is checked legal (a finite `List.all` over
  the bounded index set), and `merged_equals_direct` then yields that the
  per-row merged stable-softmax output equals the direct Real attention output,
  so two accepted configs with different tiles / launch hints still agree on the
  abstract row semantics.
-/
import VeriTac.Hardware.Target
import VeriTac.Attention.Plan

namespace VeriTac.Attention

open VeriTac.Hardware

/-! ## Compiled CUDA config observations -/

/-- A compiled CUDA `config_info` entry (a raw manifest observation). -/
structure CudaConfig where
  id : Nat
  queriesPerBlock : Nat
  keysPerBlock : Nat
  maxK : Nat
  numThreads : Nat
  dynamicSmemBytes : Nat
  staticSharedBytes : Nat
  kernelMaxThreads : Nat
  supported : Bool
  deriving Repr, DecidableEq

namespace CudaConfig

/-- A default config, never usable as a legal selection. -/
def zero : CudaConfig :=
  { id := 0, queriesPerBlock := 0, keysPerBlock := 0, maxK := 0, numThreads := 0,
    dynamicSmemBytes := 0, staticSharedBytes := 0, kernelMaxThreads := 0,
    supported := false }

/-- Total compiled shared memory = dynamic + static (never a Metal-style alias). -/
def totalSmem (c : CudaConfig) : Nat := c.dynamicSmemBytes + c.staticSharedBytes

/-- Derived thread count from the block geometry: `Q * K / 32`. -/
def derivedThreads (c : CudaConfig) : Nat := c.queriesPerBlock * c.keysPerBlock / 32

end CudaConfig

/-- Lookup a config in the catalogue by id.  The catalogue is immutable for the
    whole tactic sequence. -/
def lookupCudaConfig (catalog : List CudaConfig) (id : Nat) : Option CudaConfig :=
  catalog.find? (fun c => c.id = id)

/-- No two catalogue entries share an id (duplicate ids are rejected). -/
def NoDuplicateIds (catalog : List CudaConfig) : Prop :=
  (catalog.map CudaConfig.id).Nodup

/-- The selected config for a plan id (defaults to `zero` when not found). -/
def selectedConfig (catalog : List CudaConfig) (id : Nat) : CudaConfig :=
  (lookupCudaConfig catalog id).getD CudaConfig.zero

/-- A config-selection plan over a CUDA catalogue. -/
structure CudaPlan where
  seqLen : Nat
  headDim : Nat
  configId : Nat
  deriving Repr

/-- Per-row causal partition legality for a keys-per-block `K`: every row prefix
    `1..seqLen` (indexed by `r ∈ 0..seqLen-1`) has a legal key-tile range
    partition. -/
def cudaPlansLegalPartitions (seqLen K : Nat) : Bool :=
  (List.range seqLen).all (fun r => legalCausalPartition (r + 1) (partitionRanges (r + 1) K))

/-! ## Legality -/

/-- Config-level resource legality for a given device (independent of the plan). -/
structure CudaConfigLegal (dev : Target) (c : CudaConfig) : Prop where
  supported : c.supported = true
  q_positive : c.queriesPerBlock > 0
  k_positive : c.keysPerBlock > 0
  q_mult32 : c.queriesPerBlock % 32 = 0
  k_mult32 : c.keysPerBlock % 32 = 0
  threads_eq : c.numThreads = CudaConfig.derivedThreads c
  threads_le_kernel : c.numThreads ≤ c.kernelMaxThreads
  threads_le_device : c.numThreads ≤ dev.maxThreadsPerThreadgroup
  smem_within : CudaConfig.totalSmem c ≤ dev.maxThreadgroupMemoryBytes

/-- Plan-level legality for a given device and selected config. -/
structure CudaPlanLegal (dev : Target) (plan : CudaPlan) (c : CudaConfig) : Prop where
  backend_cuda : dev.backend = "cuda"
  warp_32 : dev.simdWidth = 32
  limits_positive : dev.maxThreadsPerThreadgroup > 0 ∧ dev.maxThreadgroupMemoryBytes > 0
  seq_len_positive : plan.seqLen > 0
  head_supported : plan.headDim = 192 ∨ plan.headDim = 256
  head_within_maxk : plan.headDim ≤ c.maxK
  config_legal : CudaConfigLegal dev c
  partitions_legal : cudaPlansLegalPartitions plan.seqLen c.keysPerBlock = true

/-- Full checked legality of a plan against a catalogue: no duplicate ids, the
    selected config is present, its id matches, and the plan + config are legal. -/
structure CudaCheckLegal (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan) : Prop where
  no_duplicate_ids : NoDuplicateIds catalog
  config_in_catalog : selectedConfig catalog plan.configId ∈ catalog
  config_id_matches : (selectedConfig catalog plan.configId).id = plan.configId
  config_legal : CudaPlanLegal dev plan (selectedConfig catalog plan.configId)

/-! ## Decidability -/

instance instDecidableNoDuplicateIds (catalog : List CudaConfig) : Decidable (NoDuplicateIds catalog) := by
  unfold NoDuplicateIds
  exact inferInstance

instance instDecidableCudaConfigLegal (dev : Target) (c : CudaConfig) :
    Decidable (CudaConfigLegal dev c) :=
  if h1 : c.supported = true then
    if h2 : c.queriesPerBlock > 0 then
      if h3 : c.keysPerBlock > 0 then
        if h4 : c.queriesPerBlock % 32 = 0 then
          if h5 : c.keysPerBlock % 32 = 0 then
            if h6 : c.numThreads = CudaConfig.derivedThreads c then
              if h7 : c.numThreads ≤ c.kernelMaxThreads then
                if h8 : c.numThreads ≤ dev.maxThreadsPerThreadgroup then
                  if h9 : CudaConfig.totalSmem c ≤ dev.maxThreadgroupMemoryBytes then
                    isTrue ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩
                  else isFalse (fun f => h9 f.smem_within)
                else isFalse (fun f => h8 f.threads_le_device)
              else isFalse (fun f => h7 f.threads_le_kernel)
            else isFalse (fun f => h6 f.threads_eq)
          else isFalse (fun f => h5 f.k_mult32)
        else isFalse (fun f => h4 f.q_mult32)
      else isFalse (fun f => h3 f.k_positive)
    else isFalse (fun f => h2 f.q_positive)
  else isFalse (fun f => h1 f.supported)

instance instDecidableCudaPlanLegal (dev : Target) (plan : CudaPlan) (c : CudaConfig) :
    Decidable (CudaPlanLegal dev plan c) :=
  if h1 : dev.backend = "cuda" then
    if h2 : dev.simdWidth = 32 then
      if h3 : dev.maxThreadsPerThreadgroup > 0 ∧ dev.maxThreadgroupMemoryBytes > 0 then
        if h4 : plan.seqLen > 0 then
          if h5 : plan.headDim = 192 ∨ plan.headDim = 256 then
            if h6 : plan.headDim ≤ c.maxK then
              if h7 : CudaConfigLegal dev c then
                if h8 : cudaPlansLegalPartitions plan.seqLen c.keysPerBlock = true then
                  isTrue ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
                else isFalse (fun f => h8 f.partitions_legal)
              else isFalse (fun f => h7 f.config_legal)
            else isFalse (fun f => h6 f.head_within_maxk)
          else isFalse (fun f => h5 f.head_supported)
        else isFalse (fun f => h4 f.seq_len_positive)
      else isFalse (fun f => h3 f.limits_positive)
    else isFalse (fun f => h2 f.warp_32)
  else isFalse (fun f => h1 f.backend_cuda)

instance instDecidableCudaCheckLegal (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan) :
    Decidable (CudaCheckLegal dev catalog plan) :=
  if h1 : NoDuplicateIds catalog then
    if h2 : selectedConfig catalog plan.configId ∈ catalog then
      if h3 : (selectedConfig catalog plan.configId).id = plan.configId then
        if h4 : CudaPlanLegal dev plan (selectedConfig catalog plan.configId) then
          isTrue ⟨h1, h2, h3, h4⟩
        else isFalse (fun f => h4 f.config_legal)
      else isFalse (fun f => h3 f.config_id_matches)
    else isFalse (fun f => h2 f.config_in_catalog)
  else isFalse (fun f => h1 f.no_duplicate_ids)

/-- Boolean acceptance predicate for a CUDA plan against a catalogue. -/
def checkCudaPlan (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan) : Bool :=
  decide (CudaCheckLegal dev catalog plan)

/-! ## Projection theorems -/

/-- A successful check witnesses the full `CudaCheckLegal` boundary. -/
theorem checkCudaPlan_legal (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) : CudaCheckLegal dev catalog plan :=
  of_decide_eq_true h

/-- Checker success implies a selected config exists (is present in the catalogue
    with the requested id and is fully legal). -/
theorem checkCudaPlan_selected_exists (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) :
    ∃ cfg, cfg ∈ catalog ∧ cfg.id = plan.configId ∧ CudaPlanLegal dev plan cfg := by
  have hl := checkCudaPlan_legal dev catalog plan h
  exact ⟨selectedConfig catalog plan.configId, hl.config_in_catalog,
         hl.config_id_matches, hl.config_legal⟩

/-- Checker success implies the catalogue has no duplicate ids. -/
theorem checkCudaPlan_no_duplicates (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) : NoDuplicateIds catalog :=
  (checkCudaPlan_legal dev catalog plan h).no_duplicate_ids

/-- Checker success implies the selected config's compiled total shared memory
    fits the device limit (dynamic + static). -/
theorem checkCudaPlan_smem_within (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) :
    CudaConfig.totalSmem (selectedConfig catalog plan.configId) ≤ dev.maxThreadgroupMemoryBytes :=
  (checkCudaPlan_legal dev catalog plan h).config_legal.config_legal.smem_within

/-- Checker success implies the selected config's thread count is the derived
    `Q * K / 32` and fits both the kernel and device limits. -/
theorem checkCudaPlan_threads (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) :
    (selectedConfig catalog plan.configId).numThreads =
      CudaConfig.derivedThreads (selectedConfig catalog plan.configId) ∧
    (selectedConfig catalog plan.configId).numThreads ≤ dev.maxThreadsPerThreadgroup := by
  have hc := (checkCudaPlan_legal dev catalog plan h).config_legal.config_legal
  exact ⟨hc.threads_eq, hc.threads_le_device⟩

/-- `List.all` soundness for the per-row partition check: a true `all` holds for
    every bounded row index. -/
theorem cuda_plan_partitions_legal (seqLen K : Nat) (h : cudaPlansLegalPartitions seqLen K = true) :
    ∀ r, r < seqLen → legalCausalPartition (r + 1) (partitionRanges (r + 1) K) = true := by
  intro r hr
  have hmem : r ∈ List.range seqLen := List.mem_range.mpr hr
  exact (List.all_eq_true.mp h) r hmem

/-- Checker success implies every causal row prefix has a legal key-tile range
    partition for the selected config's keys-per-block. -/
theorem checkCudaPlan_row_partition_legal (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (h : checkCudaPlan dev catalog plan = true) (r : Nat) (hr : r < plan.seqLen) :
    legalCausalPartition (r + 1)
      (partitionRanges (r + 1) (selectedConfig catalog plan.configId).keysPerBlock) = true :=
  cuda_plan_partitions_legal plan.seqLen (selectedConfig catalog plan.configId).keysPerBlock
    (checkCudaPlan_legal dev catalog plan h).config_legal.partitions_legal r hr

/-! ## Per-row Real semantic preservation -/

/-- The merged (rescaled) stable-softmax output over a legally partitioned row
    prefix, at keys-per-block `K`. -/
noncomputable def cudaRowMerged (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (K : Nat) (pre : Nat) : ℝ :=
  ((partitionRanges pre K).map
      (fun pt => num value score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum /
    ((partitionRanges pre K).map
      (fun pt => denom score (m pt) (rangeSet pt) * Real.exp (m pt - M))).sum

/-- For a legally partitioned causal row prefix, the merged softmax output equals
    the direct softmax output over the whole prefix (specialization of
    `merged_equals_direct` to a key-tile range partition). -/
theorem cuda_row_merged_equals_direct (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (pre K : Nat) (hlegal : legalCausalPartition pre (partitionRanges pre K) = true) :
    cudaRowMerged value score m M K pre =
      num value score M (Finset.range pre) / denom score M (Finset.range pre) := by
  unfold cudaRowMerged
  exact merged_equals_direct value score pre (partitionRanges pre K) m M hlegal

/-- Main objective: an accepted CUDA plan preserves per-row Real softmax
    semantics — the tile-merged output equals the direct output for every row,
    for arbitrary score / value / reference functions. -/
theorem cuda_accepted_row_preserves (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan)
    (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (h : checkCudaPlan dev catalog plan = true) (r : Nat) (hr : r < plan.seqLen) :
    cudaRowMerged value score m M (selectedConfig catalog plan.configId).keysPerBlock (r + 1) =
      num value score M (Finset.range (r + 1)) / denom score M (Finset.range (r + 1)) := by
  exact cuda_row_merged_equals_direct value score m M (r + 1)
    (selectedConfig catalog plan.configId).keysPerBlock
    (checkCudaPlan_row_partition_legal dev catalog plan h r hr)

/-- Corollary: two accepted configs (possibly with different tiles / launch
    hints) have the same abstract per-row Real semantics. -/
theorem cuda_configs_row_semantics_agree (dev : Target)
    (catalog1 catalog2 : List CudaConfig) (plan : CudaPlan)
    (value score : Nat → ℝ) (m : Range → ℝ) (M : ℝ)
    (h1 : checkCudaPlan dev catalog1 plan = true)
    (h2 : checkCudaPlan dev catalog2 plan = true)
    (r : Nat) (hr : r < plan.seqLen) :
    cudaRowMerged value score m M (selectedConfig catalog1 plan.configId).keysPerBlock (r + 1) =
      cudaRowMerged value score m M (selectedConfig catalog2 plan.configId).keysPerBlock (r + 1) := by
  rw [cuda_accepted_row_preserves dev catalog1 plan value score m M h1 r hr]
  rw [cuda_accepted_row_preserves dev catalog2 plan value score m M h2 r hr]

/-! ## Config-selection tactics -/

/-- A tactic that selects a config from the catalogue by id. -/
inductive CudaTactic where
  | selectConfig (id : Nat)

/-- Apply a single tactic: change the selected config id (catalogue is
    immutable). -/
def applyCudaTactic (plan : CudaPlan) (tac : CudaTactic) : CudaPlan :=
  match tac with
  | .selectConfig id => { plan with configId := id }

/-- Run a tactic sequence: every intermediate plan must be legal against the
    (fixed) catalogue, else `none`. -/
def checkCudaTacticsAux (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan) :
    List CudaTactic → Option CudaPlan
  | [] => some plan
  | tac :: rest =>
      if checkCudaPlan dev catalog (applyCudaTactic plan tac) then
        checkCudaTacticsAux dev catalog (applyCudaTactic plan tac) rest
      else none

/-- Run a tactic sequence against a target/catalogue: the initial plan and every
    state after a tactic must be legal, else `none`; otherwise the final plan. -/
def checkCudaTactics (dev : Target) (catalog : List CudaConfig) (initial : CudaPlan)
    (tactics : List CudaTactic) : Option CudaPlan :=
  if checkCudaPlan dev catalog initial then checkCudaTacticsAux dev catalog initial tactics else none

/-- Every state accepted along a tactic sequence is legal. -/
theorem checkCudaTacticsAux_legal (dev : Target) (catalog : List CudaConfig) (plan : CudaPlan) :
    ∀ (tactics : List CudaTactic) (final : CudaPlan),
      CudaCheckLegal dev catalog plan →
      checkCudaTacticsAux dev catalog plan tactics = some final → CudaCheckLegal dev catalog final := by
  intro tactics
  induction tactics generalizing plan with
  | nil =>
      intro final hl h
      simp [checkCudaTacticsAux] at h
      subst final
      exact hl
  | cons tac rest ih =>
      intro final hl h
      simp [checkCudaTacticsAux] at h
      cases hc : checkCudaPlan dev catalog (applyCudaTactic plan tac) with
      | false => simp [hc] at h
      | true =>
          simp [hc] at h
          have hl' : CudaCheckLegal dev catalog (applyCudaTactic plan tac) :=
            checkCudaPlan_legal dev catalog (applyCudaTactic plan tac) hc
          exact ih (applyCudaTactic plan tac) final hl' h

/-- Correctness of `checkCudaTactics`: a returned final plan is legal against the
    catalogue.  This is the proved core the CLI accepted branch must call. -/
theorem checkCudaTactics_legal (dev : Target) (catalog : List CudaConfig) (initial : CudaPlan)
    (tactics : List CudaTactic) (final : CudaPlan)
    (h : checkCudaTactics dev catalog initial tactics = some final) :
    CudaCheckLegal dev catalog final := by
  simp [checkCudaTactics] at h
  cases hc : checkCudaPlan dev catalog initial with
  | false => simp [hc] at h
  | true =>
      simp [hc] at h
      have hl : CudaCheckLegal dev catalog initial := checkCudaPlan_legal dev catalog initial hc
      exact checkCudaTacticsAux_legal dev catalog initial tactics final hl h

/-! ## Examples (synthetic catalogue from known runtime observations) -/

/-- The synthetic catalogue used by the CPU tests: id1 is legal for D192/D256
    (Q32 K128, 128 threads, 76288 dyn + 0 static, kernel max 128, supported);
    id2 is legal (Q32 K64, 64 threads, 37632 dyn); id0 is unsupported (Q32 K256,
    256 threads, 133632 dyn > a 101376-byte device budget). -/
def syntheticCatalog : List CudaConfig :=
  [ { id := 1, queriesPerBlock := 32, keysPerBlock := 128, maxK := 256,
      numThreads := 128, dynamicSmemBytes := 76288, staticSharedBytes := 0,
      kernelMaxThreads := 128, supported := true },
    { id := 2, queriesPerBlock := 32, keysPerBlock := 64, maxK := 256,
      numThreads := 64, dynamicSmemBytes := 37632, staticSharedBytes := 0,
      kernelMaxThreads := 128, supported := true },
    { id := 0, queriesPerBlock := 32, keysPerBlock := 256, maxK := 256,
      numThreads := 256, dynamicSmemBytes := 133632, staticSharedBytes := 0,
      kernelMaxThreads := 256, supported := false } ]

/-- A synthetic CUDA device with a 101376-byte threadgroup-memory budget and a
    32-wide warp, matching the CPU test manifest. -/
def syntheticCudaDevice : Target :=
  { backend := "cuda", deviceName := "test-cuda", maxThreadsPerThreadgroup := 1024,
    maxThreadgroupMemoryBytes := 101376, simdWidth := 32, toolchain := "",
    provenance := "" }

/-- Selecting legal config id 1 for seq 128, head_dim 192 is accepted. -/
example :
    checkCudaPlan syntheticCudaDevice syntheticCatalog
      { seqLen := 128, headDim := 192, configId := 1 } = true := by
  decide

/-- Selecting the unsupported config id 0 is rejected. -/
example :
    checkCudaPlan syntheticCudaDevice syntheticCatalog
      { seqLen := 128, headDim := 192, configId := 0 } = false := by
  decide

end VeriTac.Attention
