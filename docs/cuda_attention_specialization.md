# CUDA Attention Specialization — GB10 (sm_121, 48 SM)

## Summary

Hardware-aware re-instantiation of the **installed** PyTorch mem_eff
`AttentionKernel<float, cutlass::arch::Sm80, true, QTile, KTile, MaxDim,
false, false>` for the FP32 causal prefill operator, tuned on NVIDIA GB10
(sm_121, 48 SM, max opt-in shared memory 101376 B), plus mechanically
generated, vendor-attributed derivatives. **No vendor arithmetic is changed.**

Three hardware-aware levers:

1. **Launch-bounds occupancy hint** (`MinBlocks = 1`): removes all spills
   (local 96–120 B → 0), registers 168 → 225–234.
2. **Launch-order reversal** of independent query blocks (namespace
   `PyTorchMemEffAttentionRev`): balances imbalanced causal work; grid width
   unchanged.
3. **FP32 scratch-lifetime alias + async drains** (namespaces
   `PyTorchMemEffAttentionAlias` / `PyTorchMemEffAttentionDrain`):
   - wrap the adjacent `mm1; epilogue;` members of
     `SharedStorageEpilogueInLoop::SharedStorageAfterMM0` in a union
     (`si` stays independently live), with a class-scope `static_assert(!kPreloadV)`;
   - insert `cutlass::arch::cp_async_fence(); cp_async_wait<0>();` immediately
     before the existing `__syncthreads` after MM0 `mma(...)` and MM1 `mma_pv(...)`.

The union shrinks the multistage shared storage enough to fit a tile that was
just over the opt-in budget, and the drains eliminate the racecheck warnings.

## Confirmed results

The controller confirmed both selected configurations in three paired rounds.
See [the final results report](attention_vendor_results.md) and
`results/cuda-final-confirmation-summary.json` for the full table. Config 25
wins at N1024; config 26 wins at N2048/4096/8192 for D192 and D256. The graph
speedup ranges are 1.033–1.054× at N1024 and 1.120–1.188× at N4096/8192.

Final protocol: seeds1234/4321/90210, alternating candidate/vendor order,
10 warmups, 30 samples, 3 calls per sample, matching input hashes, and distinct
outputs per captured graph call. Earlier exploration timings are retained but
are not substituted for these confirmation measurements.

## Precision

Unmodified installed `OpMultiplyAddFastF32` (CUTLASS `MmaTensorOpFastF32`,
k3xTF32 three-component F32 matmul emulation) — same arithmetic the FP32
"efficient"/mem_eff vendor comparator uses; not a new reduced-precision
substitution; no single-TF32, no FP16, no IEEE scalar-FP32 claim. Frozen
atol 1e-4 / rtol 1e-3 vs an independent float64 reference.

Numerics: 110/110 small-N tests pass (N {17,100,257,1536} × D {192,256}) across
normal, reverse and alias configs; nmax ~1.1e-3–1.9e-3 (vendor 3xTF32 level);
graph outputs match eager.

## Config catalogue (effective ids; `base_config_id` for derivatives)

| id | Tile | MaxK | Hint | Variant | smem B | thr | regs | local B | supported |
|----|------|------|------|---------|-------:|----:|-----:|--------:|-----------|
| 0  | Q32K256 | 256 | 1 | norm | 133632 | 256 | — | — | no (smem) |
| 1  | Q32K128 | 256 | 3 | norm | 76288 | 128 | 168 | 96 | yes |
| 2  | Q32K64  | 256 | 6 | norm | 37632 | 64 | 168 | 104 | yes |
| 3  | Q64K64  | 256 | 3 | norm | 50688 | 128 | 168 | 120 | yes |
| 4  | Q32K128 | 256 | 1 | norm | 76288 | 128 | 225 | 0 | yes |
| 5  | Q64K64  | 256 | 1 | norm | 50688 | 128 | 230 | 0 | yes |
| 6  | Q64K64  | 256 | 2 | norm | 50688 | 128 | 230 | 0 | yes |
| 7  | Q32K64  | 256 | 1 | norm | 37632 | 64 | 234 | 0 | yes |
| 10 | Q64K128 | 256 | 1 | norm | 103424 | 256 | — | — | no (smem) |
| 11 | Q64K128 | 256 | 2 | norm | 103424 | 256 | — | — | no (smem) |
| 12 | Q128K64 | 256 | 1 | norm | 76800 | 256 | 225 | 0 | yes (slower) |
| 13 | Q128K64 | 256 | 2 | norm | 76800 | 256 | 128 | 280 | yes (slower) |
| 14 | Q32K128 | 256 | 3 | REV | 76288 | 128 | 168 | 96 | yes |
| 15 | Q64K64  | 256 | 3 | REV | 50688 | 128 | 168 | 120 | yes |
| 16 | Q32K128 | 256 | 1 | REV | 76288 | 128 | 225 | 0 | yes (undrained experiment) |
| 17 | Q64K64  | 256 | 1 | REV | 50688 | 128 | 230 | 0 | yes |
| 20 | Q64K128 | 256 | 1 | ALIAS | **86016** | 256 | 221 | 0 | yes (alias-only experiment) |
| 21 | Q64K128 | 256 | 2 | ALIAS | **86016** | 256 | 128 | 248 | yes (slower) |
| 22 | Q32K128 | 256 | 1 | ALIAS | **67584** | 128 | 225 | 0 | yes |
| 23 | Q64K128 | 256 | 1 | DRAIN | 103424 | 256 | — | — | no (smem) |
| 24 | Q32K128 | 256 | 1 | DRAIN | 76288 | 128 | 225 | 0 | yes |
| 25 | Q32K128 | 256 | 1 | REVDRAIN | 76288 | 128 | 225 | 0 | yes — **N1024 best** |
| 26 | Q64K128 | 256 | 1 | REVALIAS | **86016** | 256 | 221 | 0 | yes — **best overall** |

The alias union reduces the multistage scratch: Q64K128 103424 → **86016**
(now fits the 101376 opt-in budget; cfg23 drain-only control confirms the union
is what shrinks it), Q32K128 76288 → **67584**. `static_assert(!kPreloadV)`
forbids half/preload instantiation (FP32 only); `preload_v=false` recorded.
Larger tiles Q128K64 fit but are slower than the 32Q/64Q configs.

## Launch-order-reversal derivative

- Base: installed `kernel_forward.h`, SHA-256
  `fcbfa07f0bc48e239e4d8c45025d5d9e65ff017d00e9dd667918e03172637650`.
- Reversed: `vendor/kernel_forward_reversed.h`, SHA-256
  `c82a913cdecf37a0039f3b92ed35a098f9c079b8c33529acdacd34f178f6945c`.
- Diff (nothing else): namespace → `PyTorchMemEffAttentionRev`; 4×
  `query_start = blockIdx.x*kQueriesPerBlock` →
  `(gridDim.x-1-blockIdx.x)*kQueriesPerBlock` (lines 204/646/855/890).

## Alias + async-drain derivative

- Alias: `vendor/kernel_forward_alias.h`, SHA-256
  `d229a7831b4b4481eea8c0b6a74da1e5b72b85d82771996284d4dd5f18825958`.
- Drain-only control: `vendor/kernel_forward_drain.h`, SHA-256
  `3cab257986a724098f3d06cde80c1d1d6b74a53cb929a582f790b003f257ce73`.
- Reversed+drain: `vendor/kernel_forward_rev_drain.h`, SHA-256
  `4b4be8ce6bf6838b54d974192300297011533e34eecf33f4d8ee500649039162` (cfg25).
- Reversed+alias: `vendor/kernel_forward_rev_alias.h`, SHA-256
  `0a40d9c57b6a1db31a18d90c39f1b9ad101769898998a07e4d8cc5593c031107` (cfg26).
- Exact diffs (verified counts): namespace rename; the single `mm1; epilogue;`
  pair in `SharedStorageEpilogueInLoop::SharedStorageAfterMM0` wrapped in a
  union; a `static_assert(!kPreloadV)` after `kPreloadV`; two `cp_async_fence();
  cp_async_wait<0>();` insertions before the existing `__syncthreads` after
  `mma(...)` and after `mma_pv(...)`.
- License/copyright preserved (vendor LICENSE files) and full headers retained
  for reproducibility.

## Compute-sanitizer (`-lineinfo`, `--error-exitcode 255`)

- Original / hint (cfg1, cfg4): racecheck 0 errors, **8 warnings**, localized to
  `cutlass::arch::cp_async_zfill` at `memory_sm80.h:378` — the CUTLASS
  multistage `cp.async` pipeline.
- **Drain-only control (cfg24) and alias (cfg20): racecheck 0 errors,
  0 warnings (0 hazards).** Inserting `cp_async_fence(); cp_async_wait<0>();`
  before the existing syncthreads eliminates the warnings, confirming their
  source is the outstanding `cp.async` writes before scratch reuse/syncthreads.
  They are not merely "benign" — they are removed by draining.
- memcheck cfg4, cfg20, cfg26 (winner): **0 errors**. Unsupported configs (0,10,11,23)
  are rejected by the total (static+dynamic) shared-budget guard *before* any
  `cudaFuncSetAttribute`, so no invalid API call is issued.
- Selected winner cfg26 @ N257 D256: racecheck **0 hazards (0 errors, 0
  warnings)** and memcheck **0 errors**, both `--error-exitcode 255` → exit 0.

Logs: `results/racecheck-cfg1.log`, `racecheck-cfg4.log`, `racecheck-cfg24.log`,
`racecheck-cfg20.log`, `racecheck-cfg26.log`, `memcheck-cfg4.log`,
`memcheck-cfg20.log`, `memcheck-cfg26.log`.

## Adversarial numerical validation

Separate runner `run_adversarial.py` (uses build_runtime). Applies the same
adversarial patterns as the Metal worker — `zeros`, `uniform_scores`,
`constant_values`, `large_tied_scores` (Q=K=100 huge tied logits),
`peaked_scores` (q/k×16), `causal_impulse` (one nonzero value key in the corrected final suite) — at N=257,
D {192,256}, for configs 20, 25, 26, vs an independent float64 stable
causal-softmax reference at strict atol 1e-4 / rtol 1e-3. **36/36 pass**;
`large_tied_scores` stays stable (no NaN/inf), `peaked_scores` has the largest
normalized error (~0.21–0.23) but max_abs only ~3–4e-5 (within atol). All raw
results saved: `results/cuda-adversarial.json`.

## Public API (`build_runtime`)

`run_candidate.build_runtime(k192=False) -> (Runner, modules_dict)`:
- Builds the default modules `normal` / `reverse` / `alias` / `revderiv`
  (`attention_kernel_forward_standalone.cu`, `attention_reversed.cu`,
  `attention_alias.cu`, `attention_revderiv.cu`), each a separate single-TU
  extension.
- `modules_dict` maps module names → actual module objects; each exposes
  `__file__` for binary-hash binding (recorded in `meta.module_files`).
- `Runner.infos` is the complete immutable catalogue keyed by **selectable
  (effective) id**; every entry carries normalized `id`, `base_config_id`
  (the id used inside a derivative module) and `module`. This keeps the
  immutable catalogue (used by the Lean worker / frontend) consistent —
  effective ids never collide with a derivative module's internal base ids.
- `main()` and `run_sanitize.py` both use this API (no separate dispatch lists).
- Optional K192 (ids 8/9) is excluded from `ALL_CONFIGS` by default; gated
  behind `--k192` and reported template-unsupported.

## Attribution & provenance

- **Arithmetic / kernel**: Meta Platforms BSD-3-Clause PyTorch mem_eff_attention
  (installed headers, unmodified); derivatives are mechanically generated copies
  with exact diffs + hashes above.
- **CUTLASS**: NVIDIA BSD-3-Clause, pinned `e05f953a…` matching torch
  `2.14.0+cu130` (`git_version 08187d9e…`).
- **Sources (local, reproducible)**: `benchmarks/attention/cuda_candidate/`
  (`.cu` sources, `run_candidate.py`, `run_sanitize.py`, `vendor/` with
  LICENSE files). Remote: `/home/cake/.cache/veritac-attention-run/cuda_candidate/`
  (md5-verified sync). Device info in each result JSON (`12.1`, `sm_count 48`,
  opt-in `101376`).
- Inputs use the shared seed contract (seed 1234) and per-case SHA-256 hashes.

## How to reproduce

```bash
# remote: venv PATH + CUDA_HOME, MAX_JOBS=2, TORCH_CUDA_ARCH_LIST=12.1
python run_candidate.py --mode numeric --configs 1,2,3,4,5,6,7,12,13,14,15,16,17,20,21,22,24,25,26 \
    --out results/full-numeric.json
python run_candidate.py --mode sweep --configs 25,26 --out results/revderiv-sweep.json
# adversarial (36/36 pass expected):
python run_adversarial.py --out results/cuda-adversarial.json --configs 20,25,26 --dims 192,256 --seq 257
# sanitizers on selected winner:
compute-sanitizer --tool racecheck --error-exitcode 255 python run_sanitize.py --config 26 --seq 257 --dim 256
compute-sanitizer --tool memcheck  --error-exitcode 255 python run_sanitize.py --config 26 --seq 257 --dim 256
```

Controller final validation: `cuda-final-adversarial.json` passes36/36 with seed777; `controller-final-memcheck-cfg25.log`, `controller-final-racecheck-cfg25.log`, and the corresponding cfg26 logs report zero errors/warnings at N1024D192 and N2048D256. Earlier `cuda-adversarial.json` used uniform scores for the mislabeled impulse and is superseded by the corrected final suite.
