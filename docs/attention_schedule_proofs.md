# Attention scheduling proofs (Lean)

This documents the new structural Lean proofs that explain the creative CUDA
scheduling / scratch-lifetime transformations used by the attention candidate
kernels. These are **abstract proofs only**; they do not prove any compiler or
GPU behavior.

## Files

- `VeriTac/Attention/Scheduling.lean` — the proofs (imported from `VeriTac.lean`).
- `tests/lean/AttentionScheduling.lean` — concrete smoke tests (no `axiom`,
  `sorry`, or `native_decide`).

Verify:

```bash
lake build veritac
lake env lean tests/lean/AttentionScheduling.lean
```

The new theorems depend only on the core Lean axioms `propext` / `Quot.sound`
(verified with `#print axioms`); `drained_implies_pending_zero_in_later_phase`
depends on no axioms.

## 1. Forward / reverse block scheduling (`Fin.rev`)

`forwardSlot n b = b` and `reverseSlot n b = Fin.rev b` (i.e. the
`query_start = (gridDim.x - 1 - blockIdx.x) * Q` substitution, modelled for `n`
logical blocks).  Proved:

- **involution** `reverseSlot_involution` (via `Fin.rev_rev`);
- **injective** `reverseSlot_injective`, **surjective** `reverseSlot_surjective`,
  **bijective / exactly-once coverage** `reverseSlot_bijective`,
  `reverse_distinct_blocks_distinct_slots`;
- **value preservation** `reverse_value_preserved`: logical block `b`'s output
  is delivered unchanged to slot `reverseSlot n b`;
- **row ownership** `reverse_keeps_rows`: rows owned by a block are unchanged.

### Explicit output address (destination)

`reverseWrite blockOutput p : Fin n × α := (Fin.rev p, blockOutput (Fin.rev p))`
models that the physical execution slot `p` computes logical block `Fin.rev p`
**and** writes its output to the destination logical block `Fin.rev p`.  Proved:

- `reverseWrite_destination`: at `p = Fin.rev b`, `reverseWrite = (b, blockOutput b)`
  — both logical destination and value are preserved;
- `reverseWrite_dest_unique`: every destination `b` is written by exactly one
  physical slot (`Fin.rev b`).

Important boundary: the **physical execution slot is NOT the output address**.
Reversing the schedule permutes which slot computes a block; the destination is
the logical block index, which is preserved.  `scheduledOutput`/`reverseWrite`
model this distinction.

## 2. Half-open scratch lifetimes, shared capacity, and draining

`ScratchPhase { lo hi size pending }` occupies `[lo, hi)` and carries a
phase-specific `pending : Nat → Nat` (outstanding async copies at each time).

- **disjointness** `no_time_live_in_both` / `no_time_live_in_both_sym`: if
  `a.hi ≤ b.lo` (or symmetrically), no time `t` is live in both;
- **capacity** `max_capacity_covers_each`, `max_capacity_le_sum`;
- **MM1-then-epilogue instantiation** `mm1_epilogue_no_overlap`,
  `mm1_epilogue_shared_capacity`.

Draining model (not a hardware claim):

- `AsyncDrained p := ∀ t, p.hi ≤ t → p.pending t = 0` — a phase's asynchronous
  copies are all drained from its end onward;
- `drained_implies_pending_zero_in_later_phase`: if `a.hi ≤ b.lo` and `a` is
  drained, then at every time `t` live in `b`, `a.pending t = 0` — a later
  phase reusing `a`'s scratch never observes a live copy from `a`;
- positive `pendingfun0` (`∀ t, pending t = 0`, drained) and negative
  `pendingfun1` (`∀ t, pending t = 1`, not drained — the guarantee cannot fire)
  examples.

This is a meaningful conditional model: the resource-aliasing argument is sound
only under `AsyncDrained`; we do not assert that `cp.async` hardware drains on
command.

## 3. Connection to the accepted CUDA row semantics

`CudaPlan.lean` proves `cuda_accepted_row_preserves` /
`cuda_configs_row_semantics_agree`: an accepted CUDA plan preserves the
per-row Real stable-softmax output, invariant under config / tile / hint
selection.  `Scheduling.lean` restates this (`accepted_row_semantics_restated`)
and shows (`reverse_delivers_block_row_output`, `reverseWrite`) that the reverse
schedule delivers each logical block's row output to its destination unchanged.
Together: reversing the query-block launch order does not change any row's
abstract Real value — it only relocates which physical slot computes that row's
block.

## Honest boundaries

- No proof that the GPU compiler executes blocks in a particular order.
- No proof that a `min_blocks_per_sm` launch hint is honored.
- No proof of `cp.async` hardware draining semantics (modelled by
  `AsyncDrained`, documented as an assumption).
- No floating-point / precision claims.
- These structural proofs, plus the compiled resource catalogue
  (`CudaPlan.lean`) and the controller's source/sanitizer validation, are the
  honest boundary of the verified claims.
