# Proposal: Maximal weight-stationary B reuse (M48, N16, K32, DIM16)

## Transformation

Baseline is output-row-tile-major: for each of the 3 row tiles it re-streams both
B tiles (k=0, k=1), giving **6 B DMA loads** (3 row tiles x 2 K steps).

This candidate restructures to K-step-major (weight-stationary reuse):

1. **K step 0**: `mvin` B tile k=0 once into spad rows 16-31, then run all 3
   output row tiles against it. The B tile stays resident in the scratchpad for
   the whole run, so each row tile just re-issues `preload(bd=16, out)` +
   `compute(false, a, GARBAGE)`; the spad->array weight transfer is not a B DMA
   load. Only the A tile cycles through spad rows 0-15.
2. **K step 1**: `mvin` B tile k=1 once (overwrites spad rows 16-31), then all
   3 row tiles again, with `out_addr = 0xE0000000 + row` so partials from
   step 0 (written via `0xA0000000 + row`, overwrite) are accumulated in place.
3. Three `mvout`s with `buf_offset = i*16*N` and `acc_addr = 0xE0000000 + 16*i`,
   then the final `fence`.

Because each B tile's reuse run completes before the other B tile is mvin'd,
the retained-weight path (`preload(GARBAGE)` + `compute(true, ...)`) is never
needed; every compute follows a `preload(bd=16)` and uses `accumulated: false`,
exactly matching the baseline command patterns. No new instruction kinds.

## Expected B load count

- Baseline: 6 `mvin` B commands.
- This proposal: **2** `mvin` B commands (one per K-step B tile) - the theoretical
  minimum, since there are exactly 2 distinct B tiles.

Plan, capacities, and input domain are UNCHANGED from baseline.json.

## Resource uncertainty (honest disclosure)

- **Accumulator capacity is the known risk.** Retaining all 3 row-tile partials
  across the K steps requires 3 x 16 = 48 accumulator rows, but only 32 are
  configured. Row tile 2 targets `out_addr` rows 32-47 (`0xA0000000+32`,
  `0xE0000000+32`, mvout `0xE0000000+32`), which exceeds the 32-row accumulator.
  I am submitting this maximal candidate anyway for Lean to decide, per
  instructions.
- **Likely repair if Lean rejects it**: a 4-B-load fallback that keeps only 2
  partials resident at a time - process row tiles {0,1} fully (B k=0, B k=1,
  mvout both, freeing acc rows 0-31), then row tile 2 reusing acc rows 0-15.
  That needs 4 `mvin` B (B0, B1, B0, B1 runs) but stays within 32 acc rows and
  32 spad rows (A tile + one B tile at a time).
- Minor: I assume `out_addr` low bits select the accumulator row start and that
  the 0xA/0xE flag bit controls overwrite-vs-accumulate, as in the baseline.

Status: PENDING - awaiting real Lean checker feedback before any repair.
