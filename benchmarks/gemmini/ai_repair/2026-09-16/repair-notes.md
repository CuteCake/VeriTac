# Repair: weight-stationary B reuse within the 32-row accumulator

## Why the proposal was rejected

The Lean checker rejected the maximal candidate at instruction execution. Per
`VeriTac/Gemmini/Engine.lean`, `accAddress` requires
`tileFits (accRow addr) accumulatorRows`, i.e. every preload/mvout address must
map to accumulator rows `< 32`. The proposal retained all 3 row-tile partials
across K, so row tile 2 targeted `0xA0000000+32` / `0xE0000000+32` (acc rows
32-47) - exactly the resource risk flagged in proposal-notes.md. The first such
`preload` returned `none`.

## The repair

Plan, capacities (scratchpad 32 rows, accumulator 32 rows), and int8->int32
domain are UNCHANGED. The weight-stationary restructuring is kept, but only 2
row-tile partials (32 acc rows) are ever resident:

- **Pass 1 (row tiles 0, 1)**: `mvin` B tile k=0 once into spad rows 16-31,
  run t0k0 -> acc rows 0-15 and t1k0 -> acc rows 16-31 against the resident B
  tile; then `mvin` B tile k=1 once and accumulate t0k1 (rows 0-15) and t1k1
  (rows 16-31) via `0xE0000000 + row`; `mvout` both tiles, freeing all acc rows.
- **Pass 2 (row tile 2)**: reuses acc rows 0-15 (tile 0's data was already
  mvout'd to C, so the overwrite with `0xA0000000+0` is safe): B k=0 load,
  t2k0, B k=1 load, t2k1, `mvout` at `buf_offset = 2*16*16 = 512`.

Final `fence` present. All commands use the baseline patterns only: every
`preload` carries a valid `bd_spad_addr = 16` and every `compute` is
`accumulated: false` with `bd_spad_addr = 4294967295`; the retained-weight path
(`preload(GARBAGE)` + `compute(true, ...)`) is not needed because each B tile
stays scratchpad-resident for the whole run that uses it. No new instruction
kinds. 30 commands total.

I verified every step offline against the `gstep`/`step` transitions in
`VeriTac/Gemmini/Engine.lean` (address/stride bounds, spad residency at each
compute, acc-row existence before accumulate, mvout coverage of all 768 C
elements); all 30 steps pass. No checker was run, per instructions.

## Expected B-load count

- Baseline: 6 `mvin` B. Rejected maximal proposal: 2 (infeasible).
- **This repair: 4 `mvin` B** (weight-state sequence B0, B1, B0, B1).
- 4 is minimal given the 32-row accumulator: each row tile needs B0 before B1,
  only 2 partials fit concurrently so at most 2 tiles can complete per B0/B1
  run pair, and 3 row tiles force the run pair to repeat once (a 3-load
  sequence B0/B1/B0 cannot finish any tile's k=1 after the second B0 run, and
  spad cannot hold A plus both B tiles simultaneously).

Status: PENDING - awaiting controller verification of repair.json.
