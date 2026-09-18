# Adversarial review — restricted Tenstorrent protocol native checker

Scope: `specializations/tenstorrent_protocol/protocol.py`, `adapter.py` (+ `__main__.py`, `search.py` skim) against `.lake/overnight-2026-09-18/CONTRACT.md` (frozen interface v1). Read-only review; scripts in `review/`. Verdict up front: **no soundness bug found** — the checker neither accepts unsafe schedules nor misclassifies interleaving-sensitive outcomes in ~7,600 differential/trial probes. Findings below are robustness/documentation issues plus one redundancy, with the frozen-contract simplifications separated out.

## Method

1. **Independent reference model-checker** (`review/diff_fuzz.py`): recursive DFS, frozenset pending queues keyed by issue sequence numbers, int phases — deliberately different state encoding/canonicalization from `protocol.check` (BFS, sorted tuples). Contract rules re-derived from CONTRACT.md, not from protocol.py.
2. **Random fuzz** (`diff_fuzz.py`): 3,000 fully random programs (1–3 pages/slots, ≤8 ops): 0 status mismatches.
3. **Accepted-region fuzz** (`diff_fuzz3.py`): 4,000 structured near-well-formed programs (interleaved correct lifecycles + mutations): 0 mismatches; 1,444 accepted programs cross-checked — `check`=accepted ⟹ `simulate`=accepted ⟹ `replay_concrete` bit-match ⟹ `adapter.evaluate` objective/bytes consistent.
4. **Targeted adversarial shapes**: sequential double-write to same dst (different origins → `wrong-output`; same origin → `accepted`), read-while-read-pending → unsafe, release with pending write → unsafe, multi-write same slot distinct dsts → accepted, wait-on-empty-queue advances, premature publish (assumes early DMA) → unsafe, premature re-reserve → unsafe. All correct.
5. **Oracle sensitivity** (`sensitivity.py`): injected removal of the write "no pending write for destination" precondition; differential harness catches it (mutant returns `incomplete`, reference `unsafe`). The oracle is not blind.

## Findings

**F1 (redundant precondition, not a bug).** `publish`'s "no pending read for slot" check (protocol.py:417) is implied by "value initialized" + read-clears-value: a pending read on a slot ⟹ its value is None (read clears at issue, only completion sets it). A mutant deleting the pending-read check is semantically equivalent and correctly so. Harmless; worth knowing so the Lean worker doesn't treat it as an independent axiom.

**F2 (dead code / expectation setting).** `STATUS_DEADLOCK` is unreachable in `check`. A no-transition non-terminal state cannot exist: blocked waits always have completions enabled in the same state, non-wait ops either issue or return unsafe, and pc==end with pending transfers always admits completions. This is *consistent* with the contract (stuttering excluded, line 24), but anything consuming native verdicts (search.py status handling, prompts, reports) will never see `deadlock`. Defensive dead code, not a defect — flag so the deadlock leg of the taxonomy is understood as Lean-side only.

**F3 (operational, controller-owned file).** `adapter.certify` (adapter.py:59) raises `ImportError` until `certificate.py` lands (documented deferral, STATUS.txt). Two concrete consequences: (a) `search.run` crashes at the certification stage *after* all model rounds are spent, and since `report.json` is dumped only at the end (search.py:153), the aggregated report is lost (per-round receipts survive); (b) `__main__.py:60` catches `(ValueError, TypeError, OSError, RuntimeError, KeyError)` — `ImportError` is not in the tuple, so CLI `certify` exits with a raw traceback instead of the JSON error envelope. Suggest returning a structured `{'accepted': False, reason: 'certificate module absent'}` and/or catching `ImportError`/Exception in the CLI.

**F4 (docstring inaccuracy).** `simulate` (protocol.py:687-689) claims pending transfers complete "in FIFO order"; the actual canonical order is sorted-tuple order (`(slot,src)` / `(slot,dst,origin)`), not issue order. The contract explicitly says pending queues are not FIFO; simulate is one arbitrary schedule either way, but the docstring asserts a scheduling property it doesn't implement. Could mislead someone reading simulate as a reference timing model.

**F5 (diagnostic-only).** `rebuild` (protocol.py:533) appends the offending final step *then* truncates to the first `MAX_TRACE_STEPS=1000` steps, so for traces deeper than 1000 the "invalid"/terminal step is replaced by `trace_truncated`. Unreachable via the adapter (≤128 ops ⇒ ≤~256 steps/path) but reachable via direct `protocol.check` with large programs (up to 65,536 ops). Counterexample quality only; verdicts unaffected.

**F6 (analyzed, no action).** The static range pre-check (protocol.py:513-522) returns `unsafe` for out-of-range slot/src/dst even at a pc that a dynamic checker might deem unreachable. Analysis: every pc is reachable unless an earlier op is invalid in *all* reachable states — which is already `unsafe` — so the status always agrees with a purely dynamic checker; only the reported pc/message can differ. Noted for the Lean worker aligning error provenance.

**F7 (verified sound).** Checked specifically and found correct: pending-read⟹RESERVED and pending-write⟹ACQUIRED invariants make the completion handlers' phase-blindness safe; write's dst-exclusivity + release's no-pending rule enforce "slot not reused until all its writes complete"; duplicate transfer entries in pending tuples are impossible (preconditions), so `t != transfer` removal is unambiguous; budget is honest (returned only after `max_states` states fully explored with no verdict); `replay_concrete` payloads cannot collide across pages (first byte `(7·page) mod 256`, 7 invertible mod 256), so the byte-level check cannot false-pass on page confusion; `check` acceptance ⟺ all reachable terminal states good, since unsafe/terminal classification depends only on canonical state and BFS dedup is sound.

## Frozen-contract simplifications (not bugs)

- Adapter backend caps (1–16 pages, 1–8 slots, 32–16384 bytes, 1–128 ops, max_states ≤ 200k) are strictly narrower than contract maxima; documented in STATUS.txt.
- `certificate.py` absence and advisory-not-formal authority are explicit, marked in every result (`'authority': 'native protocol filter, not a Lean certificate'`).
- No liveness claim anywhere: no unconditional silicon-liveness language found; deadlock impossibility is a semantic consequence of the closed model, not a liveness assertion.

## Reproduction

`review/diff_fuzz.py`, `review/diff_fuzz3.py`, `review/sensitivity.py` — run with `python3 <script>` from repo root (PYTHONPATH handled internally). Existing suite: 42/42 pass in `tests.test_tenstorrent_protocol`.
