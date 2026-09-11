# CUDA Attention Baseline Survey

`cuda_survey.py` is a standalone, dependency-light (PyTorch + numpy) harness
that surveys torch SDPA backends on the **causal prefill attention** operator
across a grid of sequence lengths, head dims and dtypes, checking numerics
against a high-precision numpy `float64` reference and timing each backend with CUDA
events, a wall-clock series, and CUDA graph replay.

## Operator

- Causal prefill `Q, K, V` of shape `(B, H, N, D)`, contiguous, same shape.
- `scale = D**-0.5`, no dropout.
- `out = softmax(Q·Kᵀ·scale, causal) · V`.
- Inputs from a seeded `numpy.default_rng` (normal, std 0.5), distinct Q/K/V.

## Deterministic inputs

Per case the seed is `case_seed = seed + (seq*1000003 + dim*104729 + len(dtype)*131071) mod 2^31` (the same contract as the Metal worker). The **input RNG** is `default_rng(case_seed)` and is never used for backend order. Q/K/V are drawn separately and cast directly to the numpy target dtype (`float64 -> float16/float32`, no extra rounding), then moved to CUDA. The reference is always computed from these **dtype-rounded** inputs. `case_seed` and SHA-256 hashes of the contiguous rounded Q/K/V bytes are recorded for cross-machine confirmation.

## Reference

The reference is computed once per case in numpy `float64`:

1. `scores = Q64 @ K64ᵀ * scale`
2. add causal mask (`-inf` above the diagonal, `j > i`)
3. stable softmax (subtract row max, exp, normalize)
4. `out = attn @ V64`

## Backends & status

`auto` (torch default selection) plus forced `cudnn`, `flash`, `efficient`,
`math` via `torch.nn.attention.sdpa_kernel`. A forced backend that cannot run
reports an error (never silently falls back). A `--backends` comma list selects
a subset for focused reruns.

Every backend result has a `status`:

- `ok` — output computed and within tolerance.
- `unsupported` — backend could not run (`NotImplementedError`, "no available
  kernel", etc.).
- `numeric_failed` — output computed but out of tolerance / non-finite.
- `failed` — other error.

A numerical failure is **never** labeled `ok`. Timing is still recorded for
diagnostics when a backend runs, and warnings are captured on the success and
exception paths. Everything runs under `torch.inference_mode()`.

## Numerical comparison

Per output value:

- finite checks (output and reference)
- `max_abs_error`
- normalized max error: `max(|err| / (atol + rtol*|ref|))`
- RMS error

Frozen tolerances:

| dtype    | atol   | rtol   |
|----------|--------|--------|
| float32  | 1e-4   | 1e-3   |
| float16  | 2e-3   | 2e-2   |

## Timing

The `sdpa_kernel` context is held open for the whole backend measurement so the
timed callable is the raw `F.scaled_dot_product_attention` call (no per-call
context entry). Three timing series are reported per backend:

1. `cuda_event_timing` — CUDA events over `--inner` fresh output computations
   per sample, divided by `--inner`; `--samples` raw samples.
2. `wall_timing` — synchronized wall-clock, one call per sample.
3. `graph_timing` — CUDA graph replay: `inner` fresh SDPA calls are captured
   into one graph (warmed on a side stream first, outputs kept alive), then
   warmup replays and event-timed `graph.replay()` divided by `inner`. This
   removes most CPU enqueue gaps and reflects GPU throughput, **not** a raw
   per-kernel duration.

Each reports median, p10, p90 and all samples. A graph capture error is
reported separately and never replaces a valid eager result. Dispatch/operator
names are profiled once per backend (CPU activities) **outside** the timed
region, under the same forced backend context. Timing excludes input transfer,
reference computation and the profiler; it includes the full SDPA op.

## CLI

```
python3 benchmarks/attention/cuda_survey.py \
  --output cuda_survey.json \
  --seqs 256,1024 --dims 64,128,192,256 \
  --dtypes float32,float16 --backends auto,cudnn,flash,efficient,math \
  --heads 8 --batch 1 --warmup 10 --samples 30 --inner 5
```

Progress is printed (flushed). Output is written incrementally after each
backend result (at minimum after each case) with `allow_nan=False` (non-finite
floats sanitized to `null`). Backend order is rotated reproducibly per case
using a separate RNG seeded `case_seed + 17`, and the order is recorded.

## Caveats

- Fixed timing caveats are recorded in the JSON (`timing_caveats`).
- No fabricated timings: a backend that fails to run is not given a number.
- TF32 is disabled for float32 and the settings recorded (guarded getters;
  absent attrs recorded as `null`).
- The memory estimate includes the numpy float64 reference intermediates
  (`batch*heads*N*N*8*6`) plus the linear tensors; `--mem-cap-bytes`
  (default 20 GiB) refuses absurd shapes.
