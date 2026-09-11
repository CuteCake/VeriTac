# Metal attention baseline survey (MLX vs MPSGraph)

Compares Apple baselines for causal prefill attention on the local Metal
device, validated against an independent numpy float64 stable-causal reference
computed once per case from the dtype-rounded inputs.

| Backend | Implementation | Timing |
|---|---|---|
| `mlx_fast` | MLX `mx.fast.scaled_dot_product_attention(mask="causal")`, default dispatch | synchronized API latency (no GPU timestamps exposed) |
| `mlx_fused` | same op with `force_fused=True` | synchronized API latency |
| `mlx_expr` | unfused MLX expression baseline: `QK^T -> float32 softmax -> PV` | synchronized API latency |
| `mpsgraph` | MPSGraph `scaledDotProductAttention` via a compiled Swift helper | GPUStartTime/GPUEndTime deltas + synchronized API latency |

All operators: causal square prefill `[B,H,N,D]`, scale `D**-0.5`, no dropout,
contiguous. FP32 and matched FP16 tracks. Frozen tolerances: fp32
`atol=1e-4, rtol=1e-3`; fp16 `atol=2e-3, rtol=2e-2`.

`mlx_expr` arithmetic policy: QK^T and PV matmuls run in the input dtype; the
softmax runs in float32 (max-subtraction, stable); probabilities are cast back
to the input dtype before the PV matmul. The causal mask is pre-built and
materialized outside the timed region, so the timed callable does no numpy
work, no input casts, and no input evaluation. It is an explicitly-labeled
expression baseline, not a vendor winner.

## Usage

```bash
# Requires mlx + numpy, and a Metal device. Run from the project root.
PYTHONPATH=. python3 benchmarks/attention/metal_survey.py \
    --output metal_survey.json \
    --build-dir /tmp/metal-survey-build \
    --backends mlx_fast,mlx_fused,mlx_expr,mpsgraph \
    --seqs 256,1024 --dims 64,128,192,256 --dtypes float32,float16 \
    --heads 8 --batch 1 --warmup 10 --samples 30 --inner 5
```

- `--backends` selects which baselines to run (default: all four). Backends
  within a case execute serially in a randomized order seeded by `case_seed+17`
  (recorded in `backend_order`); an `mx.synchronize()` drains MLX work before
  the Swift helper runs.
- The Swift helper is compiled once into `--build-dir` (outside tracked files)
  via `xcrun swiftc -O` with the `Metal`, `MetalPerformanceShaders` and
  `MetalPerformanceShadersGraph` frameworks. It is only compiled when
  `mpsgraph` is among the selected backends (`--no-mps`/MLX-only runs skip it).
- Output JSON is written incrementally after every case; each backend/case
  exception is caught and recorded as a failure rather than aborting. A case
  whose float64 reference peak exceeds `--mem-cap-bytes` is skipped and noted.
  Non-finite values are never emitted as raw JSON numbers.
- Input Q/K/V are distinct, seeded `numpy.default_rng` draws rounded directly
  to the target dtype; SHA256 hashes of the rounded bytes are recorded.

## What is measured (all units: milliseconds)

- **MPSGraph GPU time:** per invocation, `MPSCommandBuffer.gpuStartTime` /
  `gpuEndTime` after `waitUntilCompleted`, excluding input/output CPU copies
  (preallocated shared `MTLBuffer`s) and compile/warmup. If GPU timestamps are
  invalid/incomplete, `gpu_time_available` is `false` and `gpu_timing` is
  `null` (never a fabricated summary).
- **Synchronized API latency:** MLX (around `mx.eval` + `mx.synchronize`) and
  MPSGraph (timed from before command-buffer allocation through
  `waitUntilCompleted`). `samples` samples × `inner` invocations each; raw
  timings are kept in `wall_timing.samples_ms` and `gpu_timing.samples_ms`.
- A Metal command-buffer error during MPSGraph timing fails the case rather
  than reporting a bogus timing.

## Validation and status

Each backend result is checked for full-output finiteness, shape equality, and
`abs(error) <= atol + rtol*abs(ref)` element-wise. A backend that computes but
fails the numeric contract is reported `numeric_failed`, not `ok`.

## Known limitations

- **MLX GPU timestamps:** no verified Metal GPU-timestamp API is exposed for
  MLX here, so `gpu_time_available` is `false`; MLX timings are wall API
  latency only.
- **MLX kernel selection:** not queried; reported as `kernel: unknown`. For
  `mlx_fast` the default fused/unfused dispatch is source-inferred, never
  guessed as an actual kernel name.
- **Coverage:** MLX or MPSGraph may fall back or fail for some
  `(dtype, head_dim)` combinations; a failure is recorded with its error,
  never silently replaced by a different operator's timing.
- Head dims 192/256 are included to probe coverage; a timing there does not
  prove a fused kernel ran.
