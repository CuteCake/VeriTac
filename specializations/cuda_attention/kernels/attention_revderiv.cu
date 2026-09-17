// CUDA attention candidate: reversed scheduling combined with FP32
// scratch-alias / async-drain derivatives.
//
// This TU instantiates mechanically generated DERIVED copies of the installed
// PyTorch mem_eff AttentionKernel<float, cutlass::arch::Sm80, ...> that combine
// the launch-order reversal (all four query_start sites reversed) with:
//   * PyTorchMemEffAttentionRevDrain (vendor/kernel_forward_rev_drain.h):
//     reversal + cp_async drains, unchanged storage.
//   * PyTorchMemEffAttentionRevAlias (vendor/kernel_forward_rev_alias.h):
//     reversal + alias union(mm1/epilogue) + static_assert(!kPreloadV) +
//     cp_async drains.
// Effective configs 25/26; arithmetic/masks/tile-shape/block-count unchanged.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <cuda_runtime.h>

#include "vendor/kernel_forward_rev_drain.h"
#include "vendor/kernel_forward_rev_alias.h"

#include <cstdint>
#include <string>

namespace veritac_revderiv {

using AK25 = PyTorchMemEffAttentionRevDrain::AttentionKernel<
    float, cutlass::arch::Sm80, true, 32, 128, 256, false, false>;
using AK26 = PyTorchMemEffAttentionRevAlias::AttentionKernel<
    float, cutlass::arch::Sm80, true, 64, 128, 256, false, false>;

template <typename AK, int MinBlocks>
__global__ void __launch_bounds__(AK::kNumThreads, MinBlocks)
revderiv_attn_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK, int MinBlocks>
struct Cfg {
  static constexpr int smem_bytes = (int)sizeof(typename AK::SharedStorage);
  static constexpr int num_threads = AK::kNumThreads;
  static constexpr int num_warps = AK::kNumWarpsPerBlock;
  static constexpr int kKeysPerBlock = AK::kKeysPerBlock;
  static constexpr int kQueriesPerBlock = AK::kQueriesPerBlock;
  static constexpr int kMaxK = AK::kMaxK;
  static constexpr int kMinBlocksPerSm = AK::kMinBlocksPerSm;
  static constexpr bool kSingleValueIteration = AK::kSingleValueIteration;
  static constexpr bool kKeepOutputInRF = AK::kKeepOutputInRF;
  static constexpr bool kPreloadV = AK::kPreloadV;

  static void launch(const typename AK::Params& p, cudaStream_t stream) {
    auto grid = p.getBlocksGrid();
    dim3 threads((unsigned)AK::kWarpSize, (unsigned)AK::kNumWarpsPerBlock, 1);
    revderiv_attn_kernel<AK, MinBlocks>
        <<<grid, threads, smem_bytes, stream>>>(p);
  }
};

template <typename AK>
typename AK::Params make_params(const float* q, const float* k, const float* v,
                                float* o, int N, int H, int D, int causal,
                                float scale) {
  typename AK::Params p;
  p.query_ptr = q;
  p.key_ptr = k;
  p.value_ptr = v;
  p.output_ptr = o;
  p.attn_bias_ptr = nullptr;
  p.seqstart_q_ptr = nullptr;
  p.seqstart_k_ptr = nullptr;
  p.seqlen_k_ptr = nullptr;
  p.causal_diagonal_offset = 0;
  p.output_accum_ptr = nullptr;
  p.logsumexp_ptr = nullptr;
  p.window_size = 0;
  p.scale = scale;
  p.head_dim = D;
  p.head_dim_value = D;
  p.num_queries = N;
  p.num_keys = N;
  p.num_keys_absolute = N;
  p.custom_mask_type = (uint8_t)(causal ? (int)AK::CausalFromTopLeft
                                        : (int)AK::NoCustomMask);
  p.q_strideM = D;
  p.k_strideM = D;
  p.v_strideM = D;
  p.bias_strideM = 0;
  p.o_strideM = H * D;
  p.q_strideH = N * D;
  p.k_strideH = N * D;
  p.v_strideH = N * D;
  p.bias_strideH = 0;
  p.q_strideB = (int64_t)H * N * D;
  p.k_strideB = (int64_t)H * N * D;
  p.v_strideB = (int64_t)H * N * D;
  p.bias_strideB = 0;
  p.num_batches = 1;
  p.num_heads = H;
  p.q_heads_per_kv = 1;
  p.use_dropout = false;
  return p;
}

struct CfgInfo {
  int64_t smem_bytes, num_threads, num_warps, queries_per_block, keys_per_block;
  int64_t max_k, min_blocks_hint, installed_min_blocks;
  int64_t num_regs, local_size_bytes, static_shared_bytes, total_shared_bytes;
  int64_t kernel_max_threads, device_max_optin_shared, device_max_threads;
  bool single_value_iteration, keep_output_in_rf, preload_v, supported;
  bool alias_scratch, async_drains, reversed;
  std::string unsupported_reason;
};

template <typename AK, int MinBlocks, bool AliasScratch>
CfgInfo prep_impl() {
  using CFG = Cfg<AK, MinBlocks>;
  CfgInfo c{};
  c.smem_bytes = CFG::smem_bytes;
  c.num_threads = CFG::num_threads;
  c.num_warps = CFG::num_warps;
  c.queries_per_block = CFG::kQueriesPerBlock;
  c.keys_per_block = CFG::kKeysPerBlock;
  c.max_k = CFG::kMaxK;
  c.min_blocks_hint = MinBlocks;
  c.installed_min_blocks = CFG::kMinBlocksPerSm;
  c.single_value_iteration = CFG::kSingleValueIteration;
  c.keep_output_in_rf = CFG::kKeepOutputInRF;
  c.preload_v = CFG::kPreloadV;
  c.alias_scratch = AliasScratch;
  c.async_drains = true;
  c.reversed = true;
  c.supported = true;

  int maxOptin = 0, maxThreads = 0;
  cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, 0);
  c.device_max_optin_shared = maxOptin;
  c.device_max_threads = maxThreads;
  if (CFG::num_threads > maxThreads) {
    c.supported = false;
    c.total_shared_bytes = CFG::smem_bytes;
    c.unsupported_reason += " threads " + std::to_string(CFG::num_threads) +
                            " > device max " + std::to_string(maxThreads);
    return c;
  }

  cudaFuncAttributes attr;
  cudaError_t e = cudaFuncGetAttributes(&attr,
      (const void*)revderiv_attn_kernel<AK, MinBlocks>);
  if (e != cudaSuccess) {
    c.supported = false;
    c.total_shared_bytes = CFG::smem_bytes;
    c.unsupported_reason = std::string("cudaFuncGetAttributes: ") +
                           cudaGetErrorString(e);
    return c;
  }
  c.num_regs = attr.numRegs;
  c.local_size_bytes = attr.localSizeBytes;
  c.static_shared_bytes = attr.sharedSizeBytes;
  c.kernel_max_threads = attr.maxThreadsPerBlock;
  c.total_shared_bytes = (int64_t)attr.sharedSizeBytes + CFG::smem_bytes;

  if (c.total_shared_bytes > maxOptin) {
    c.supported = false;
    c.unsupported_reason = "total shared " + std::to_string(c.total_shared_bytes) +
                           " > device opt-in max " + std::to_string(maxOptin);
    return c;
  }
  if (CFG::num_threads > (int64_t)attr.maxThreadsPerBlock) {
    c.supported = false;
    c.unsupported_reason += " threads " + std::to_string(CFG::num_threads) +
                            " > kernel max " + std::to_string(attr.maxThreadsPerBlock);
    return c;
  }
  if (CFG::smem_bytes > 48 * 1024) {
    e = cudaFuncSetAttribute((void*)revderiv_attn_kernel<AK, MinBlocks>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             CFG::smem_bytes);
    if (e != cudaSuccess) {
      c.supported = false;
      c.unsupported_reason = std::string("cudaFuncSetAttribute: ") +
                             cudaGetErrorString(e);
      return c;
    }
  }
  return c;
}

CfgInfo prep(int id) {
#define PREP(AK, MB, AS) return prep_impl<AK, MB, AS>();
  switch (id) {
    case 25: PREP(AK25, 1, false);  // rev + drain, Q32K128 min1
    case 26: PREP(AK26, 1, true);   // rev + alias, Q64K128 min1
    default: TORCH_CHECK(false, "unknown config id ", id);
  }
#undef PREP
}

void launch_dispatch(int id, const float* q, const float* k, const float* v,
                     float* o, int N, int H, int D, int causal, float scale,
                     cudaStream_t stream) {
#define LAUNCH(AK, MB)                                                        \
  do {                                                                        \
    using CFG = Cfg<AK, MB>;                                                  \
    auto p = make_params<AK>(q, k, v, o, N, H, D, causal, scale);             \
    CFG::launch(p, stream);                                                   \
  } while (0)

  switch (id) {
    case 25: LAUNCH(AK25, 1); break;
    case 26: LAUNCH(AK26, 1); break;
    default: TORCH_CHECK(false, "unknown config id ", id);
  }
#undef LAUNCH
}

std::string validate_dispatch(int id, const float* q, const float* k,
                              const float* v, float* o, int N, int H, int D,
                              int causal, float scale) {
#define VALIDATE(AK, MB)                                                       \
  do {                                                                         \
    auto p = make_params<AK>(q, k, v, o, N, H, D, causal, scale);              \
    try {                                                                      \
      AK::check_supported(p);                                                  \
      return std::string("ok");                                                \
    } catch (const c10::Error& ex) {                                           \
      return std::string("check_supported: ") + ex.what();                     \
    }                                                                          \
  } while (0)

  switch (id) {
    case 25: VALIDATE(AK25, 1);
    case 26: VALIDATE(AK26, 1);
    default: return std::string("unknown config id ") + std::to_string(id);
  }
#undef VALIDATE
}

}  // namespace veritac_revderiv

using namespace veritac_revderiv;

void launch_attn_revderiv(int64_t id, int64_t q, int64_t k, int64_t v,
                          int64_t o, int64_t N, int64_t H, int64_t D,
                          int64_t causal, double scale, int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  launch_dispatch((int)id, reinterpret_cast<const float*>(q),
                  reinterpret_cast<const float*>(k),
                  reinterpret_cast<const float*>(v), reinterpret_cast<float*>(o),
                  (int)N, (int)H, (int)D, (int)causal, (float)scale, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

py::dict validate_revderiv(int64_t id, int64_t q, int64_t k, int64_t v,
                           int64_t o, int64_t N, int64_t H, int64_t D,
                           int64_t causal, double scale) {
  std::string r = validate_dispatch(
      (int)id, reinterpret_cast<const float*>(q),
      reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v),
      reinterpret_cast<float*>(o), (int)N, (int)H, (int)D, (int)causal,
      (float)scale);
  py::dict d;
  d["ok"] = (r == "ok");
  d["detail"] = r;
  return d;
}

py::dict config_info_revderiv(int64_t id) {
  auto c = prep((int)id);
  py::dict d;
  d["id"] = id;
  d["smem_bytes"] = c.smem_bytes;
  d["num_threads"] = c.num_threads;
  d["num_warps"] = c.num_warps;
  d["queries_per_block"] = c.queries_per_block;
  d["keys_per_block"] = c.keys_per_block;
  d["max_k"] = c.max_k;
  d["min_blocks_hint"] = c.min_blocks_hint;
  d["installed_min_blocks"] = c.installed_min_blocks;
  d["single_value_iteration"] = c.single_value_iteration;
  d["keep_output_in_rf"] = c.keep_output_in_rf;
  d["preload_v"] = c.preload_v;
  d["alias_scratch"] = c.alias_scratch;
  d["async_drains"] = c.async_drains;
  d["reversed"] = c.reversed;
  d["num_regs"] = c.num_regs;
  d["local_size_bytes"] = c.local_size_bytes;
  d["static_shared_bytes"] = c.static_shared_bytes;
  d["total_shared_bytes"] = c.total_shared_bytes;
  d["kernel_max_threads"] = c.kernel_max_threads;
  d["device_max_optin_shared"] = c.device_max_optin_shared;
  d["device_max_threads"] = c.device_max_threads;
  d["supported"] = c.supported;
  d["unsupported_reason"] = c.unsupported_reason;
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_attn_revderiv", &launch_attn_revderiv);
  m.def("config_info_revderiv", &config_info_revderiv);
  m.def("validate_revderiv", &validate_revderiv);
}
