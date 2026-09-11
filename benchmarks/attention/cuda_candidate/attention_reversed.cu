// CUDA attention candidate: launch-order-reversed derivative.
//
// This TU instantiates a DERIVED copy of the installed PyTorch mem_eff
// AttentionKernel<float, cutlass::arch::Sm80, ...> where the independent
// query-block launch order is reversed: every
//   query_start = blockIdx.x * kQueriesPerBlock
// is replaced by
//   query_start = (gridDim.x - 1 - blockIdx.x) * kQueriesPerBlock
// (all four occurrences, so pointer indexing, output/LSE indexing and the
// causal/window mask all stay consistent for the query rows a block owns).
//
// The derivative lives in a DISTINCT namespace (PyTorchMemEffAttentionRev)
// and is a verbatim copy of the installed header otherwise. Arithmetic,
// masks, tile shapes and template block-count coverage are unchanged: the
// grid is still ceil_div(num_queries, kQueriesPerBlock) wide, so the same set
// of query blocks is produced — only the mapping from blockIdx.x to query
// rows (and therefore scheduling order) is a permutation. This can balance the
// imbalanced causal work across the wave.
//
// Exact patch + hashes recorded in docs/cuda_attention_specialization.md.
// Attribution: Meta Platforms BSD-3-Clause PyTorch mem_eff_attention (derived
// copy), NVIDIA CUTLASS BSD-3-Clause (pinned submodule, see main module).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <cuda_runtime.h>

#include "vendor/kernel_forward_reversed.h"

#include <cstdint>
#include <string>

namespace veritac_rev {

using AK1 = PyTorchMemEffAttentionRev::AttentionKernel<
    float, cutlass::arch::Sm80, true, 32, 128, 256, false, false>;
using AK2 = PyTorchMemEffAttentionRev::AttentionKernel<
    float, cutlass::arch::Sm80, true, 64, 64, 256, false, false>;

template <typename AK, int MinBlocks>
__global__ void __launch_bounds__(AK::kNumThreads, MinBlocks)
rev_attn_kernel(typename AK::Params p) {
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

  static void launch(const typename AK::Params& p, cudaStream_t stream) {
    auto grid = p.getBlocksGrid();
    dim3 threads((unsigned)AK::kWarpSize, (unsigned)AK::kNumWarpsPerBlock, 1);
    rev_attn_kernel<AK, MinBlocks><<<grid, threads, smem_bytes, stream>>>(p);
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

// Reversed configs reuse the main module's ids for the same tile shapes.
void cfg_params(int id, int& Q, int& K, int& MB) {
  switch (id) {
    case 1: Q = 32; K = 128; MB = 3; break;
    case 3: Q = 64; K = 64;  MB = 3; break;
    case 4: Q = 32; K = 128; MB = 1; break;
    case 5: Q = 64; K = 64;  MB = 1; break;
    default: Q = K = MB = 0; break;
  }
}

struct CfgInfo {
  int64_t smem_bytes, num_threads, num_warps, queries_per_block, keys_per_block;
  int64_t max_k, min_blocks_hint, installed_min_blocks;
  int64_t num_regs, local_size_bytes, static_shared_bytes, total_shared_bytes;
  int64_t kernel_max_threads, device_max_optin_shared, device_max_threads;
  bool single_value_iteration, keep_output_in_rf, supported;
  std::string unsupported_reason;
};

template <typename AK, int MinBlocks>
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
  c.supported = true;

  int maxOptin = 0, maxThreads = 0;
  cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, 0);
  c.device_max_optin_shared = maxOptin;
  c.device_max_threads = maxThreads;
  if (CFG::smem_bytes > maxOptin) {
    c.supported = false;
    c.total_shared_bytes = CFG::smem_bytes;
    c.unsupported_reason = "dynamic shared " + std::to_string(CFG::smem_bytes) +
                           " > device opt-in max " + std::to_string(maxOptin);
    return c;
  }
  if (CFG::num_threads > maxThreads) {
    c.supported = false;
    c.total_shared_bytes = CFG::smem_bytes;
    c.unsupported_reason += " threads " + std::to_string(CFG::num_threads) +
                            " > device max " + std::to_string(maxThreads);
    return c;
  }

  cudaFuncAttributes attr;
  cudaError_t e = cudaFuncGetAttributes(&attr,
      (const void*)rev_attn_kernel<AK, MinBlocks>);
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

  if (CFG::smem_bytes > 48 * 1024) {
    e = cudaFuncSetAttribute((void*)rev_attn_kernel<AK, MinBlocks>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             CFG::smem_bytes);
    if (e != cudaSuccess) {
      c.supported = false;
      c.unsupported_reason = std::string("cudaFuncSetAttribute: ") +
                             cudaGetErrorString(e);
      return c;
    }
  }
  if (CFG::num_threads > (int64_t)attr.maxThreadsPerBlock) {
    c.supported = false;
    c.unsupported_reason += " threads " + std::to_string(CFG::num_threads) +
                            " > kernel max " + std::to_string(attr.maxThreadsPerBlock);
  }
  return c;
}

CfgInfo prep(int id) {
#define PREP(AK, MB) return prep_impl<AK, MB>();
  switch (id) {
    case 1: PREP(AK1, 3);
    case 3: PREP(AK2, 3);
    case 4: PREP(AK1, 1);
    case 5: PREP(AK2, 1);
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
    case 1: LAUNCH(AK1, 3); break;
    case 3: LAUNCH(AK2, 3); break;
    case 4: LAUNCH(AK1, 1); break;
    case 5: LAUNCH(AK2, 1); break;
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
    case 1: VALIDATE(AK1, 3);
    case 3: VALIDATE(AK2, 3);
    case 4: VALIDATE(AK1, 1);
    case 5: VALIDATE(AK2, 1);
    default: return std::string("unknown config id ") + std::to_string(id);
  }
#undef VALIDATE
}

}  // namespace veritac_rev

using namespace veritac_rev;

void launch_attn_rev(int64_t id, int64_t q, int64_t k, int64_t v, int64_t o,
                     int64_t N, int64_t H, int64_t D, int64_t causal,
                     double scale, int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  launch_dispatch((int)id, reinterpret_cast<const float*>(q),
                  reinterpret_cast<const float*>(k),
                  reinterpret_cast<const float*>(v), reinterpret_cast<float*>(o),
                  (int)N, (int)H, (int)D, (int)causal, (float)scale, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

py::dict validate_rev(int64_t id, int64_t q, int64_t k, int64_t v, int64_t o,
                      int64_t N, int64_t H, int64_t D, int64_t causal,
                      double scale) {
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

py::dict config_info_rev(int64_t id) {
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
  m.def("launch_attn_rev", &launch_attn_rev);
  m.def("config_info_rev", &config_info_rev);
  m.def("validate_rev", &validate_rev);
}
