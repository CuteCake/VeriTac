// CUDA attention candidate: hardware-aware instantiation of the *installed*
// PyTorch mem_eff AttentionKernel<float, cutlass::arch::Sm80, ...>.
//
// We do NOT change the arithmetic of the installed kernel. For float32 on
// compute capability >= 80 the PyTorch DefaultGemmType selects
// cutlass::arch::OpMultiplyAddFastF32, i.e. CUTLASS's MmaTensorOpFastF32 which
// internally performs a 3-component (k3xTF32) emulation of F32 matmul on the
// tensor cores. This is exactly the arithmetic used by the FP32 "efficient"
// (mem_eff) vendor comparator in the baseline survey. Reproducing the same
// arithmetic is therefore not a new reduced-precision substitution; it is a
// re-instantiation of the vendor kernel with chosen tile shapes and a
// launch-bounds (min-blocks-per-SM) occupancy hint.
//
// Source of arithmetic/attribution: Meta Platforms BSD-3-Clause licensed
// PyTorch mem_eff_attention (kernel_forward.h, gemm/*, custom_mma*) and
// NVIDIA CUTLASS (BSD-3-Clause) pinned to the PyTorch submodule commit
// e05f953a5b3d38adc240df2ff928e0421c2abba3 (CUTLASS) corresponding to the
// installed torch 2.14.0+cu130 (git_version 08187d9e...). We add no single-TF32
// and no FP16 path.
//
// The installed kernel launches via
//   attention_kernel_batched_impl<AK>  __launch_bounds__(kNumThreads,
//                                                       kMinBlocksPerSm)
// where for float kMinBlocksPerSm = 12 / kNumWarpsPerBlock. For tile shapes
// whose dynamic shared memory only permits one block per SM, that launch bound
// forces a low register budget and spills. Here we reproduce the identical
// device body (p.advance_to_block() then AK::attention_kernel(p)) under a
// named __global__ wrapper whose min-blocks hint is a tunable template
// parameter, letting ptxas trade occupancy for register pressure / fewer
// spills. No arithmetic changes.
//
// Output ABI: the installed kernel's advance_to_block advances
//   output_ptr += query_start*o_strideM + head_id*head_dim_value
// i.e. physical output layout [num_queries, num_heads, head_dim_value] with
// per-query stride o_strideM. For B=1 we allocate a (N, H, D) contiguous
// buffer and set o_strideM = H*D; the caller views it as (B,H,N,D).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <cuda_runtime.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#include <cstdint>
#include <string>

namespace veritac_cuda {

using namespace PyTorchMemEffAttention;

// Named __global__ wrapper replicating the installed launch wrapper exactly
// (advance_to_block + attention_kernel), but with a tunable min-blocks hint.
template <typename AK, int MinBlocks>
__global__ void __launch_bounds__(AK::kNumThreads, MinBlocks)
veritac_attn_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

// One instantiation: tile shape + min-blocks occupancy hint.
template <int QTile, int KTile, int MD, int MinBlocks>
struct Cfg {
  using AK = AttentionKernel<float, cutlass::arch::Sm80, true, QTile, KTile, MD,
                             false, false>;
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
    veritac_attn_kernel<AK, MinBlocks>
        <<<grid, threads, smem_bytes, stream>>>(p);
  }
};

// Build Params for the operator: B=1, H heads, N queries==keys, D head dim,
// contiguous (B,H,N,D) Q/K/V; output physical (N,H,D) contiguous with
// o_strideM = H*D. causal==1 -> CausalFromTopLeft.
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
  p.o_strideM = H * D;  // physical output layout is [N, H, D]
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
  int64_t smem_bytes;
  int64_t num_threads;
  int64_t num_warps;
  int64_t queries_per_block;
  int64_t keys_per_block;
  int64_t max_k;
  int64_t min_blocks_hint;
  int64_t installed_min_blocks;
  bool single_value_iteration;
  bool keep_output_in_rf;
  int64_t num_regs;
  int64_t local_size_bytes;
  int64_t static_shared_bytes;
  int64_t total_shared_bytes;
  int64_t kernel_max_threads;
  int64_t device_max_optin_shared;
  int64_t device_max_threads;
  bool supported;
  std::string unsupported_reason;
};

template <int QTile, int KTile, int MD, int MinBlocks>
CfgInfo prep_impl() {
  using CFG = Cfg<QTile, KTile, MD, MinBlocks>;
  using AK = typename CFG::AK;
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
  c.unsupported_reason = "";

  // Device budget attributes first (read-only). Reject oversized shared or
  // thread counts BEFORE any cudaFuncSetAttribute so an invalid API call is
  // never issued for an unsupported config.
  int maxOptin = 0, maxThreads = 0;
  cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, 0);
  c.device_max_optin_shared = maxOptin;
  c.device_max_threads = maxThreads;

  if (CFG::smem_bytes > maxOptin) {
    c.supported = false;
    c.total_shared_bytes = CFG::smem_bytes;
    c.unsupported_reason =
        "dynamic shared " + std::to_string(CFG::smem_bytes) +
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

  // Read-only kernel attributes (registers/spills/etc).
  cudaFuncAttributes attr;
  cudaError_t e = cudaFuncGetAttributes(
      &attr, (const void*)veritac_attn_kernel<AK, MinBlocks>);
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

  // Request the dynamic shared memory size we pass at launch if it exceeds the
  // default (48 KB) so that <<<>>> allows it. Only reached for supported
  // configs (shared already verified <= opt-in max above).
  if (CFG::smem_bytes > 48 * 1024) {
    e = cudaFuncSetAttribute(
        (void*)veritac_attn_kernel<AK, MinBlocks>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CFG::smem_bytes);
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

// Config id -> (Q, K, MD, MinBlocks). ids 0-3 reproduce the installed
// launch-bound behaviour (original), ids 4-7 tune the hint to 1 or 2 blocks/SM,
// ids 10-13 are larger query-reuse tiles with hint 1/2. ids 8/9 (exact D192 /
// K192) are compiled only when VERITAC_ENABLE_K192 is defined so a
// template/static-assert failure there cannot break the others. K192 configs
// are restricted to D == 192 (kMaxK = 192 < 256).
void cfg_params(int id, int& Q, int& K, int& MD, int& MB) {
  switch (id) {
    case 0: Q = 32; K = 256; MD = 256; MB = 1; break;      // original Q32K256 (installed 12/8=1, unsupported smem)
    case 1: Q = 32; K = 128; MD = 256; MB = 3; break;      // original Q32K128
    case 2: Q = 32; K = 64;  MD = 256; MB = 6; break;      // original Q32K64
    case 3: Q = 64; K = 64;  MD = 256; MB = 3; break;      // original Q64K64
    case 4: Q = 32; K = 128; MD = 256; MB = 1; break;      // hint Q32K128 min1
    case 5: Q = 64; K = 64;  MD = 256; MB = 1; break;      // hint Q64K64 min1
    case 6: Q = 64; K = 64;  MD = 256; MB = 2; break;      // hint Q64K64 min2
    case 7: Q = 32; K = 64;  MD = 256; MB = 1; break;      // hint Q32K64 min1
    case 10: Q = 64; K = 128; MD = 256; MB = 1; break;     // Q64K128 min1
    case 11: Q = 64; K = 128; MD = 256; MB = 2; break;     // Q64K128 min2
    case 12: Q = 128; K = 64; MD = 256; MB = 1; break;     // Q128K64 min1
    case 13: Q = 128; K = 64; MD = 256; MB = 2; break;     // Q128K64 min2
#ifdef VERITAC_ENABLE_K192
    case 8: Q = 32; K = 192; MD = 192; MB = 2; break;      // exact D192/K192
    case 9: Q = 32; K = 192; MD = 192; MB = 1; break;      // exact D192/K192 min1
#endif
    default: Q = K = MD = MB = 0; break;
  }
}

bool d192_only(int id) {
#ifdef VERITAC_ENABLE_K192
  return id == 8 || id == 9;
#else
  return false;
#endif
}

CfgInfo prep(int id) {
  int Q, K, MD, MB;
  cfg_params(id, Q, K, MD, MB);
  if (MB == 0) TORCH_CHECK(false, "unknown config id ", id);
#define PREP(QT, KT, MDD, MBB) return prep_impl<QT, KT, MDD, MBB>();
  switch (id) {
    case 0: PREP(32, 256, 256, 1);
    case 1: PREP(32, 128, 256, 3);
    case 2: PREP(32, 64, 256, 6);
    case 3: PREP(64, 64, 256, 3);
    case 4: PREP(32, 128, 256, 1);
    case 5: PREP(64, 64, 256, 1);
    case 6: PREP(64, 64, 256, 2);
    case 7: PREP(32, 64, 256, 1);
    case 10: PREP(64, 128, 256, 1);
    case 11: PREP(64, 128, 256, 2);
    case 12: PREP(128, 64, 256, 1);
    case 13: PREP(128, 64, 256, 2);
#ifdef VERITAC_ENABLE_K192
    case 8: PREP(32, 192, 192, 2);
    case 9: PREP(32, 192, 192, 1);
#endif
    default: TORCH_CHECK(false, "unknown config id ", id);
  }
#undef PREP
}

void launch_dispatch(int id, const float* q, const float* k, const float* v,
                     float* o, int N, int H, int D, int causal, float scale,
                     cudaStream_t stream) {
#define LAUNCH(QT, KT, MDD, MBB)                                               \
  do {                                                                         \
    using CFG = Cfg<QT, KT, MDD, MBB>;                                         \
    auto p = make_params<typename CFG::AK>(q, k, v, o, N, H, D, causal, scale);\
    CFG::launch(p, stream);                                                    \
  } while (0)

  switch (id) {
    case 0: LAUNCH(32, 256, 256, 1); break;
    case 1: LAUNCH(32, 128, 256, 3); break;
    case 2: LAUNCH(32, 64, 256, 6); break;
    case 3: LAUNCH(64, 64, 256, 3); break;
    case 4: LAUNCH(32, 128, 256, 1); break;
    case 5: LAUNCH(64, 64, 256, 1); break;
    case 6: LAUNCH(64, 64, 256, 2); break;
    case 7: LAUNCH(32, 64, 256, 1); break;
    case 10: LAUNCH(64, 128, 256, 1); break;
    case 11: LAUNCH(64, 128, 256, 2); break;
    case 12: LAUNCH(128, 64, 256, 1); break;
    case 13: LAUNCH(128, 64, 256, 2); break;
#ifdef VERITAC_ENABLE_K192
    case 8: LAUNCH(32, 192, 192, 2); break;
    case 9: LAUNCH(32, 192, 192, 1); break;
#endif
    default: TORCH_CHECK(false, "unknown config id ", id);
  }
#undef LAUNCH
}

std::string validate_dispatch(int id, const float* q, const float* k,
                              const float* v, float* o, int N, int H, int D,
                              int causal, float scale) {
  if (d192_only(id) && D != 192) {
    return std::string("config ") + std::to_string(id) +
           " supports only D=192";
  }
#define VALIDATE(QT, KT, MDD, MBB)                                             \
  do {                                                                         \
    using CFG = Cfg<QT, KT, MDD, MBB>;                                         \
    auto p = make_params<typename CFG::AK>(q, k, v, o, N, H, D, causal, scale);\
    try {                                                                      \
      CFG::AK::check_supported(p);                                             \
      return std::string("ok");                                                \
    } catch (const c10::Error& ex) {                                           \
      return std::string("check_supported: ") + ex.what();                     \
    }                                                                          \
  } while (0)

  switch (id) {
    case 0: VALIDATE(32, 256, 256, 1);
    case 1: VALIDATE(32, 128, 256, 3);
    case 2: VALIDATE(32, 64, 256, 6);
    case 3: VALIDATE(64, 64, 256, 3);
    case 4: VALIDATE(32, 128, 256, 1);
    case 5: VALIDATE(64, 64, 256, 1);
    case 6: VALIDATE(64, 64, 256, 2);
    case 7: VALIDATE(32, 64, 256, 1);
    case 10: VALIDATE(64, 128, 256, 1);
    case 11: VALIDATE(64, 128, 256, 2);
    case 12: VALIDATE(128, 64, 256, 1);
    case 13: VALIDATE(128, 64, 256, 2);
#ifdef VERITAC_ENABLE_K192
    case 8: VALIDATE(32, 192, 192, 2);
    case 9: VALIDATE(32, 192, 192, 1);
#endif
    default: return std::string("unknown config id ") + std::to_string(id);
  }
#undef VALIDATE
}

}  // namespace veritac_cuda

using namespace veritac_cuda;

void launch_attn(int64_t id, int64_t q, int64_t k, int64_t v, int64_t o,
                 int64_t N, int64_t H, int64_t D, int64_t causal, double scale,
                 int64_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  launch_dispatch((int)id, reinterpret_cast<const float*>(q),
                  reinterpret_cast<const float*>(k),
                  reinterpret_cast<const float*>(v), reinterpret_cast<float*>(o),
                  (int)N, (int)H, (int)D, (int)causal, (float)scale, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Validates Params for a config (host-side, outside timed path).
py::dict validate(int64_t id, int64_t q, int64_t k, int64_t v, int64_t o,
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

py::dict config_info(int64_t id) {
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
  m.def("launch_attn", &launch_attn);
  m.def("config_info", &config_info);
  m.def("validate", &validate);
}
