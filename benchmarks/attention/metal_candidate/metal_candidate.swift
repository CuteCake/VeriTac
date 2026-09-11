// metal_candidate.swift
//
// Standalone Metal FP32 causal-prefill attention candidate kernel and host
// runner. It is invoked by the Python orchestrator (metal_candidate.py), one
// process per (shape, mapping, query_tile, key_tile) configuration, and reports
// raw per-invocation command-buffer GPU durations, synchronized API latencies,
// and the complete output buffer bytes.
//
// This is a correctness-first candidate kernel, explicitly NOT a claimed
// vendor-speedup baseline. It is a reviewable starting point that implements
// the exact flash-style streaming online-softmax schedule frozen for this
// benchmark. Two selectable mappings are provided:
//
//   * "scalar": one thread per query row (query_tile 8 or 16); each thread owns
//     the full D output accumulator (the original correctness baseline).
//   * "simdgroup": one Metal SIMD-group (32 lanes) per query row
//     (query_tile 4 or 8); threads_per_threadgroup == query_tile*32. Each lane
//     computes Q.K over d = lane, lane+32, ... and the dot product is finished
//     with a Metal SIMD reduction; each lane owns D/32 output accumulator
//     scalars (6 for D=192, 8 for D=256), avoiding D floats per thread.
//
// Both mappings keep the identical streaming online-softmax schedule and O(ND)
// storage: K and V tiles are cooperatively staged in one dynamic threadgroup
// allocation of exactly 2*key_tile*D*4 bytes, and there is never an N-by-N
// score/probability buffer.
//
// Usage:
//   xcrun swiftc -O -framework Metal -framework MetalPerformanceShaders \
//       metal_candidate.swift -o metal_candidate
//   ./metal_candidate <request.json>
//
// Request JSON fields:
//   shape:        [B, H, N, D] (rank 4, positive ints)
//   dtype:        "float32" only
//   mapping:      "scalar" or "simdgroup"
//   q_path:       contiguous binary for Q [B,H,N,D] float32
//   k_path:       contiguous binary for K [B,H,N,D] float32
//   v_path:       contiguous binary for V [B,H,N,D] float32
//   output_path:  where to write the result [B,H,N,D] float32 raw bytes
//   response_path:where to write the metadata / timing JSON
//   query_tile:   scalar: 8 or 16; simdgroup: 4 or 8
//   key_tile:     8 or 16 (cooperate-staged key/value tile width)
//   warmup:       warm invocation count (outside timing)
//   samples:      timed sample count (each sample runs `inner` invocations)
//   inner:        invocations per timed sample
//
// Kernel contract (shared by both mappings, frozen for this benchmark):
//   * BHND float32 Q,K,V,O; B=1, H=8, N variable, D only 192 or 256.
//   * Every threadgroup uniformly loops over a number of K/V tiles based only
//     on the maximum query row in its tile, and reaches both barriers on every
//     iteration even when its query row is out of range.
//   * K and V are cooperatively staged in a single dynamic threadgroup
//     allocation of exactly 2*key_tile*D*4 bytes (no N-by-N score/probability
//     buffer anywhere).
//   * Each valid unit (thread for scalar, SIMD-group for simdgroup) computes
//     causal scaled dot products and maintains streaming online-softmax state
//     m, l, and an output accumulator. Partial query/key tiles and N=1536 are
//     handled. State is explicitly initialized.
//
// GPU timing: per invocation, MTLCommandBuffer.gpuStartTime/gpuEndTime deltas
// read after waitUntilCompleted. If those timestamps are invalid (zero or
// non-increasing) they are reported unavailable, never mislabeled.

import Foundation
import Metal
import QuartzCore

// ---------------------------------------------------------------------------
// Request decoding
// ---------------------------------------------------------------------------

private struct Request: Decodable {
    let shape: [Int]
    let dtype: String
    let mapping: String?
    let q_path: String
    let k_path: String
    let v_path: String
    let output_path: String
    let response_path: String
    let query_tile: Int
    let key_tile: Int
    let warmup: Int
    let samples: Int
    let inner: Int

    var mappingValue: String { mapping ?? "scalar" }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

private func fail(_ message: String) -> Never {
    fputs("metal_candidate: \(message)\n", stderr)
    exit(2)
}

/// Structured error used for pre-dispatch failures so that, once the
/// response_path is known, they can be reported in the response JSON instead of
/// exiting the process.
private struct CandidateError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private func readFile(_ path: String) throws -> Data {
    do {
        return try Data(contentsOf: URL(fileURLWithPath: path))
    } catch {
        throw CandidateError(message: "cannot read \(path): \(error)")
    }
}

/// JSON-safe value: NaN/Inf are replaced by string markers so the file never
/// contains non-finite JSON numbers.
private func jsonSafe(_ value: Double) -> Any {
    if value.isNaN { return "NaN" }
    if value.isInfinite { return value > 0 ? "Infinity" : "-Infinity" }
    return value
}

private func writeJSON(_ object: [String: Any], to path: String) {
    do {
        let data = try JSONSerialization.data(withJSONObject: object,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        fail("cannot write \(path): \(error)")
    }
}

// ---------------------------------------------------------------------------
// Metal kernel (embedded source; compiled once at runtime via makeLibrary)
// ---------------------------------------------------------------------------

private let kernelSource = """
#include <metal_stdlib>
using namespace metal;

// Scalar mapping: register output accumulator bound. Only [0, D) lanes are
// used. Must be >= max supported D (256).
constant uint MAX_D = 256;
// SIMD mapping: per-lane accumulator count bound. D/32 for D=256 is 8.
constant uint MAX_ACC = 8;
constant uint SIMD_WIDTH = 32;

struct KernelParams {
    uint N;
    uint D;
    uint H;
    uint B;
    uint queryTile;
    uint keyTile;
    float scale;
};

// Shared by both kernels: number of uniform key blocks this threadgroup must
// loop over, derived only from the maximum query row in this query tile so
// every thread in the group reaches both barriers the same number of times.
static inline uint keyBlockCount(uint rowStart, uint queryTile, uint N, uint keyTile) {
    const uint lastRow = min(rowStart + queryTile, N) - 1u; // rowStart < N always
    return (lastRow + 1u + keyTile - 1u) / keyTile;
}

// ---------------------------------------------------------------------------
// Scalar mapping: one thread per query row, threads_per_threadgroup == queryTile
// ---------------------------------------------------------------------------
kernel void causal_prefill_metal(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant KernelParams& params [[buffer(10)]],
    threadgroup float* tg [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tptg_in [[thread_position_in_threadgroup]]
) {
    const uint N = params.N;
    const uint D = params.D;
    const uint H = params.H;
    const uint B = params.B;
    const uint queryTile = params.queryTile;
    const uint keyTile = params.keyTile;
    const float scale = params.scale;

    const uint ktD = keyTile * D;                 // floats per staged K (or V) tile

    // 2D grid: gid.x iterates query tiles within a head; gid.y encodes the
    // (batch, head) plane as (b*H + h). All threads in a threadgroup therefore
    // share one head, so the K/V staging is uniform across the group.
    const uint qy = tgid.y;                       // b*H + h
    const uint h = qy % H;
    const uint b = qy / H;

    const uint tid = tptg_in.x;               // thread within the threadgroup
    const uint rowStart = tgid.x * queryTile; // first query row of this tile
    const uint i = rowStart + tid;            // this thread's query row
    const bool valid = (i < N);
    // Clamp to a safe in-bounds row before forming the device pointer below.
    // Tail threads (i >= N) still traverse both barriers but never dereference
    // qrow; clamping keeps every Q pointer within the allocation.
    const uint safeI = min(i, N - 1u);
    const uint qidx = (b * H + h) * N + safeI; // flat [B,H,N] row index

    // Base offset into the K/V buffers for this (batch, head) plane.
    const uint kvBase = qy * N * D;

    // Explicitly initialize all per-thread streaming state.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_D];
    for (uint d = 0; d < D; ++d) { acc[d] = 0.0f; }

    const device float* qrow = Q + qidx * D;

    // Uniformly loop over every needed K/V tile. The bound is derived from the
    // maximum query row in this tile (uniform across the group), and both
    // barriers are reached by every thread on every iteration, including
    // out-of-range query rows, so the staged K/V tile is never overwritten
    // while another thread still reads it.
    const uint nKeyBlocks = keyBlockCount(rowStart, queryTile, N, keyTile);
    for (uint kb = 0; kb < nKeyBlocks; ++kb) {
        const uint kt = kb * keyTile;
        // Cooperative load of the K and V tiles for key columns
        // [kt, kt+keyTile) into the single threadgroup allocation of
        // 2*keyTile*D floats: [0,ktD) = K tile, [ktD,2*ktD) = V tile.
        const uint total = 2u * ktD;
        for (uint e = tid; e < total; e += queryTile) {
            if (e < ktD) {
                const uint localCol = e / D;
                const uint dd = e % D;
                const uint c = kt + localCol;
                // Guard against a partial trailing key tile (c >= N).
                tg[e] = (c < N) ? K[kvBase + c * D + dd] : 0.0f;
            } else {
                const uint e2 = e - ktD;
                const uint localCol = e2 / D;
                const uint dd = e2 % D;
                const uint c = kt + localCol;
                tg[e] = (c < N) ? V[kvBase + c * D + dd] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // barrier 1: staging ready

        if (valid) {
            // Causal range for this query row within the current tile.
            // Only keys j with j <= i are attended to; hi never exceeds N, so a
            // partial trailing key tile contributes nothing and is never read.
            const uint hi = min(kt + keyTile, i + 1u);
            for (uint jj = kt; jj < hi; ++jj) {
                const uint localCol = jj - kt;
                const threadgroup float* kcol = tg + localCol * D;
                const threadgroup float* vcol = tg + (ktD + localCol * D);

                // Scaled dot product in float32. FP32 accuracy is established
                // by numerical validation against the float64 reference, not
                // claimed a priori.
                float s = 0.0f;
                for (uint d = 0; d < D; ++d) {
                    s += qrow[d] * kcol[d];
                }
                s *= scale;

                // Streaming online-softmax update (flash-style).
                const float new_m = max(m, s);
                const float alpha = exp(m - new_m);
                const float p = exp(s - new_m);
                l = l * alpha + p;
                for (uint d = 0; d < D; ++d) {
                    acc[d] = acc[d] * alpha + p * vcol[d];
                }
                m = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // barrier 2: before next tile
    }

    if (valid) {
        for (uint d = 0; d < D; ++d) {
            O[qidx * D + d] = acc[d] / l;
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD mapping: one SIMD-group (32 lanes) per query row,
// threads_per_threadgroup == queryTile * 32
// ---------------------------------------------------------------------------
kernel void causal_prefill_metal_simd(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant KernelParams& params [[buffer(10)]],
    threadgroup float* tg [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tptg_in [[thread_position_in_threadgroup]]
) {
    const uint N = params.N;
    const uint D = params.D;
    const uint H = params.H;
    const uint B = params.B;
    const uint queryTile = params.queryTile;
    const uint keyTile = params.keyTile;
    const float scale = params.scale;

    const uint ktD = keyTile * D;                 // floats per staged K (or V) tile
    const uint threads = queryTile * SIMD_WIDTH;  // threads per threadgroup

    const uint qy = tgid.y;                       // b*H + h
    const uint h = qy % H;
    const uint b = qy / H;

    const uint tid = tptg_in.x;               // thread within the threadgroup
    const uint sg = tid / SIMD_WIDTH;         // SIMD-group index (= query row in tile)
    const uint lane = tid % SIMD_WIDTH;       // lane within the SIMD-group
    const uint accPerLane = D / SIMD_WIDTH;   // accumulators owned by this lane

    const uint rowStart = tgid.x * queryTile;
    const uint i = rowStart + sg;             // this SIMD-group's query row
    // valid is uniform across the SIMD-group (all 32 lanes share sg -> same i),
    // so masked keys are skipped without diverging the group's lanes.
    const bool valid = (i < N);
    const uint safeI = min(i, N - 1u);
    const uint qidx = (b * H + h) * N + safeI;
    const uint kvBase = qy * N * D;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_ACC];
    for (uint k = 0; k < accPerLane; ++k) { acc[k] = 0.0f; }

    const device float* qrow = Q + qidx * D;

    const uint nKeyBlocks = keyBlockCount(rowStart, queryTile, N, keyTile);
    for (uint kb = 0; kb < nKeyBlocks; ++kb) {
        const uint kt = kb * keyTile;
        const uint total = 2u * ktD;
        // Cooperative load across all threads in the group (all SIMD-groups
        // participate), so the barrier below is uniform.
        for (uint e = tid; e < total; e += threads) {
            if (e < ktD) {
                const uint localCol = e / D;
                const uint dd = e % D;
                const uint c = kt + localCol;
                tg[e] = (c < N) ? K[kvBase + c * D + dd] : 0.0f;
            } else {
                const uint e2 = e - ktD;
                const uint localCol = e2 / D;
                const uint dd = e2 % D;
                const uint c = kt + localCol;
                tg[e] = (c < N) ? V[kvBase + c * D + dd] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // barrier 1: staging ready

        if (valid) {
            const uint hi = min(kt + keyTile, i + 1u);
            for (uint jj = kt; jj < hi; ++jj) {
                const uint localCol = jj - kt;
                const threadgroup float* kcol = tg + localCol * D;
                const threadgroup float* vcol = tg + (ktD + localCol * D);

                // Each lane accumulates its strided D subset (d = lane + 32*k),
                // then the SIMD reduction completes the dot product in every
                // lane.
                float s = 0.0f;
                for (uint k = 0; k < accPerLane; ++k) {
                    const uint d = lane + SIMD_WIDTH * k;
                    s += qrow[d] * kcol[d];
                }
                s = simd_sum(s) * scale;

                // Streaming online-softmax update. All 32 lanes see the same s
                // (reduction broadcast) and perform identical arithmetic, so m
                // and l stay consistent across the group.
                const float new_m = max(m, s);
                const float alpha = exp(m - new_m);
                const float p = exp(s - new_m);
                l = l * alpha + p;
                for (uint k = 0; k < accPerLane; ++k) {
                    const uint d = lane + SIMD_WIDTH * k;
                    acc[k] = acc[k] * alpha + p * vcol[d];
                }
                m = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // barrier 2: before next tile
    }

    if (valid) {
        const float invL = 1.0f / l;
        for (uint k = 0; k < accPerLane; ++k) {
            const uint d = lane + SIMD_WIDTH * k;
            O[qidx * D + d] = acc[k] * invL;
        }
    }
}
"""

// Packed kernel parameters, layout must match the MSL KernelParams struct.
private struct KernelParams {
    var N: UInt32
    var D: UInt32
    var H: UInt32
    var B: UInt32
    var queryTile: UInt32
    var keyTile: UInt32
    var scale: Float
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

private func run() {
    let args = CommandLine.arguments
    guard args.count == 2 else {
        fail("usage: metal_candidate <request.json>")
    }

    let requestPath = args[1]
    let request: Request
    do {
        request = try JSONDecoder().decode(Request.self,
                                           from: readFile(requestPath))
    } catch {
        // response_path is not yet known here, so a structured response cannot
        // be written; report on stderr and exit.
        fail("cannot decode \(requestPath): \(error)")
    }

    var response: [String: Any] = ["status": "error"]

    let finish = { (payload: [String: Any]) in
        do {
            let data = try JSONSerialization.data(withJSONObject: payload,
                                                  options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: request.response_path))
        } catch {
            fail("cannot write response: \(error)")
        }
    }

    do {
        // ---- Validate sizes before dispatch --------------------------------
        guard request.shape.count == 4 else {
            throw NSError(domain: "metal_candidate", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shape must be rank 4"])
        }
        let b = request.shape[0]
        let h = request.shape[1]
        let n = request.shape[2]
        let d = request.shape[3]
        guard b == 1 else {
            throw NSError(domain: "metal_candidate", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "B must be 1"])
        }
        guard h == 8 else {
            throw NSError(domain: "metal_candidate", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "H must be 8"])
        }
        guard n > 0 else {
            throw NSError(domain: "metal_candidate", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "N must be positive"])
        }
        guard d == 192 || d == 256 else {
            throw NSError(domain: "metal_candidate", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "D must be 192 or 256"])
        }
        let mapping = request.mappingValue
        let isSimd = (mapping == "simdgroup")
        let isScalar = (mapping == "scalar")
        guard isScalar || isSimd else {
            throw NSError(domain: "metal_candidate", code: 6,
                          userInfo: [NSLocalizedDescriptionKey:
                            "mapping must be \"scalar\" or \"simdgroup\" (got \"\(mapping)\")"])
        }
        let validQT: Set<Int> = isScalar ? [8, 16] : [4, 8]
        guard validQT.contains(request.query_tile) else {
            throw NSError(domain: "metal_candidate", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                            "query_tile \(request.query_tile) invalid for mapping \(mapping); allowed \(validQT.sorted())"])
        }
        guard request.key_tile == 8 || request.key_tile == 16 else {
            throw NSError(domain: "metal_candidate", code: 8,
                          userInfo: [NSLocalizedDescriptionKey: "key_tile must be 8 or 16"])
        }
        guard request.dtype == "float32" else {
            throw NSError(domain: "metal_candidate", code: 9,
                          userInfo: [NSLocalizedDescriptionKey: "dtype must be float32"])
        }
        guard request.warmup >= 0, request.samples >= 0, request.inner >= 1 else {
            throw NSError(domain: "metal_candidate", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "warmup/samples >= 0, inner >= 1"])
        }

        // threads per threadgroup depends on the mapping.
        let threadsPerTG = isScalar ? request.query_tile : request.query_tile * 32

        // Scalar cooperative staging requires 2*key_tile*D elements to divide
        // evenly among the query_tile threads (which own the full D accumulator).
        // The SIMD mapping stages cooperatively across query_tile*32 threads with
        // a plain strided loop, so no divisibility constraint applies there.
        if isScalar {
            let loadElems = 2 * request.key_tile * d
            guard loadElems % request.query_tile == 0 else {
                throw NSError(domain: "metal_candidate", code: 11,
                              userInfo: [NSLocalizedDescriptionKey:
                                "cooperative load count \(loadElems) not divisible by query_tile \(request.query_tile)"])
            }
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "metal_candidate", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "no default Metal device"])
        }
        response["mtl_device_name"] = device.name
        response["mtl_device_registry_id"] = NSNumber(value: device.registryID)
        response["mtl_is_low_power"] = device.isLowPower
        response["mtl_has_unified_memory"] = device.hasUnifiedMemory

        let tgMemBytes = 2 * request.key_tile * d * 4
        guard tgMemBytes <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "metal_candidate", code: 13,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgMemBytes) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }
        let maxTG = device.maxThreadsPerThreadgroup
        guard threadsPerTG <= maxTG.width else {
            throw NSError(domain: "metal_candidate", code: 14,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threads per threadgroup \(threadsPerTG) exceeds max threads per threadgroup width \(maxTG.width)"])
        }

        // ---- Launch geometry -------------------------------------------------
        let gridX = (n + request.query_tile - 1) / request.query_tile
        let gridY = b * h                       // = 8 for B=1, H=8
        let totalThreads = gridX * gridY * threadsPerTG
        let validRows = b * h * n

        let queue = device.makeCommandQueue()!

        // ---- Compile once ----------------------------------------------------
        let compileStart = CACurrentMediaTime()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw NSError(domain: "metal_candidate", code: 15,
                          userInfo: [NSLocalizedDescriptionKey: "MSL compile failed: \(error)"])
        }
        let kernelName = isScalar ? "causal_prefill_metal" : "causal_prefill_metal_simd"
        guard let fn = library.makeFunction(name: kernelName),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw NSError(domain: "metal_candidate", code: 16,
                          userInfo: [NSLocalizedDescriptionKey: "could not create compute pipeline for \(kernelName)"])
        }

        // Validate the compiled pipeline itself against the device limits, not
        // just the raw device numbers, and record the compiled values.
        let maxTotalTG = pipeline.maxTotalThreadsPerThreadgroup
        let staticTG = pipeline.staticThreadgroupMemoryLength
        let simdWidth = pipeline.threadExecutionWidth
        guard threadsPerTG <= maxTotalTG else {
            throw NSError(domain: "metal_candidate", code: 17,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threads per threadgroup \(threadsPerTG) exceeds compiled-pipeline maxTotalThreadsPerThreadgroup \(maxTotalTG)"])
        }
        if isSimd {
            guard simdWidth == 32 else {
                throw NSError(domain: "metal_candidate", code: 19,
                              userInfo: [NSLocalizedDescriptionKey:
                                "simdgroup mapping requires pipeline SIMD width 32, got \(simdWidth)"])
            }
        }
        let tgMemPlusStatic = tgMemBytes + staticTG
        guard tgMemPlusStatic <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "metal_candidate", code: 18,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgMemBytes) + static \(staticTG) = \(tgMemPlusStatic) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }
        response["pipeline"] = [
            "max_total_threads_per_threadgroup": NSNumber(value: maxTotalTG),
            "simd_width": NSNumber(value: simdWidth),
            "static_threadgroup_memory_length": NSNumber(value: staticTG),
            "dynamic_threadgroup_memory_length": tgMemBytes,
            "total_threadgroup_memory_length": tgMemPlusStatic,
            "threadgroup_memory_within_limit": (tgMemPlusStatic <= device.maxThreadgroupMemoryLength),
            "threads_per_threadgroup_within_max_total": (threadsPerTG <= maxTotalTG),
            "simd_width_ok_for_simdgroup": (!isSimd || simdWidth == 32),
        ]
        response["compile_wall_ms"] = jsonSafe((CACurrentMediaTime() - compileStart) * 1000.0)

        // ---- Inputs ----------------------------------------------------------
        let totalBytes = b * h * n * d * 4
        func inputBuffer(_ path: String) throws -> MTLBuffer {
            let data = try readFile(path)
            guard data.count == totalBytes else {
                throw CandidateError(message: "input \(path) has \(data.count) bytes, expected \(totalBytes)")
            }
            let buf = device.makeBuffer(length: totalBytes, options: .storageModeShared)!
            data.withUnsafeBytes { raw in
                buf.contents().copyMemory(from: raw.baseAddress!, byteCount: totalBytes)
            }
            return buf
        }

        let qBuf = try inputBuffer(request.q_path)
        let kBuf = try inputBuffer(request.k_path)
        let vBuf = try inputBuffer(request.v_path)
        let outBuf = device.makeBuffer(length: totalBytes, options: .storageModeShared)!

        var params = KernelParams(N: UInt32(n), D: UInt32(d), H: UInt32(h),
                                  B: UInt32(b), queryTile: UInt32(request.query_tile),
                                  keyTile: UInt32(request.key_tile),
                                  scale: Float(1.0 / Double(d).squareRoot()))
        let paramsBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<KernelParams>.stride,
                                          options: .storageModeShared)!

        // ---- Encoded invocation ----------------------------------------------
        func runOnce() throws -> (wall: Double, gpuStart: Double, gpuEnd: Double,
                                  kernelStart: Double, kernelEnd: Double) {
            let t0 = CACurrentMediaTime()
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2)
            enc.setBuffer(outBuf, offset: 0, index: 3)
            enc.setBuffer(paramsBuf, offset: 0, index: 10)
            enc.setThreadgroupMemoryLength(tgMemBytes, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: gridX, height: gridY, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerTG,
                                                                    height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let t1 = CACurrentMediaTime()
            if cb.status == .error {
                let detail = cb.error?.localizedDescription ?? "unknown command-buffer error"
                throw NSError(domain: "metal_candidate", code: 20,
                              userInfo: [NSLocalizedDescriptionKey: "GPU error: \(detail)"])
            }
            return ((t1 - t0) * 1000.0,
                    cb.gpuStartTime * 1000.0,
                    cb.gpuEndTime * 1000.0,
                    cb.kernelStartTime * 1000.0,
                    cb.kernelEndTime * 1000.0)
        }

        // ---- Warmup / compile outside the timed region ------------------------
        let warmupCount = max(request.warmup, 0)
        for _ in 0..<warmupCount {
            _ = try runOnce()
        }
        response["warmup_count"] = warmupCount

        // ---- Timed samples ----------------------------------------------------
        var wallTimes: [Double] = []
        var gpuTimes: [Double] = []
        var gpuStartTimes: [Double] = []
        var gpuEndTimes: [Double] = []
        var gpuValidCount = 0
        let totalRuns = request.samples * request.inner
        for _ in 0..<request.samples {
            for _ in 0..<request.inner {
                let r = try runOnce()
                wallTimes.append(r.wall)
                gpuStartTimes.append(r.gpuStart)
                gpuEndTimes.append(r.gpuEnd)
                if r.gpuStart > 0, r.gpuEnd > 0, r.gpuEnd >= r.gpuStart {
                    gpuTimes.append(r.gpuEnd - r.gpuStart)
                    gpuValidCount += 1
                }
            }
        }
        let gpuAvailable = (gpuValidCount == totalRuns)
        response["gpu_time_available"] = gpuAvailable
        response["gpu_times"] = gpuTimes.map { jsonSafe($0) }
        response["gpu_start_times"] = gpuStartTimes.map { jsonSafe($0) }
        response["gpu_end_times"] = gpuEndTimes.map { jsonSafe($0) }
        response["wall_times"] = wallTimes.map { jsonSafe($0) }
        response["time_unit"] = "milliseconds"
        response["gpu_timestamp_units"] = "milliseconds"

        // ---- Output -----------------------------------------------------------
        let outRaw = Data(bytes: outBuf.contents(), count: totalBytes)
        try outRaw.write(to: URL(fileURLWithPath: request.output_path))

        response["status"] = "ok"
        response["shape"] = request.shape
        response["dtype"] = request.dtype
        response["mapping"] = mapping
        response["query_tile"] = request.query_tile
        response["key_tile"] = request.key_tile
        response["threads_per_threadgroup"] = threadsPerTG
        response["samples"] = request.samples
        response["inner"] = request.inner
        response["scale"] = Double(params.scale)
        response["output_bytes"] = totalBytes
        response["launch"] = [
            "grid": ["x": gridX, "y": gridY, "z": 1],
            "mapping": mapping,
            "threads_per_threadgroup": threadsPerTG,
            "total_threads_dispatched": totalThreads,
            "valid_query_rows": validRows,
            "dynamic_threadgroup_memory_bytes": tgMemBytes,
            "static_threadgroup_memory_bytes": staticTG,
            "total_threadgroup_memory_bytes": tgMemPlusStatic,
            "threadgroup_memory_float_elements": 2 * request.key_tile * d,
            "kernel_name": kernelName,
            "dynamic_tg_staging": "k_and_v_in_single_allocation",
            "no_n_by_n_buffer": true,
        ]
        response["device"] = [
            "max_threadgroup_memory_length": NSNumber(value: device.maxThreadgroupMemoryLength),
            "max_threads_per_threadgroup": [
                "width": NSNumber(value: maxTG.width),
                "height": NSNumber(value: maxTG.height),
                "depth": NSNumber(value: maxTG.depth),
            ],
            "threadgroup_memory_bytes": tgMemBytes,
            "threadgroup_memory_within_limit": (tgMemBytes <= device.maxThreadgroupMemoryLength),
        ]
        response["os"] = [
            "version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        response["measurement_notes"] = [
            "All timings are in milliseconds (time_unit=milliseconds).",
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per invocation, read after waitUntilCompleted; they exclude input/output CPU copies (preallocated shared MTLBuffers reused) and exclude compile/warmup.",
            "Wall times are synchronized API latency, timed from before command-buffer allocation through waitUntilCompleted, per invocation.",
            "The Metal shader is compiled once at startup via MTLLibrary makeLibrary(source:) and reused across all invocations.",
            "A command buffer ending in a Metal error fails the case rather than reporting a bogus timing.",
            "Mapping \"scalar\" launches one thread per query row (threads_per_threadgroup == query_tile); mapping \"simdgroup\" launches one 32-lane SIMD-group per query row (threads_per_threadgroup == query_tile * 32) with a SIMD reduction to finish each dot product.",
            "This candidate kernel is a correctness-first flash-style streaming online-softmax implementation; no vendor speedup is claimed.",
        ]

        finish(response)
    } catch {
        response["status"] = "error"
        response["error"] = "\(type(of: error)): \(error.localizedDescription)"
        finish(response)
    }
}

run()
