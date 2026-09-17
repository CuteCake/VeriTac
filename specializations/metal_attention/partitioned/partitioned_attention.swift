// partitioned_attention.swift
//
// Standalone Metal FP32 causal-prefill attention kernel + host runner for the
// "partitioned exact-real" transform experiment.
//
// Idea (creative exact-real transform, no approximate / lower-precision
// substitution, FP32 contract atol=1e-4 rtol=1e-3 frozen):
//
//   For each query row r, the causal key range [0, r] is partitioned into
//   `splits` contiguous chunks (floor-division bounds, so every key is counted
//   exactly once and irregular tails are handled).  Each partition is processed
//   by one 32-lane SIMD-group which computes, fully independently:
//
//       m_p = max_j exp-scale score            (partition local max)
//       l_p = sum_j exp(s_j - m_p)             (partition sum_exp)
//       o_p[d] = sum_j exp(s_j - m_p) * v_j[d] (partition weighted value sum)
//
//   Then a single threadgroup barrier + a stable merge combines the
//   independent per-partition triples into the global result:
//
//       M   = max_p m_p
//       l   = sum_p l_p * exp(m_p - M)
//       o[d]= sum_p o_p[d] * exp(m_p - M)
//       out = o / l
//
//   Empty partitions (hi <= lo, which occur for small rows with many splits)
//   are stored with m_p = -inf, l_p = 0, o_p = 0.  exp(-inf - M) = 0 so they
//   contribute zero to both the normalization and the weighted sum: no NaN, no
//   double counting, each key counted exactly once.
//
// Compile-time specialization: D, `splits`, and the pass-1 max scheme are
// function constants set at pipeline build, so each (dim, splits, scheme)
// configuration gets a fully specialized pipeline (loop trip counts and the
// per-lane accumulator counts become constants).
//
// Pass-1 max scheme (how a partition computes its local max m_p):
//   * "dsplit":  the 32 lanes split the D dimension (each lane owns D/32
//                elements), iterating keys serially and finishing each score
//                with a simd_sum.  K/V reads are coalesced across lanes and Q
//                is cached in registers (each lane caches only its D/32 subset
//                = 6 or 8 floats).
//   * "keysplit":the 32 lanes split the partition's keys (each lane scans its
//                own keys with a full-D dot product, no per-key simd_sum) and
//                combine with a single simd_max at the end.  Pass B (weighted
//                sum) is always "dsplit" because the o accumulator must be
//                distributed across the D dimension to fit in registers.
//
// Both schemes produce the exact same partition max m_p; only the reduction
// structure and memory access pattern differ.
//
// Usage:
//   xcrun swiftc -O -framework Metal -framework MetalPerformanceShaders \
//       partitioned_attention.swift -o partitioned_attention
//   ./partitioned_attention <request.json>
//
// Request JSON fields (same file-based contract as metal_candidate):
//   shape:      [B, H, N, D]  (B=1, H=8, D=192|256)
//   dtype:      "float32"
//   q/k/v_path: contiguous float32 binaries
//   output_path, response_path
//   splits:     partition count (1, 4, 8, 16, 32)
//   max_scheme: "dsplit" or "keysplit"
//   warmup/samples/inner
//
// Timing: per invocation the whole single-kernel pipeline (pass A + pass B +
// merge, one MTLComputeCommandEncoder) is measured with BOTH
// MTLCommandBuffer gpuStartTime/gpuEndTime deltas and synchronized wall
// latency.  Nothing is cached across samples; each invocation writes a fresh
// output.  GPU timestamps that are invalid are reported unavailable, never
// fabricated.

import Foundation
import Metal
import QuartzCore

// ---------------------------------------------------------------------------
// Request decoding
// ---------------------------------------------------------------------------

private struct Request: Decodable {
    let shape: [Int]
    let dtype: String
    let q_path: String
    let k_path: String
    let v_path: String
    let output_path: String
    let response_path: String
    let splits: Int
    let max_scheme: String
    let warmup: Int
    let samples: Int
    let inner: Int
}

private func fail(_ message: String) -> Never {
    fputs("partitioned_attention: \(message)\n", stderr)
    exit(2)
}

private struct RunnerError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private func readFile(_ path: String) throws -> Data {
    do {
        return try Data(contentsOf: URL(fileURLWithPath: path))
    } catch {
        throw RunnerError(message: "cannot read \(path): \(error)")
    }
}

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
// Metal kernel (embedded source; specialized per config via function constants)
// ---------------------------------------------------------------------------

private let kernelSource = """
#include <metal_stdlib>
using namespace metal;

// Function constants (set at pipeline build => compile-time specialization).
constant uint D_FC     [[function_constant(0)]];
constant uint SPLITS_FC[[function_constant(1)]];
constant uint SCHEME_FC[[function_constant(2)]];   // 0 = dsplit, 1 = keysplit

constant uint SIMD_WIDTH = 32;
constant uint MAXD  = 256;
constant uint MAXACC = 8;
constant uint MAXS  = 32;

struct KernelParams {
    uint N;
    uint H;
    uint B;
    float scale;
};

// Partition bounds for row `row` (which has row+1 keys), partition p:
//   lo = p*(row+1)/S , hi = (p+1)*(row+1)/S   (integer floor)
// The chunks are contiguous, cover [0, row+1) exactly, and each key lands in
// exactly one partition; empty partitions occur when hi <= lo.
static inline void partition_bounds(uint row, uint p, uint S,
                                    thread uint& lo, thread uint& hi) {
    const uint r1 = row + 1u;
    lo = (p * r1) / S;
    hi = ((p + 1u) * r1) / S;
}

kernel void partitioned_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant KernelParams& params [[buffer(10)]],
    threadgroup float* tg [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tptg_in [[thread_position_in_threadgroup]]
) {
    const uint tid = tptg_in.x;
    const uint N = params.N;
    const uint H = params.H;
    const uint D = D_FC;
    const uint S = SPLITS_FC;
    const uint scheme = SCHEME_FC;
    const float scale = params.scale;
    const uint accPerLane = D / SIMD_WIDTH;   // 6 (D=192) or 8 (D=256)

    const uint row = tgid.x;                 // query row within head (b=0)
    const uint h = tgid.y;                   // head plane
    const uint base = (h * N + row) * D;     // Q row base
    const uint kvbase = h * N * D;           // K/V head base

    const uint sg = tid / SIMD_WIDTH;        // SIMD-group index == partition p
    const uint lane = tid % SIMD_WIDTH;
    const uint p = sg;

    // Cache this lane's Q subset in registers (d = lane + 32*k).
    float qr[MAXACC];
    for (uint k = 0; k < accPerLane; ++k) {
        qr[k] = Q[base + lane + SIMD_WIDTH * k];
    }

    // Partition bounds.
    uint lo, hi;
    partition_bounds(row, p, S, lo, hi);
    const bool empty = (hi <= lo);

    // ---- Pass A: partition-local max m_p --------------------------------
    float m = -INFINITY;
    if (scheme == 0u) {
        // dsplit: lanes split D; per-key simd_sum; coalesced K.
        for (uint j = lo; j < hi; ++j) {
            float partial = 0.0f;
            for (uint k = 0; k < accPerLane; ++k) {
                const uint d = lane + SIMD_WIDTH * k;
                partial += qr[k] * K[kvbase + j * D + d];
            }
            m = max(m, simd_sum(partial) * scale);
        }
    } else {
        // keysplit: lanes split keys; full-D dot; single simd_max at end.
        float mloc = -INFINITY;
        for (uint j = lo + lane; j < hi; j += SIMD_WIDTH) {
            float s = 0.0f;
            for (uint d = 0; d < D; ++d) {
                s += Q[base + d] * K[kvbase + j * D + d];
            }
            mloc = max(mloc, s * scale);
        }
        m = simd_max(mloc);
    }

    // ---- Pass B: partition-local sums (weighted value sum, D-split) -----
    float l = 0.0f;
    float o[MAXACC];
    for (uint k = 0; k < accPerLane; ++k) { o[k] = 0.0f; }

    if (!empty) {
        for (uint j = lo; j < hi; ++j) {
            float partial = 0.0f;
            for (uint k = 0; k < accPerLane; ++k) {
                const uint d = lane + SIMD_WIDTH * k;
                partial += qr[k] * K[kvbase + j * D + d];
            }
            const float s = simd_sum(partial) * scale;
            const float pv = exp(s - m);
            l += pv;
            for (uint k = 0; k < accPerLane; ++k) {
                const uint d = lane + SIMD_WIDTH * k;
                o[k] += pv * V[kvbase + j * D + d];
            }
        }
    }

    // ---- Store per-partition triples to threadgroup, then merge ---------
    // Layout: tg[0..S)      = m_p
    //         tg[S..2S)     = l_p
    //         tg[2S .. 2S+S*D) = o_p[d]
    threadgroup float* tg_m = tg;
    threadgroup float* tg_l = tg + S;
    threadgroup float* tg_o = tg + 2u * S;

    if (empty) {
        if (lane == 0u) { tg_m[p] = -INFINITY; tg_l[p] = 0.0f; }
        for (uint k = 0; k < accPerLane; ++k) {
            tg_o[p * D + lane + SIMD_WIDTH * k] = 0.0f;
        }
    } else {
        if (lane == 0u) { tg_m[p] = m; tg_l[p] = l; }
        for (uint k = 0; k < accPerLane; ++k) {
            tg_o[p * D + lane + SIMD_WIDTH * k] = o[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Stable merge across partitions ---------------------------------
    // Exactly one SIMD-group owns the final output. All groups participated
    // in the preceding barrier, and no later barrier requires them to remain.
    if (sg != 0u) { return; }
    float M = -INFINITY;
    for (uint pp = 0; pp < S; ++pp) { M = max(M, tg_m[pp]); }

    float alpha[MAXS];
    for (uint pp = 0; pp < S; ++pp) { alpha[pp] = exp(tg_m[pp] - M); }

    float lsum = 0.0f;
    for (uint pp = 0; pp < S; ++pp) { lsum += tg_l[pp] * alpha[pp]; }
    const float invL = 1.0f / lsum;

    // Only the first SIMD-group writes the output; every partition's lanes
    // compute the same merged value, so writing from every SIMD-group would
    // race on the same addresses.  sg==0 is the single writer.
    if (sg == 0u) {
        for (uint k = 0; k < accPerLane; ++k) {
            const uint d = lane + SIMD_WIDTH * k;
            float os = 0.0f;
            for (uint pp = 0; pp < S; ++pp) {
                os += tg_o[pp * D + d] * alpha[pp];
            }
            O[base + d] = os * invL;
        }
    }
}
"""

// Layout of the MSL KernelParams struct.
private struct KernelParams {
    var N: UInt32
    var H: UInt32
    var B: UInt32
    var scale: Float
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

private func run() {
    let args = CommandLine.arguments
    guard args.count == 2 else {
        fail("usage: partitioned_attention <request.json>")
    }

    let requestPath = args[1]
    let request: Request
    do {
        request = try JSONDecoder().decode(Request.self,
                                           from: readFile(requestPath))
    } catch {
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
        // ---- Validate request -------------------------------------------
        guard request.shape.count == 4 else {
            throw NSError(domain: "partitioned_attention", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shape must be rank 4"])
        }
        let b = request.shape[0]
        let h = request.shape[1]
        let n = request.shape[2]
        let d = request.shape[3]
        guard b == 1 else {
            throw NSError(domain: "partitioned_attention", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "B must be 1"])
        }
        guard h == 8 else {
            throw NSError(domain: "partitioned_attention", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "H must be 8"])
        }
        guard n > 0 else {
            throw NSError(domain: "partitioned_attention", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "N must be positive"])
        }
        guard d == 192 || d == 256 else {
            throw NSError(domain: "partitioned_attention", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "D must be 192 or 256"])
        }
        let allowedSplits = [1, 4, 8, 16, 32]
        guard allowedSplits.contains(request.splits) else {
            throw NSError(domain: "partitioned_attention", code: 6,
                          userInfo: [NSLocalizedDescriptionKey:
                            "splits must be one of \(allowedSplits) (got \(request.splits))"])
        }
        let scheme: UInt32
        switch request.max_scheme {
        case "dsplit": scheme = 0
        case "keysplit": scheme = 1
        default:
            throw NSError(domain: "partitioned_attention", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                            "max_scheme must be dsplit or keysplit (got \(request.max_scheme))"])
        }
        guard request.dtype == "float32" else {
            throw NSError(domain: "partitioned_attention", code: 8,
                          userInfo: [NSLocalizedDescriptionKey: "dtype must be float32"])
        }
        guard request.warmup >= 0, request.samples >= 0, request.inner >= 1 else {
            throw NSError(domain: "partitioned_attention", code: 9,
                          userInfo: [NSLocalizedDescriptionKey: "warmup/samples >= 0, inner >= 1"])
        }

        let threadsPerTG = request.splits * 32

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "partitioned_attention", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "no default Metal device"])
        }
        response["mtl_device_name"] = device.name
        response["mtl_device_registry_id"] = NSNumber(value: device.registryID)
        response["mtl_is_low_power"] = device.isLowPower
        response["mtl_has_unified_memory"] = device.hasUnifiedMemory

        // Threadgroup memory for the merge: S + S + S*D floats.
        let tgMemBytes = (request.splits * d + 2 * request.splits) * 4
        guard tgMemBytes <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "partitioned_attention", code: 11,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgMemBytes) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }
        let maxTG = device.maxThreadsPerThreadgroup
        guard threadsPerTG <= maxTG.width else {
            throw NSError(domain: "partitioned_attention", code: 12,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threads per threadgroup \(threadsPerTG) exceeds max width \(maxTG.width)"])
        }

        // ---- Launch geometry ---------------------------------------------
        let gridX = n                    // one threadgroup per query row
        let gridY = b * h                // = 8
        let totalThreads = gridX * gridY * threadsPerTG
        let validRows = b * h * n

        let queue = device.makeCommandQueue()!

        // ---- Compile with function-constant specialization ---------------
        let compileStart = CACurrentMediaTime()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw NSError(domain: "partitioned_attention", code: 13,
                          userInfo: [NSLocalizedDescriptionKey: "MSL compile failed: \(error)"])
        }
        let constants = MTLFunctionConstantValues()
        var dVal = UInt32(d)
        constants.setConstantValue(&dVal, type: .uint, index: 0)
        var sVal = UInt32(request.splits)
        constants.setConstantValue(&sVal, type: .uint, index: 1)
        var scVal = scheme
        constants.setConstantValue(&scVal, type: .uint, index: 2)
        guard let fn = try? library.makeFunction(name: "partitioned_attention",
                                                 constantValues: constants),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw NSError(domain: "partitioned_attention", code: 14,
                          userInfo: [NSLocalizedDescriptionKey:
                            "could not create specialized pipeline for D=\(d) splits=\(request.splits) scheme=\(request.max_scheme)"])
        }
        let maxTotalTG = pipeline.maxTotalThreadsPerThreadgroup
        let staticTG = pipeline.staticThreadgroupMemoryLength
        let simdWidth = pipeline.threadExecutionWidth
        guard threadsPerTG <= maxTotalTG else {
            throw NSError(domain: "partitioned_attention", code: 15,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threads per threadgroup \(threadsPerTG) exceeds compiled maxTotalThreadsPerThreadgroup \(maxTotalTG)"])
        }
        guard simdWidth == 32 else {
            throw NSError(domain: "partitioned_attention", code: 16,
                          userInfo: [NSLocalizedDescriptionKey:
                            "pipeline SIMD width must be 32, got \(simdWidth)"])
        }
        let tgMemPlusStatic = tgMemBytes + staticTG
        guard tgMemPlusStatic <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "partitioned_attention", code: 17,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgMemBytes) + static \(staticTG) = \(tgMemPlusStatic) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }
        response["pipeline"] = [
            "max_total_threads_per_threadgroup": NSNumber(value: maxTotalTG),
            "simd_width": NSNumber(value: simdWidth),
            "static_threadgroup_memory_length": NSNumber(value: staticTG),
            "dynamic_threadgroup_memory_length": tgMemBytes,
            "total_threadgroup_memory_length": tgMemPlusStatic,
            "specialization": [
                "D": d,
                "splits": request.splits,
                "max_scheme": request.max_scheme,
            ],
        ]
        response["compile_wall_ms"] = jsonSafe((CACurrentMediaTime() - compileStart) * 1000.0)

        // ---- Inputs -------------------------------------------------------
        let totalBytes = b * h * n * d * 4
        func inputBuffer(_ path: String) throws -> MTLBuffer {
            let data = try readFile(path)
            guard data.count == totalBytes else {
                throw RunnerError(message: "input \(path) has \(data.count) bytes, expected \(totalBytes)")
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

        var params = KernelParams(N: UInt32(n), H: UInt32(h), B: UInt32(b),
                                  scale: Float(1.0 / Double(d).squareRoot()))
        let paramsBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<KernelParams>.stride,
                                          options: .storageModeShared)!

        // ---- Encoded invocation ------------------------------------------
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
                throw NSError(domain: "partitioned_attention", code: 18,
                              userInfo: [NSLocalizedDescriptionKey: "GPU error: \(detail)"])
            }
            return ((t1 - t0) * 1000.0,
                    cb.gpuStartTime * 1000.0,
                    cb.gpuEndTime * 1000.0,
                    cb.kernelStartTime * 1000.0,
                    cb.kernelEndTime * 1000.0)
        }

        // ---- Warmup / compile outside timed region ------------------------
        let warmupCount = max(request.warmup, 0)
        for _ in 0..<warmupCount { _ = try runOnce() }
        response["warmup_count"] = warmupCount

        // ---- Timed samples -------------------------------------------------
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

        // ---- Output ---------------------------------------------------------
        let outRaw = Data(bytes: outBuf.contents(), count: totalBytes)
        try outRaw.write(to: URL(fileURLWithPath: request.output_path))

        response["status"] = "ok"
        response["shape"] = request.shape
        response["dtype"] = request.dtype
        response["splits"] = request.splits
        response["max_scheme"] = request.max_scheme
        response["threads_per_threadgroup"] = threadsPerTG
        response["samples"] = request.samples
        response["inner"] = request.inner
        response["scale"] = Double(params.scale)
        response["output_bytes"] = totalBytes
        response["launch"] = [
            "grid": ["x": gridX, "y": gridY, "z": 1],
            "mapping": "one_simd_group_per_partition",
            "simd_groups_per_row": request.splits,
            "lanes_per_simd_group": 32,
            "threads_per_threadgroup": threadsPerTG,
            "total_threads_dispatched": totalThreads,
            "valid_query_rows": validRows,
            "dynamic_threadgroup_memory_bytes": tgMemBytes,
            "static_threadgroup_memory_bytes": staticTG,
            "total_threadgroup_memory_bytes": tgMemPlusStatic,
            "kernel_name": "partitioned_attention",
            "passes": [
                "pass_a_local_max": request.max_scheme,
                "pass_b_weighted_sum": "dsplit",
                "merge": "stable_partition_merge_threadgroup",
            ],
            "single_threadgroup_barrier": true,
            "no_n_by_n_buffer": true,
        ]
        response["device"] = [
            "max_threadgroup_memory_length": NSNumber(value: device.maxThreadgroupMemoryLength),
            "max_threads_per_threadgroup": [
                "width": NSNumber(value: maxTG.width),
                "height": NSNumber(value: maxTG.height),
                "depth": NSNumber(value: maxTG.depth),
            ],
        ]
        response["os"] = ["version": ProcessInfo.processInfo.operatingSystemVersionString]
        response["measurement_notes"] = [
            "All timings are in milliseconds (time_unit=milliseconds).",
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per invocation, read after waitUntilCompleted; they exclude input/output CPU copies (preallocated shared MTLBuffers reused) and exclude compile/warmup.",
            "Wall times are synchronized API latency, timed from before command-buffer allocation through waitUntilCompleted, per invocation.",
            "The Metal shader is compiled once at startup with function-constant specialization for (D, splits, max_scheme) and reused across all invocations.",
            "Each timed invocation runs the FULL single-kernel pipeline: pass A (partition local max), pass B (partition weighted sum), and the stable partition merge. Nothing is cached across samples.",
            "A command buffer ending in a Metal error fails the case rather than reporting a bogus timing.",
        ]

        finish(response)
    } catch {
        response["status"] = "error"
        response["error"] = "\(type(of: error)): \(error.localizedDescription)"
        finish(response)
    }
}

run()
