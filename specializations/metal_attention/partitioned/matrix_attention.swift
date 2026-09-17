// matrix_attention.swift
//
// Standalone Metal FP32 causal-prefill attention kernel using the simdgroup
// matrix (MMA) instruction path.  Original FP32 implementation throughout --
// no fp16, no approximate or lower-precision substitution; FP32 contract
// atol=1e-4 rtol=1e-3 frozen.
//
// Design (exact-real, two-pass so the softmax needs no per-tile rescaling):
//
//   One 32-lane SIMD-group owns a tile of BM=8 query rows.  The causal key
//   range for each row is walked in tiles of BN keys (8 or 16).  Each key tile
//   is a matrix product computed with simdgroup_float8x8 fragments:
//
//     QK^T :  S[BM x BN] = sum over 8-wide D-slabs of Q[BM x 8] * K^T[8 x BN]
//     PV   :  O[BM x D ] += P[BM x BN] * V[BN x D]
//
//   Two passes over the key tiles avoid all per-tile online-softmax rescaling:
//     * Pass 1 computes each row's global max m[i] (causal-masked) from S.
//     * Pass 2 recomputes S, forms P = exp(S - m[i]) (causal-masked), and
//       accumulates P*V into O fragments with a fixed m -- so O is never
//       rescaled and the softmax is exactly stable.
//   Finally O is stored to threadgroup scratch (reusing the K/V scratch whose
//   lifetime has ended), normalized by the per-row l, and written to device.
//
//   The softmax is done by storing/loading score fragments through small
//   threadgroup scratch (simdgroup_store / simdgroup_load), never by depending
//   on the fragment lane layout.  O is kept as per-lane float2 fragment
//   elements and converted to simdgroup_float8x8 only at the MMA/store points.
//
//   Causal masking and tails: rows with row0+i >= N are not written; key
//   columns kk >= N are padded with 0 in the staged K/V; for row i only keys
//   kk <= row0+i are counted (masked in both the max and the P sums), so every
//   valid key is counted exactly once and no out-of-causal key contributes.
//
// Threadgroup memory (disjoint lifetimes, fits 32 KiB for D=256, BN=16):
//     Qtg [BM*D], KV [BN*D] (reused for K tile, V tile, then final O scratch),
//     Stg [BM*BN], Mtg [BM], Ltg [BM].
//
// D and BN are function constants specialized at pipeline build (compile time).
//
// Usage:
//   xcrun swiftc -O -framework Metal -framework MetalPerformanceShaders \
//       matrix_attention.swift -o matrix_attention
//   ./matrix_attention <request.json>
//
// Request JSON fields:
//   shape: [B, H, N, D]  (B=1, H=8, D=192|256)
//   dtype: "float32"
//   q/k/v_path, output_path, response_path
//   qt: query rows per simdgroup (must be 8 in this single-simdgroup version)
//   kt: key tile width BN (8 or 16)
//   warmup/samples/inner
//
// Timing: each invocation times the FULL single-kernel two-pass pipeline via
// MTLCommandBuffer gpuStartTime/gpuEndTime deltas and synchronized wall
// latency.  Nothing is cached across samples.

import Foundation
import Metal
import QuartzCore

private struct Request: Decodable {
    let shape: [Int]
    let dtype: String
    let q_path: String
    let k_path: String
    let v_path: String
    let output_path: String
    let response_path: String
    let qt: Int
    let kt: Int
    let warmup: Int
    let samples: Int
    let inner: Int
}

private func fail(_ message: String) -> Never {
    fputs("matrix_attention: \(message)\n", stderr)
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
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint D_FC [[function_constant(0)]];
constant uint BN_FC [[function_constant(1)]];

constant uint BM = 8;        // query rows per simdgroup (one simdgroup / threadgroup)
constant uint SIMD = 32;
constant uint MAXO = 32;     // D/8 max (D=256)
constant uint MAXBNF = 2;    // BN/8 max (BN=16)

struct KernelParams {
    uint N;
    uint H;
    uint B;
    float scale;
};

// Convert a per-lane float2 fragment element to a simdgroup_float8x8.
static inline simdgroup_float8x8 mat_from_f2(float2 v) {
    simdgroup_float8x8 m;
    m.thread_elements()[0] = v.x;
    m.thread_elements()[1] = v.y;
    return m;
}

kernel void matrix_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant KernelParams& params [[buffer(10)]],
    threadgroup float* tg [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tptg_in [[thread_position_in_threadgroup]])
{
    const uint N = params.N;
    const uint H = params.H;
    const uint D = D_FC;
    const uint BN = BN_FC;
    const float scale = params.scale;
    const uint lane = tptg_in.x;

    const uint row0 = tgid.x * BM;           // first query row of this block
    const uint h = tgid.y;                   // head plane (B=1)
    const uint qdevbase = (h * N + row0) * D;
    const uint kvbase = h * N * D;

    threadgroup float* Qtg = tg;
    threadgroup float* KV  = tg + BM * D;
    threadgroup float* Stg = tg + BM * D + BN * D;
    threadgroup float* Mtg = Stg + BM * BN;
    threadgroup float* Ltg = Mtg + BM;

    // Tiles wholly beyond the last query row are causally masked for every
    // row in this group, so neither pass needs to load or multiply them.
    const uint nTiles = (min(N, row0 + BM) + BN - 1u) / BN;

    // ---- Load Q tile [BM x D] into threadgroup; init streaming state ----
    for (uint e = lane; e < BM * D; e += SIMD) {
        const uint r = e / D;
        const uint d = e % D;
        const uint rr = row0 + r;
        Qtg[e] = (rr < N) ? Q[qdevbase + r * D + d] : 0.0f;
    }
    if (lane < BM) { Mtg[lane] = -INFINITY; Ltg[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- O accumulator: per-lane float2 for each D/8 column-block ----
    float2 Ofrag[MAXO];
    for (uint cb = 0; cb < D / 8; ++cb) { Ofrag[cb] = float2(0.0f, 0.0f); }

    // ================= PHASE 1: per-row global max ========================
    for (uint kb = 0; kb < nTiles; ++kb) {
        const uint kt = kb * BN;
        for (uint e = lane; e < BN * D; e += SIMD) {
            const uint c = e / D;
            const uint d = e % D;
            const uint kk = kt + c;
            KV[e] = (kk < N) ? K[kvbase + kk * D + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q * K^T [BM x BN]
        float2 Sfrag[MAXBNF];
        for (uint j = 0; j < BN / 8; ++j) { Sfrag[j] = float2(0.0f, 0.0f); }
        for (uint db = 0; db < D; db += 8) {
            simdgroup_float8x8 A, B;
            simdgroup_load(A, &Qtg[db], D);
            for (uint j = 0; j < BN / 8; ++j) {
                simdgroup_load(B, &KV[j * 8 * D + db], D, ulong2(0), true); // K^T
                simdgroup_float8x8 Sm = mat_from_f2(Sfrag[j]);
                simdgroup_multiply_accumulate(Sm, A, B, Sm);
                Sfrag[j].x = Sm.thread_elements()[0];
                Sfrag[j].y = Sm.thread_elements()[1];
            }
        }
        for (uint j = 0; j < BN / 8; ++j) { Sfrag[j] *= scale; }
        for (uint j = 0; j < BN / 8; ++j) {
            simdgroup_store(mat_from_f2(Sfrag[j]), &Stg[j * 8], BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // causal row max over this tile
        if (lane < BM) {
            const uint i = lane;
            float tmax = -INFINITY;
            for (uint c = 0; c < BN; ++c) {
                const uint kk = kt + c;
                if (kk <= row0 + i) { tmax = max(tmax, Stg[i * BN + c]); }
            }
            Mtg[i] = max(Mtg[i], tmax);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ================= PHASE 2: causal weighted sum =======================
    for (uint kb = 0; kb < nTiles; ++kb) {
        const uint kt = kb * BN;
        for (uint e = lane; e < BN * D; e += SIMD) {
            const uint c = e / D;
            const uint d = e % D;
            const uint kk = kt + c;
            KV[e] = (kk < N) ? K[kvbase + kk * D + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q * K^T [BM x BN]
        float2 Sfrag[MAXBNF];
        for (uint j = 0; j < BN / 8; ++j) { Sfrag[j] = float2(0.0f, 0.0f); }
        for (uint db = 0; db < D; db += 8) {
            simdgroup_float8x8 A, B;
            simdgroup_load(A, &Qtg[db], D);
            for (uint j = 0; j < BN / 8; ++j) {
                simdgroup_load(B, &KV[j * 8 * D + db], D, ulong2(0), true);
                simdgroup_float8x8 Sm = mat_from_f2(Sfrag[j]);
                simdgroup_multiply_accumulate(Sm, A, B, Sm);
                Sfrag[j].x = Sm.thread_elements()[0];
                Sfrag[j].y = Sm.thread_elements()[1];
            }
        }
        for (uint j = 0; j < BN / 8; ++j) { Sfrag[j] *= scale; }
        for (uint j = 0; j < BN / 8; ++j) {
            simdgroup_store(mat_from_f2(Sfrag[j]), &Stg[j * 8], BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V tile into KV (K no longer needed) and form P in Stg.
        for (uint e = lane; e < BN * D; e += SIMD) {
            const uint c = e / D;
            const uint d = e % D;
            const uint kk = kt + c;
            KV[e] = (kk < N) ? V[kvbase + kk * D + d] : 0.0f;
        }
        if (lane < BM) {
            const uint i = lane;
            const float mi = Mtg[i];
            float lsum = 0.0f;
            for (uint c = 0; c < BN; ++c) {
                const uint kk = kt + c;
                if (kk <= row0 + i) {
                    const float p = exp(Stg[i * BN + c] - mi);
                    Stg[i * BN + c] = p;
                    lsum += p;
                } else {
                    Stg[i * BN + c] = 0.0f;
                }
            }
            Ltg[i] += lsum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P * V
        for (uint cb = 0; cb < D / 8; ++cb) {
            simdgroup_float8x8 Omat = mat_from_f2(Ofrag[cb]);
            for (uint j = 0; j < BN / 8; ++j) {
                simdgroup_float8x8 Pmat, Vmat;
                simdgroup_load(Pmat, &Stg[j * 8], BN);
                simdgroup_load(Vmat, &KV[j * 8 * D + cb * 8], D);
                simdgroup_multiply_accumulate(Omat, Pmat, Vmat, Omat);
            }
            Ofrag[cb].x = Omat.thread_elements()[0];
            Ofrag[cb].y = Omat.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ================= Finalize: normalize and write device ===============
    // KV scratch is now free; use it to hold the raw O tile for row-wise
    // normalization (layout-agnostic; avoids depending on fragment layout).
    for (uint cb = 0; cb < D / 8; ++cb) {
        simdgroup_store(mat_from_f2(Ofrag[cb]), &KV[cb * 8], D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < BM) {
        const uint i = lane;
        const uint rr = row0 + i;
        if (rr < N) {
            const float invL = 1.0f / Ltg[i];
            device float* orow = O + (h * N + rr) * D;
            for (uint d = 0; d < D; ++d) { orow[d] = KV[i * D + d] * invL; }
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
    guard args.count == 2 else { fail("usage: matrix_attention <request.json>") }

    let requestPath = args[1]
    let request: Request
    do {
        request = try JSONDecoder().decode(Request.self, from: readFile(requestPath))
    } catch {
        fail("cannot decode \(requestPath): \(error)")
    }

    var response: [String: Any] = ["status": "error"]
    let finish = { (payload: [String: Any]) in
        do {
            let data = try JSONSerialization.data(withJSONObject: payload,
                                                  options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: request.response_path))
        } catch { fail("cannot write response: \(error)") }
    }

    do {
        guard request.shape.count == 4 else {
            throw NSError(domain: "matrix_attention", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shape must be rank 4"])
        }
        let b = request.shape[0], h = request.shape[1], n = request.shape[2], d = request.shape[3]
        guard b == 1 else { throw NSError(domain: "matrix_attention", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "B must be 1"]) }
        guard h == 8 else { throw NSError(domain: "matrix_attention", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "H must be 8"]) }
        guard n > 0 else { throw NSError(domain: "matrix_attention", code: 4,
            userInfo: [NSLocalizedDescriptionKey: "N must be positive"]) }
        guard d == 192 || d == 256 else { throw NSError(domain: "matrix_attention", code: 5,
            userInfo: [NSLocalizedDescriptionKey: "D must be 192 or 256"]) }
        guard request.qt == 8 else { throw NSError(domain: "matrix_attention", code: 6,
            userInfo: [NSLocalizedDescriptionKey: "qt must be 8 (single-simdgroup version)"]) }
        guard request.kt == 8 || request.kt == 16 else { throw NSError(domain: "matrix_attention", code: 7,
            userInfo: [NSLocalizedDescriptionKey: "kt must be 8 or 16"]) }
        guard request.dtype == "float32" else { throw NSError(domain: "matrix_attention", code: 8,
            userInfo: [NSLocalizedDescriptionKey: "dtype must be float32"]) }
        guard request.warmup >= 0, request.samples >= 0, request.inner >= 1 else {
            throw NSError(domain: "matrix_attention", code: 9,
                          userInfo: [NSLocalizedDescriptionKey: "warmup/samples >= 0, inner >= 1"])
        }

        let threadsPerTG = 32
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "matrix_attention", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "no default Metal device"])
        }
        response["mtl_device_name"] = device.name
        response["mtl_device_registry_id"] = NSNumber(value: device.registryID)
        response["mtl_is_low_power"] = device.isLowPower
        response["mtl_has_unified_memory"] = device.hasUnifiedMemory

        // Threadgroup memory: BM*D + BN*D + BM*BN + 2*BM floats.
        let tgMemBytes = (8 * d + request.kt * d + 8 * request.kt + 16) * 4
        guard tgMemBytes <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "matrix_attention", code: 11,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgMemBytes) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }

        let gridX = (n + 7) / 8          // 8 query rows per threadgroup
        let gridY = b * h                // = 8
        let totalThreads = gridX * gridY * threadsPerTG
        let validRows = b * h * n

        let queue = device.makeCommandQueue()!

        let compileStart = CACurrentMediaTime()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw NSError(domain: "matrix_attention", code: 13,
                          userInfo: [NSLocalizedDescriptionKey: "MSL compile failed: \(error)"])
        }
        let constants = MTLFunctionConstantValues()
        var dVal = UInt32(d)
        constants.setConstantValue(&dVal, type: .uint, index: 0)
        var bnVal = UInt32(request.kt)
        constants.setConstantValue(&bnVal, type: .uint, index: 1)
        guard let fn = try? library.makeFunction(name: "matrix_attention", constantValues: constants),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw NSError(domain: "matrix_attention", code: 14,
                          userInfo: [NSLocalizedDescriptionKey:
                            "could not create specialized pipeline for D=\(d) kt=\(request.kt)"])
        }
        let maxTotalTG = pipeline.maxTotalThreadsPerThreadgroup
        let staticTG = pipeline.staticThreadgroupMemoryLength
        let simdWidth = pipeline.threadExecutionWidth
        guard threadsPerTG <= maxTotalTG else {
            throw NSError(domain: "matrix_attention", code: 15,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threads per threadgroup exceeds compiled maxTotalThreadsPerThreadgroup \(maxTotalTG)"])
        }
        guard simdWidth == 32 else {
            throw NSError(domain: "matrix_attention", code: 16,
                          userInfo: [NSLocalizedDescriptionKey:
                            "pipeline SIMD width must be 32, got \(simdWidth)"])
        }
        response["pipeline"] = [
            "max_total_threads_per_threadgroup": NSNumber(value: maxTotalTG),
            "simd_width": NSNumber(value: simdWidth),
            "static_threadgroup_memory_length": NSNumber(value: staticTG),
            "dynamic_threadgroup_memory_length": tgMemBytes,
            "specialization": ["D": d, "kt": request.kt, "qt": 8],
        ]
        response["compile_wall_ms"] = jsonSafe((CACurrentMediaTime() - compileStart) * 1000.0)

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
                                     threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let t1 = CACurrentMediaTime()
            if cb.status == .error {
                let detail = cb.error?.localizedDescription ?? "unknown command-buffer error"
                throw NSError(domain: "matrix_attention", code: 17,
                              userInfo: [NSLocalizedDescriptionKey: "GPU error: \(detail)"])
            }
            return ((t1 - t0) * 1000.0,
                    cb.gpuStartTime * 1000.0,
                    cb.gpuEndTime * 1000.0,
                    cb.kernelStartTime * 1000.0,
                    cb.kernelEndTime * 1000.0)
        }

        let warmupCount = max(request.warmup, 0)
        for _ in 0..<warmupCount { _ = try runOnce() }
        response["warmup_count"] = warmupCount

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

        let outRaw = Data(bytes: outBuf.contents(), count: totalBytes)
        try outRaw.write(to: URL(fileURLWithPath: request.output_path))

        response["status"] = "ok"
        response["shape"] = request.shape
        response["dtype"] = request.dtype
        response["qt"] = 8
        response["kt"] = request.kt
        response["threads_per_threadgroup"] = threadsPerTG
        response["samples"] = request.samples
        response["inner"] = request.inner
        response["scale"] = Double(params.scale)
        response["output_bytes"] = totalBytes
        response["launch"] = [
            "grid": ["x": gridX, "y": gridY, "z": 1],
            "mapping": "one_simdgroup_per_8_query_rows",
            "query_rows_per_simdgroup": 8,
            "key_tile": request.kt,
            "lanes_per_simdgroup": 32,
            "threads_per_threadgroup": threadsPerTG,
            "total_threads_dispatched": totalThreads,
            "valid_query_rows": validRows,
            "dynamic_threadgroup_memory_bytes": tgMemBytes,
            "static_threadgroup_memory_bytes": staticTG,
            "kernel_name": "matrix_attention",
            "matrix_units": [
                "qk_contraction": "simdgroup_float8x8 over 8-wide D slabs",
                "pv": "simdgroup_float8x8 over D/8 column blocks",
                "dtype": "float32",
            ],
            "passes": ["pass1_row_max", "pass2_causal_weighted_sum", "finalize_normalize"],
            "no_n_by_n_buffer": true,
        ]
        response["device"] = [
            "max_threadgroup_memory_length": NSNumber(value: device.maxThreadgroupMemoryLength),
            "max_threads_per_threadgroup": [
                "width": NSNumber(value: device.maxThreadsPerThreadgroup.width),
                "height": NSNumber(value: device.maxThreadsPerThreadgroup.height),
                "depth": NSNumber(value: device.maxThreadsPerThreadgroup.depth),
            ],
        ]
        response["os"] = ["version": ProcessInfo.processInfo.operatingSystemVersionString]
        response["measurement_notes"] = [
            "All timings are in milliseconds (time_unit=milliseconds).",
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per invocation of the FULL single-kernel two-pass pipeline (pass1 max, pass2 weighted sum, finalize). They exclude input/output CPU copies and compile/warmup.",
            "Wall times are synchronized API latency per invocation.",
            "The Metal shader is compiled once at startup with function-constant specialization for (D, kt) and reused across all invocations.",
            "FP32 matrices throughout; the softmax is done via threadgroup scratch, not fragment-layout dependence.",
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
