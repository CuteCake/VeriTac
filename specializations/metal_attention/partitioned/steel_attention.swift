// steel_attention.swift
//
// Standalone Metal FP32 causal-prefill attention runner that specializes the
// INSTALLED MLX "steel" attention template (Apple's flash-attention GEMM kernel)
// with small FP32 tiles that fit 32 KiB of threadgroup memory.
//
// IMPORTANT ATTRIBUTION / LICENSE:
//   The Metal kernel executed here is a template-instantiated specialization of
//   the vendor MLX steel attention kernel, which is Copyright (c) 2024-25 Apple
//   Inc. and distributed under the MIT license (see the mlx package headers
//   copied verbatim into the generated kernel source).  This benchmark does NOT
//   claim to be an original generated kernel: it is a resource/schedule
//   specialization experiment of the existing vendor kernel, labeled honestly
//   as vendor-derived.  Attribution and license text are preserved in the
//   generated source.
//
// This Swift host does not embed the kernel: the Python harness flattens the
// installed mlx headers (recursively inlined, keeping system Metal includes)
// into a single MSL source file, appends explicit template instantiations with
// host_names, and passes that file path via the request (`kernel_path`).  The
// host reads it and calls MTLLibrary.makeLibrary(source:) at startup.
//
// Config -> specialization mapping:
//   qt (BQ) -> WM = BQ/8, WN = 1  (BQ = WM*WN*8, TQ == 1 constraint)
//   kt (BK), D (BD)
//   function name = "steel_attn_f32_bq<BQ>_bk<BK>_bd<D>_wm<WM>_wn<WN>"
//   function constants: align_Q (200), align_K (201), has_mask (300),
//                       do_causal (301), has_sinks (302)
//
// Request JSON fields:
//   shape: [B,H,N,D]  (B=1, H=8, D=192|256)
//   dtype: "float32"
//   qt, kt
//   kernel_path: flattened MSL source (vendor-derived instantiations)
//   q/k/v_path, output_path, response_path
//   warmup/samples/inner
//
// Timing: each invocation times the FULL single-kernel pipeline (QK^T + stable
// online softmax + PV + normalize, all in one compute encoder) via
// MTLCommandBuffer gpuStartTime/gpuEndTime deltas and synchronized wall latency.

import Foundation
import Metal
import QuartzCore

private struct Request: Decodable {
    let shape: [Int]
    let dtype: String
    let qt: Int
    let kt: Int
    let kernel_path: String
    let q_path: String
    let k_path: String
    let v_path: String
    let output_path: String
    let response_path: String
    let warmup: Int
    let samples: Int
    let inner: Int
}

private func fail(_ message: String) -> Never {
    fputs("steel_attention: \(message)\n", stderr)
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
    } catch { fail("cannot write \(path): \(error)") }
}

// Layout must match mlx::steel::AttnParams (steel/attn/params.h):
//   14 x int32/float scalars (56 bytes) then 4 x int64[3] stride triplets.
//   56 is 8-aligned, so there is no padding before the int64 arrays.
private struct AttnParams {
    var B: Int32
    var H: Int32
    var D: Int32
    var qL: Int32
    var kL: Int32
    var gqa_factor: Int32
    var scale: Float
    var NQ: Int32
    var NK: Int32
    var NQ_aligned: Int32
    var NK_aligned: Int32
    var qL_rem: Int32
    var kL_rem: Int32
    var qL_off: Int32
    var Q_strides: (Int64, Int64, Int64)
    var K_strides: (Int64, Int64, Int64)
    var V_strides: (Int64, Int64, Int64)
    var O_strides: (Int64, Int64, Int64)
}

private func run() {
    let args = CommandLine.arguments
    guard args.count == 2 else { fail("usage: steel_attention <request.json>") }

    let requestPath = args[1]
    let request: Request
    do {
        request = try JSONDecoder().decode(Request.self, from: readFile(requestPath))
    } catch { fail("cannot decode \(requestPath): \(error)") }

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
            throw NSError(domain: "steel_attention", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shape must be rank 4"])
        }
        let b = request.shape[0], h = request.shape[1], n = request.shape[2], d = request.shape[3]
        guard b == 1 else { throw NSError(domain: "steel_attention", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "B must be 1"]) }
        guard h == 8 else { throw NSError(domain: "steel_attention", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "H must be 8"]) }
        guard n > 0 else { throw NSError(domain: "steel_attention", code: 4,
            userInfo: [NSLocalizedDescriptionKey: "N must be positive"]) }
        guard d == 192 || d == 256 else { throw NSError(domain: "steel_attention", code: 5,
            userInfo: [NSLocalizedDescriptionKey: "D must be 192 or 256"]) }
        let bq = request.qt
        let bk = request.kt
        guard [8, 16, 24].contains(bq) else { throw NSError(domain: "steel_attention", code: 6,
            userInfo: [NSLocalizedDescriptionKey: "qt must be 8, 16, or 24"]) }
        guard bk == 8 || bk == 16 else { throw NSError(domain: "steel_attention", code: 7,
            userInfo: [NSLocalizedDescriptionKey: "kt must be 8 or 16"]) }
        guard request.dtype == "float32" else { throw NSError(domain: "steel_attention", code: 8,
            userInfo: [NSLocalizedDescriptionKey: "dtype must be float32"]) }
        guard request.warmup >= 0, request.samples >= 0, request.inner >= 1 else {
            throw NSError(domain: "steel_attention", code: 9,
                          userInfo: [NSLocalizedDescriptionKey: "warmup/samples >= 0, inner >= 1"])
        }

        let wm = bq / 8
        let wn = 1
        let threadsPerTG = wm * wn * 32
        let kernelName = "steel_attn_f32_bq\(bq)_bk\(bk)_bd\(d)_wm\(wm)_wn\(wn)"

        // Resource formula (vendor header): Q=BQ*(D+4)*4, KV=max((BK+4)*D,BK*(D+4))*4.
        let qBytes = bq * (d + 4) * 4
        let kvBytes = max((bk + 4) * d, bk * (d + 4)) * 4
        let tgBytes = qBytes + kvBytes

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "steel_attention", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "no default Metal device"])
        }
        response["mtl_device_name"] = device.name
        response["mtl_device_registry_id"] = NSNumber(value: device.registryID)
        response["mtl_is_low_power"] = device.isLowPower
        response["mtl_has_unified_memory"] = device.hasUnifiedMemory
        guard tgBytes <= device.maxThreadgroupMemoryLength else {
            throw NSError(domain: "steel_attention", code: 11,
                          userInfo: [NSLocalizedDescriptionKey:
                            "threadgroup memory \(tgBytes) exceeds device max \(device.maxThreadgroupMemoryLength)"])
        }

        let nq = (n + bq - 1) / bq
        let nk = (n + bk - 1) / bk
        let gridX = nq
        let gridY = h
        let gridZ = b
        let totalThreads = gridX * gridY * gridZ * threadsPerTG
        let validRows = b * h * n

        let queue = device.makeCommandQueue()!

        // ---- Read flattened vendor-derived kernel source and compile -------
        let compileStart = CACurrentMediaTime()
        let kernelSrc = String(decoding: try readFile(request.kernel_path), as: UTF8.self)
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSrc, options: nil)
        } catch {
            throw NSError(domain: "steel_attention", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "MSL compile failed: \(error)"])
        }
        let constants = MTLFunctionConstantValues()
        var alignQ = Bool(n % bq == 0)
        constants.setConstantValue(&alignQ, type: .bool, index: 200)
        var alignK = Bool(n % bk == 0)
        constants.setConstantValue(&alignK, type: .bool, index: 201)
        var hasMask = false
        constants.setConstantValue(&hasMask, type: .bool, index: 300)
        var doCausal = true
        constants.setConstantValue(&doCausal, type: .bool, index: 301)
        var hasSinks = false
        constants.setConstantValue(&hasSinks, type: .bool, index: 302)
        guard let fn = try? library.makeFunction(name: kernelName, constantValues: constants),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw NSError(domain: "steel_attention", code: 13,
                          userInfo: [NSLocalizedDescriptionKey:
                            "could not create pipeline for \(kernelName)"])
        }
        guard pipeline.threadExecutionWidth == 32,
              threadsPerTG <= pipeline.maxTotalThreadsPerThreadgroup,
              threadsPerTG <= device.maxThreadsPerThreadgroup.width,
              pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
            throw RunnerError(message: "compiled pipeline limits: SIMD=\(pipeline.threadExecutionWidth), threads=\(threadsPerTG)/\(pipeline.maxTotalThreadsPerThreadgroup), static shared=\(pipeline.staticThreadgroupMemoryLength)/\(device.maxThreadgroupMemoryLength)")
        }
        response["pipeline"] = [
            "max_total_threads_per_threadgroup": NSNumber(value: pipeline.maxTotalThreadsPerThreadgroup),
            "simd_width": NSNumber(value: pipeline.threadExecutionWidth),
            "static_threadgroup_memory_length": NSNumber(value: pipeline.staticThreadgroupMemoryLength),
            "threads_per_threadgroup": threadsPerTG,
            "kernel_name": kernelName,
            "specialization": [
                "qt": bq, "kt": bk, "D": d, "WM": wm, "WN": wn,
                "align_Q": alignQ, "align_K": alignK,
                "has_mask": false, "do_causal": true, "has_sinks": false,
            ],
        ]
        response["compile_wall_ms"] = jsonSafe((CACurrentMediaTime() - compileStart) * 1000.0)

        // ---- Inputs ---------------------------------------------------------
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

        // Contiguous [B,H,N,D] float32 -> strides (B=H*N*D, H=N*D, L=D).
        let sB = Int64(h * n * d)
        let sH = Int64(n * d)
        let sL = Int64(d)
        var params = AttnParams(
            B: Int32(b), H: Int32(h), D: Int32(d),
            qL: Int32(n), kL: Int32(n),
            gqa_factor: 1, scale: Float(1.0 / Double(d).squareRoot()),
            NQ: Int32(nq), NK: Int32(nk),
            NQ_aligned: Int32(nq - 1), NK_aligned: Int32(nk - 1),
            qL_rem: Int32(n - (nq - 1) * bq), kL_rem: Int32(n - (nk - 1) * bk),
            qL_off: 0,
            Q_strides: (sB, sH, sL), K_strides: (sB, sH, sL),
            V_strides: (sB, sH, sL), O_strides: (sB, sH, sL))
        let paramsBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<AttnParams>.stride,
                                          options: .storageModeShared)!

        // ---- Encoded invocation --------------------------------------------
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
            enc.setBuffer(paramsBuf, offset: 0, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: gridX, height: gridY, depth: gridZ),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let t1 = CACurrentMediaTime()
            if cb.status == .error {
                let detail = cb.error?.localizedDescription ?? "unknown command-buffer error"
                throw NSError(domain: "steel_attention", code: 14,
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
        response["qt"] = bq
        response["kt"] = bk
        response["threads_per_threadgroup"] = threadsPerTG
        response["samples"] = request.samples
        response["inner"] = request.inner
        response["scale"] = Double(params.scale)
        response["output_bytes"] = totalBytes
        response["launch"] = [
            "grid": ["x": gridX, "y": gridY, "z": gridZ],
            "mapping": "vendor_steel_small_tile_specialization",
            "qt": bq, "kt": bk, "D": d, "WM": wm, "WN": wn,
            "threads_per_threadgroup": threadsPerTG,
            "total_threads_dispatched": totalThreads,
            "valid_query_rows": validRows,
            "threadgroup_memory_bytes": tgBytes,
            "kernel_name": kernelName,
            "vendor": "mlx-steel-attention",
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
            "GPU times are MTLCommandBuffer gpuStartTime/gpuEndTime deltas per invocation of the FULL single-kernel pipeline (QK^T + stable online softmax + PV + normalize). They exclude input/output CPU copies and compile/warmup.",
            "Wall times are synchronized API latency per invocation.",
            "The kernel is a vendor-derived specialization of the installed mlx steel attention template (Apple Inc., MIT), not an original generated kernel. Attribution/license are preserved in kernel_path.",
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
