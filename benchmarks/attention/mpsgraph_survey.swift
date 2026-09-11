// mpsgraph_survey.swift
//
// Swift helper that runs Apple MPSGraph `scaledDotProductAttention` for the
// causal prefill attention operator on the local Metal device. It is invoked
// by the Python orchestrator (metal_survey.py), one process per (shape, dtype)
// case, and reports raw per-invocation GPU and wall timings plus the output
// buffer bytes.
//
// Usage:
//   xcrun swiftc -O -framework Metal -framework MetalPerformanceShaders \
//       -framework MetalPerformanceShadersGraph mpsgraph_survey.swift \
//       -o mpsgraph_survey
//   ./mpsgraph_survey <request.json>
//
// Request JSON fields:
//   shape:        [B, H, N, D] (rank 4, positive ints)
//   dtype:        "float32" | "float16"
//   q_path:       contiguous binary for Q [B,H,N,D] in `dtype`
//   k_path:       contiguous binary for K [B,H,N,D] in `dtype`
//   v_path:       contiguous binary for V [B,H,N,D] in `dtype`
//   output_path:  where to write the result [B,H,N,D] raw bytes in `dtype`
//   response_path:where to write the metadata / timing JSON
//   warmup:       warm invocation count (outside timing)
//   samples:      timed sample count (each sample runs `inner` invocations)
//   inner:        invocations per timed sample
//
// The graph encodes a single SDPA op with an additive causal mask (0 lower
// triangle, -inf upper triangle), scale D**-0.5, onto an MPSCommandBuffer so
// per-invocation GPUStartTime/GPUEndTime can be read after waitUntilCompleted.
// If those GPU timestamps come back invalid (zero / non-increasing) they are
// reported as unavailable rather than mislabeled.

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
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
    let warmup: Int
    let samples: Int
    let inner: Int
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

private func fail(_ message: String) -> Never {
    fputs("mpsgraph_survey: \(message)\n", stderr)
    exit(2)
}

private func readFile(_ path: String) -> Data {
    do {
        return try Data(contentsOf: URL(fileURLWithPath: path))
    } catch {
        fail("cannot read \(path): \(error)")
    }
}

/// Causal additive mask as raw bytes in the requested dtype: 0 where j <= i
/// (lower triangle including the diagonal), -infinity where j > i.
private func causalMaskData(n: Int, dtype: String) -> Data {
    let count = n * n
    if dtype == "float16" {
        var arr = [Float16](repeating: 0, count: count)
        for i in 0..<n {
            for j in 0..<n where j > i {
                arr[i * n + j] = -Float16.infinity
            }
        }
        return arr.withUnsafeBytes { Data($0) }
    }
    var arr = [Float](repeating: 0, count: count)
    for i in 0..<n {
        for j in 0..<n where j > i {
            arr[i * n + j] = -Float.infinity
        }
    }
    return arr.withUnsafeBytes { Data($0) }
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
// Main
// ---------------------------------------------------------------------------

private func run() {
    let args = CommandLine.arguments
    guard args.count == 2 else {
        fail("usage: mpsgraph_survey <request.json>")
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

    // Default failure response writes even if something below throws.
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
        guard request.shape.count == 4 else {
            throw NSError(domain: "mpsgraph", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shape must be rank 4"])
        }
        let b = request.shape[0]
        let h = request.shape[1]
        let n = request.shape[2]
        let d = request.shape[3]
        guard b > 0, h > 0, n > 0, d > 0 else {
            throw NSError(domain: "mpsgraph", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "shape entries must be positive"])
        }

        let isHalf = (request.dtype == "float16")
        guard isHalf || request.dtype == "float32" else {
            throw NSError(domain: "mpsgraph", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "unsupported dtype \(request.dtype)"])
        }
        let dataType: MPSDataType = isHalf ? .float16 : .float32
        let elemSize = isHalf ? 2 : 4
        let totalBytes = b * h * n * d * elemSize

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "mpsgraph", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "no default Metal device"])
        }
        response["mtl_device_name"] = device.name
        response["mtl_device_registry_id"] = NSNumber(value: device.registryID)
        response["mtl_is_low_power"] = device.isLowPower
        response["mtl_has_unified_memory"] = device.hasUnifiedMemory
        // Device limits useful for later scheduling design.
        response["mtl_max_threadgroup_memory_length"] = NSNumber(
            value: device.maxThreadgroupMemoryLength)
        let maxTG = device.maxThreadsPerThreadgroup
        response["mtl_max_threads_per_threadgroup"] = [
            "width": NSNumber(value: maxTG.width),
            "height": NSNumber(value: maxTG.height),
            "depth": NSNumber(value: maxTG.depth),
        ]


        let queue = device.makeCommandQueue()!

        // ---- Build the graph -------------------------------------------------
        let graph = MPSGraph()
        let shapeNS: [NSNumber] = [b, h, n, d].map { NSNumber(value: $0) }
        let qPh = graph.placeholder(shape: shapeNS, dataType: dataType, name: "q")
        let kPh = graph.placeholder(shape: shapeNS, dataType: dataType, name: "k")
        let vPh = graph.placeholder(shape: shapeNS, dataType: dataType, name: "v")

        let scale = Float(1.0 / Double(d).squareRoot())  // D**-0.5
        let maskShape: [NSNumber] = [n, n].map { NSNumber(value: $0) }
        let mask = graph.constant(causalMaskData(n: n, dtype: request.dtype),
                                  shape: maskShape, dataType: dataType)
        let outTensor = graph.scaledDotProductAttention(query: qPh, key: kPh,
                                                        value: vPh, mask: mask,
                                                        scale: scale, name: "sdpa")

        // ---- Inputs ----------------------------------------------------------
        // Preallocated shared-storage MTLBuffers so the graph reads the data
        // directly from unified memory (no per-invocation CPU copies in timing).
        func inputBuffer(_ path: String) -> MTLBuffer {
            let data = readFile(path)
            guard data.count == totalBytes else {
                fail("input \(path) has \(data.count) bytes, expected \(totalBytes)")
            }
            let buf = device.makeBuffer(length: totalBytes,
                                        options: .storageModeShared)!
            data.withUnsafeBytes { raw in
                buf.contents().copyMemory(from: raw.baseAddress!, byteCount: totalBytes)
            }
            return buf
        }

        let qBuf = inputBuffer(request.q_path)
        let kBuf = inputBuffer(request.k_path)
        let vBuf = inputBuffer(request.v_path)
        let outBuf = device.makeBuffer(length: totalBytes,
                                       options: .storageModeShared)!

        let qData = MPSGraphTensorData(qBuf, shape: shapeNS, dataType: dataType)
        let kData = MPSGraphTensorData(kBuf, shape: shapeNS, dataType: dataType)
        let vData = MPSGraphTensorData(vBuf, shape: shapeNS, dataType: dataType)
        let outData = MPSGraphTensorData(outBuf, shape: shapeNS, dataType: dataType)

        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            qPh: qData, kPh: kData, vPh: vData,
        ]

        // ---- Encode / run ----------------------------------------------------
        // Fresh MTLCommandBuffer per invocation. Encoded onto an MPSCommandBuffer
        // so GPUStartTime/GPUEndTime are available after waitUntilCompleted.
        // Wall timing starts BEFORE creating the command buffers so the measured
        // API latency includes command-buffer allocation/encoding. All durations
        // are converted to milliseconds before being written to JSON.
        func runOnce() throws -> (wall: Double, gpuStart: Double, gpuEnd: Double) {
            let t0 = CACurrentMediaTime()
            let mtlCB = queue.makeCommandBuffer()!
            let mpsCB = MPSCommandBuffer(commandBuffer: mtlCB)
            graph.encode(to: mpsCB, feeds: feeds, targetOperations: nil,
                         resultsDictionary: [outTensor: outData],
                         executionDescriptor: nil)
            mpsCB.commit()
            mpsCB.waitUntilCompleted()
            let t1 = CACurrentMediaTime()
            // Fail on a GPU error rather than reporting a bogus timing.
            if mpsCB.status == .error {
                let detail = mpsCB.error?.localizedDescription ?? "unknown GPU error"
                throw NSError(domain: "mpsgraph", code: 5,
                              userInfo: [NSLocalizedDescriptionKey: "GPU error: \(detail)"])
            }
            return ((t1 - t0) * 1000.0,
                    mpsCB.gpuStartTime * 1000.0,
                    mpsCB.gpuEndTime * 1000.0)
        }

        // Warmup / compile outside the timed region.
        let compileStart = CACurrentMediaTime()
        let warmupCount = max(request.warmup, 0)
        for _ in 0..<warmupCount {
            _ = try runOnce()
        }
        response["compile_wall_ms"] = jsonSafe((CACurrentMediaTime() - compileStart) * 1000.0)
        response["warmup_count"] = warmupCount

        // Timed samples: each sample runs `inner` invocations.
        var wallTimes: [Double] = []
        var gpuTimes: [Double] = []
        var gpuValidCount = 0
        let totalRuns = request.samples * request.inner
        for _ in 0..<request.samples {
            for _ in 0..<request.inner {
                let r = try runOnce()
                wallTimes.append(r.wall)
                if r.gpuStart > 0, r.gpuEnd > 0, r.gpuEnd >= r.gpuStart {
                    gpuTimes.append(r.gpuEnd - r.gpuStart)
                    gpuValidCount += 1
                }
            }
        }
        let gpuAvailable = (gpuValidCount == totalRuns)
        response["gpu_time_available"] = gpuAvailable
        response["gpu_times"] = gpuTimes.map { jsonSafe($0) }
        response["wall_times"] = wallTimes.map { jsonSafe($0) }
        response["time_unit"] = "milliseconds"
        response["gpu_timestamp_units"] = "milliseconds"

        // ---- Output ----------------------------------------------------------
        let outRaw = Data(bytes: outBuf.contents(), count: totalBytes)
        try outRaw.write(to: URL(fileURLWithPath: request.output_path))

        response["status"] = "ok"
        response["shape"] = request.shape
        response["dtype"] = request.dtype
        response["samples"] = request.samples
        response["inner"] = request.inner
        response["scale"] = Double(scale)
        response["mask"] = "causal_additive"
        response["output_bytes"] = totalBytes

        // Framework / OS identity for provenance.
        response["os"] = [
            "version": ProcessInfo.processInfo.operatingSystemVersionString,
            "product": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        response["mpsgraph_api"] = "scaledDotProductAttention(query:key:value:mask:scale:name:)"
        response["measurement_notes"] = [
            "All timings are in milliseconds (time_unit=milliseconds).",
            "GPU times are MPSCommandBuffer GPUStartTime/GPUEndTime deltas per invocation; they exclude input/output CPU copies (preallocated shared MTLBuffers are reused) and exclude compile/warmup.",
            "Wall times are synchronized API latency, timed from before command-buffer allocation through waitUntilCompleted, per invocation.",
            "Each invocation encodes a fresh command buffer and re-executes the graph; outputs are never cached across samples.",
            "An invocation that ends with a Metal command-buffer error fails the case rather than reporting a bogus timing.",
        ]

        finish(response)
    } catch {
        response["status"] = "error"
        response["error"] = "\(type(of: error)): \(error.localizedDescription)"
        finish(response)
    }
}

run()
