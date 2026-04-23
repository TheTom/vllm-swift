// SPDX-License-Identifier: Apache-2.0
// vllm-swift C bridge implementation
//
// Wraps mlx-swift-lm's TokenIterator to expose a C API for Python ctypes.
// All GPU compute stays here in Swift/Metal — Python only drives scheduling.

import Foundation
import MLX
import MLXNN
import MLXLMCommon
@_exported import MLXLLM
// HuggingFace macros require the full HF SDK. For now, load models
// from local directories (Python downloads via huggingface_hub first).

// MARK: - Stub tokenizer (Python handles tokenization)

/// Minimal tokenizer that satisfies the protocol. All actual tokenization
/// happens in Python via HuggingFace transformers. Swift only needs token
/// IDs for model forward passes.
struct StubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        StubTokenizer()
    }
}

private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { String($0) }.joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - Engine state

/// Per-request session state (KV cache + iterator).
struct RequestSession {
    var iterator: TokenIterator
    var temperature: Float
    var topP: Float
}

/// Holds model + all active request sessions.
final class InferenceEngine {
    let model: any LanguageModel
    let tokenizer: any Tokenizer
    let configuration: ModelConfiguration

    /// Active sessions keyed by request ID (supports concurrent requests)
    var sessions: [String: RequestSession] = [:]
    var generateParams: GenerateParameters

    /// Batched KV caches: one per layer, shared across all requests.
    /// Used by fullyBatchedDecode when model is Qwen3.
    var batchedCaches: [BatchedKVCache]?
    /// Maps request ID → batch slot index in batchedCaches.
    var batchSlots: [String: Int] = [:]
    /// Last token per batch slot for batched decode.
    var batchTokens: [Int] = []

    // Perf tracking
    var prefillTokensPerSec: Double = 0
    var totalDecodeTokens: Int32 = 0
    var totalDecodeTime: Double = 0
    var peakMemoryBytes: Int64 = 0

    init(
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        params: GenerateParameters
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.generateParams = params
    }
}

// Engine storage — nonisolated(unsafe) silences Swift 6 concurrency
// checker. Actual thread safety provided by engineQueue.
nonisolated(unsafe) private var engines: [UnsafeMutableRawPointer: InferenceEngine] = [:]
private let engineQueue = DispatchQueue(label: "vsm.engine.queue")

// MARK: - C API implementations

@_cdecl("vsm_engine_create")
public func vsm_engine_create(
    modelPath: UnsafePointer<CChar>?,
    dtype: UnsafePointer<CChar>?,
    maxKVSize: Int32,
    kvScheme: UnsafePointer<CChar>?,
    kvBits: Int32,
    memoryFraction: Float
) -> UnsafeMutableRawPointer? {
    guard let modelPath else { return nil }
    let modelId = String(cString: modelPath)

    // Build generation parameters
    var params = GenerateParameters()
    if maxKVSize > 0 {
        params.maxKVSize = Int(maxKVSize)
    }
    if let kvScheme {
        params.kvScheme = String(cString: kvScheme)
    }
    if kvBits > 0 {
        params.kvBits = Int(kvBits)
    }
    params.temperature = 0  // default greedy, overridden per-call

    // Set memory limit
    if memoryFraction > 0 && memoryFraction < 1 {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let limit = Int(Double(totalMemory) * Double(memoryFraction))
        Memory.cacheLimit = limit
    }

    // Load model synchronously (nonisolated(unsafe) for C bridge context)
    nonisolated(unsafe) var loadedContext: ModelContext?
    nonisolated(unsafe) var loadError: (any Error)?
    let semaphore = DispatchSemaphore(value: 0)

    Task {
        do {
            // Load from local directory. Python is responsible for
            // downloading via huggingface_hub before calling create.
            // modelId can be a HF cache path or direct directory.
            // Load model from local directory. Python downloads via
            // huggingface_hub and passes the cache path.
            let modelURL = URL(fileURLWithPath: modelId)
            let context = try await loadModel(
                from: modelURL,
                using: StubTokenizerLoader()
            )
            loadedContext = context
        } catch {
            loadError = error
        }
        semaphore.signal()
    }
    semaphore.wait()

    guard let context = loadedContext else {
        let errMsg = loadError?.localizedDescription ?? "unknown"
        print("[vsm] Failed to load model \(modelId): \(errMsg)")
        return nil
    }

    let engine = InferenceEngine(
        model: context.model,
        tokenizer: context.tokenizer,
        configuration: ModelConfiguration(id: modelId),
        params: params
    )

    // Create stable pointer as opaque handle
    let ptr = Unmanaged.passRetained(engine).toOpaque()
    let handle = UnsafeMutableRawPointer(ptr)
    engineQueue.sync { engines[handle] = engine }

    print("[vsm] Engine created: \(modelId)")
    return handle
}

@_cdecl("vsm_engine_destroy")
public func vsm_engine_destroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    engineQueue.sync {
        if engines.removeValue(forKey: handle) != nil {
            Unmanaged<InferenceEngine>.fromOpaque(handle).release()
        }
    }
}

@_cdecl("vsm_engine_vocab_size")
public func vsm_engine_vocab_size(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }
    return engineQueue.sync {
        // TODO: get actual vocab size from model config
        // Tokenizer protocol doesn't expose vocab size directly
        return Int32(0)
    }
}

@_cdecl("vsm_engine_num_layers")
public func vsm_engine_num_layers(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }
    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(0) }
        // Count layers from model parameters
        let params = engine.model.parameters()
        let layerIndices = Set(params.keys.compactMap { key -> Int? in
            guard let range = key.range(of: "layers.") else { return nil }
            let after = key[range.upperBound...]
            guard let dotIdx = after.firstIndex(of: ".") else { return nil }
            return Int(after[..<dotIdx])
        })
        return Int32(layerIndices.count)
    }
}

@_cdecl("vsm_engine_head_dim")
public func vsm_engine_head_dim(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    // TODO: extract from model config
    return 128
}

@_cdecl("vsm_engine_model_memory_bytes")
public func vsm_engine_model_memory_bytes(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    return Int64(Memory.activeMemory)
}

// Single-request API (backward compat — uses "_default" session)

@_cdecl("vsm_engine_prefill")
public func vsm_engine_prefill(
    _ handle: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>?,
    numTokens: Int32,
    temperature: Float,
    topP: Float
) -> Int32 {
    return vsm_engine_prefill_req(
        handle, reqId: "_default",
        promptTokens: promptTokens, numTokens: numTokens,
        temperature: temperature, topP: topP
    )
}

@_cdecl("vsm_engine_decode_step")
public func vsm_engine_decode_step(
    _ handle: UnsafeMutableRawPointer?,
    temperature: Float,
    topP: Float
) -> Int32 {
    return vsm_engine_decode_step_req(handle, reqId: "_default")
}

// Multi-request API

@_cdecl("vsm_engine_prefill_req")
public func vsm_engine_prefill_req(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?,
    promptTokens: UnsafePointer<Int32>?,
    numTokens: Int32,
    temperature: Float,
    topP: Float
) -> Int32 {
    guard let handle, let promptTokens, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(-1) }

        let tokens = (0..<Int(numTokens)).map { Int(promptTokens[$0]) }
        let tokenArray = MLXArray(tokens)

        var params = engine.generateParams
        params.temperature = temperature
        params.topP = topP

        do {
            let input = LMInput(text: .init(tokens: tokenArray))
            var iterator = try TokenIterator(
                input: input,
                model: engine.model,
                parameters: params
            )

            guard let firstToken = iterator.next() else {
                return Int32(-1)
            }

            engine.sessions[rid] = RequestSession(
                iterator: iterator,
                temperature: temperature,
                topP: topP
            )
            return Int32(firstToken)
        } catch {
            print("[vsm] Prefill error for \(rid): \(error)")
            return Int32(-1)
        }
    }
}

@_cdecl("vsm_engine_decode_step_req")
public func vsm_engine_decode_step_req(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?
) -> Int32 {
    guard let handle, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync {
        guard let engine = engines[handle],
              var session = engine.sessions[rid] else { return Int32(-1) }

        let start = CFAbsoluteTimeGetCurrent()
        guard let token = session.iterator.next() else {
            return Int32(-1)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        engine.sessions[rid] = session
        engine.totalDecodeTokens += 1
        engine.totalDecodeTime += elapsed
        engine.peakMemoryBytes = max(
            engine.peakMemoryBytes,
            Int64(Memory.peakMemory)
        )

        return Int32(token)
    }
}

/// Batch decode: all active sessions in one batched forward pass.
/// Projections + MLP batched across B requests, attention per-request.
@_cdecl("vsm_engine_decode_all")
public func vsm_engine_decode_all(
    _ handle: UnsafeMutableRawPointer?,
    reqIds: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    outTokens: UnsafeMutablePointer<Int32>?,
    maxReqs: Int32
) -> Int32 {
    guard let handle, let reqIds, let outTokens else { return 0 }

    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(0) }

        let start = CFAbsoluteTimeGetCurrent()
        let rids = Array(engine.sessions.keys.prefix(Int(maxReqs)))
        guard !rids.isEmpty else { return Int32(0) }

        // Fully batched path for Qwen3 with BatchedKVCache
        if let qwenModel = engine.model as? Qwen3Model,
           let bCaches = engine.batchedCaches,
           !engine.batchSlots.isEmpty
        {
            let B = engine.batchSlots.count
            let tokens = engine.batchTokens

            // Single batched forward: [B, 1] → [B, 1, vocab]
            let inputBatch = MLXArray(tokens[0..<B]).reshaped(B, 1)
            let logitsBatch = qwenModel.fullyBatchedDecode(inputBatch, caches: bCaches)

            let lastLogits = logitsBatch[0..., -1, 0...]  // [B, vocab]

            // Check if all requests use greedy sampling
            let sortedSlots = engine.batchSlots.sorted { $0.value < $1.value }
            let allGreedy = sortedSlots.allSatisfy { (rid, _) in
                engine.sessions[rid]?.temperature == 0
            }

            var count: Int32 = 0
            if allGreedy {
                // Fast path: batched argmax
                let sampledTokens = argMax(lastLogits, axis: -1)
                eval(sampledTokens)
                for (rid, slotIdx) in sortedSlots {
                    let tokenId = sampledTokens[slotIdx].item(Int.self)
                    engine.batchTokens[slotIdx] = tokenId
                    reqIds[Int(count)] = strdup(rid)
                    outTokens[Int(count)] = Int32(tokenId)
                    count += 1
                }
            } else {
                // Per-request sampling with temperature/top_p
                for (rid, slotIdx) in sortedSlots {
                    let logits_i = lastLogits[slotIdx]
                    let session = engine.sessions[rid]
                    let token = session?.iterator.sampler.sample(logits: logits_i)
                        ?? argMax(logits_i, axis: -1)
                    eval(token)
                    let tokenId = token.item(Int.self)
                    engine.batchTokens[slotIdx] = tokenId
                    reqIds[Int(count)] = strdup(rid)
                    outTokens[Int(count)] = Int32(tokenId)
                    count += 1
                }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            engine.totalDecodeTokens += count
            engine.totalDecodeTime += elapsed
            return count
        }

        // Semi-batched path for Qwen3 with per-request caches
        if let qwenModel = engine.model as? Qwen3Model {
            var tokens: [Int] = []
            var allCaches: [[KVCache]] = []
            var activeRids: [String] = []

            for rid in rids {
                guard let session = engine.sessions[rid] else { continue }
                let tokenId = session.iterator.y.tokens.item(Int.self)
                tokens.append(tokenId)
                allCaches.append(session.iterator.cache)
                activeRids.append(rid)
            }

            guard !tokens.isEmpty else { return Int32(0) }

            let inputBatch = MLXArray(tokens).reshaped(tokens.count, 1)
            let logitsBatch = qwenModel.batchedDecode(inputBatch, caches: allCaches)

            var count: Int32 = 0
            for (idx, rid) in activeRids.enumerated() {
                guard var session = engine.sessions[rid] else { continue }
                let logits = logitsBatch[idx, -1, 0...]
                let newToken = session.iterator.sampler.sample(logits: logits)
                eval(newToken)
                let tokenId = newToken.item(Int.self)
                session.iterator.y = .init(tokens: newToken)
                session.iterator.tokenCount += 1
                engine.sessions[rid] = session
                reqIds[Int(count)] = strdup(rid)
                outTokens[Int(count)] = Int32(tokenId)
                count += 1
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            engine.totalDecodeTokens += count
            engine.totalDecodeTime += elapsed
            return count
        }

        // Fallback: sequential stepAsync/readToken for non-Qwen3 models
        var stepped: [String] = []
        for rid in rids {
            guard var session = engine.sessions[rid] else { continue }
            if session.iterator.stepAsync() {
                stepped.append(rid)
            }
            engine.sessions[rid] = session
        }

        var count: Int32 = 0
        for rid in stepped {
            guard var session = engine.sessions[rid] else { continue }
            let tokenId = session.iterator.readToken()
            engine.sessions[rid] = session
            reqIds[Int(count)] = strdup(rid)
            outTokens[Int(count)] = Int32(tokenId)
            count += 1
        }

        for rid in rids where !stepped.contains(rid) {
            reqIds[Int(count)] = strdup(rid)
            outTokens[Int(count)] = -1
            count += 1
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        engine.totalDecodeTokens += count
        engine.totalDecodeTime += elapsed
        return count
    }
}

/// VLM prefill: tokens + image pixels
@_cdecl("vsm_engine_prefill_vlm")
public func vsm_engine_prefill_vlm(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?,
    promptTokens: UnsafePointer<Int32>?,
    numTokens: Int32,
    pixels: UnsafePointer<Float>?,
    pixelCount: Int32,
    imageHeight: Int32,
    imageWidth: Int32,
    temperature: Float,
    topP: Float
) -> Int32 {
    guard let handle, let promptTokens, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(-1) }

        let tokens = (0..<Int(numTokens)).map { Int(promptTokens[$0]) }
        let tokenArray = MLXArray(tokens)

        var params = engine.generateParams
        params.temperature = temperature
        params.topP = topP

        // Build LMInput with image if provided
        let input: LMInput
        if let pixels, pixelCount > 0 {
            let pixelData = Array(UnsafeBufferPointer(start: pixels, count: Int(pixelCount)))
            let channels = 3
            let h = Int(imageHeight)
            let w = Int(imageWidth)
            let pixelArray = MLXArray(pixelData).reshaped(1, channels, h, w)
            let processedImage = LMInput.ProcessedImage(
                pixels: pixelArray,
                frames: [THW(1, h, w)]
            )
            input = LMInput(
                text: .init(tokens: tokenArray),
                image: processedImage
            )
        } else {
            input = LMInput(text: .init(tokens: tokenArray))
        }

        do {
            var iterator = try TokenIterator(
                input: input,
                model: engine.model,
                parameters: params
            )
            guard let firstToken = iterator.next() else { return Int32(-1) }

            engine.sessions[rid] = RequestSession(
                iterator: iterator,
                temperature: temperature,
                topP: topP
            )
            return Int32(firstToken)
        } catch {
            print("[vsm] VLM prefill error for \(rid): \(error)")
            return Int32(-1)
        }
    }
}

/// Initialize batched KV caches and prefill all requests for fully batched decode.
/// Must be called AFTER all prefill_req calls. Copies cache state into BatchedKVCache.
@_cdecl("vsm_engine_init_batched")
public func vsm_engine_init_batched(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }

    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(0) }
        guard let qwenModel = engine.model as? Qwen3Model else { return Int32(-1) }

        let rids = Array(engine.sessions.keys)
        let B = rids.count
        guard B > 0 else { return Int32(0) }

        // Get model dimensions from first session's cache
        guard let firstSession = engine.sessions[rids[0]] else { return Int32(0) }
        let numLayers = firstSession.iterator.cache.count
        guard numLayers > 0 else { return Int32(0) }

        // Determine KV heads and head dim from first cache
        guard let firstCache = firstSession.iterator.cache[0] as? KVCacheSimple,
              let firstKeys = firstCache.peek()?.0 else { return Int32(0) }
        let kvHeads = firstKeys.dim(1)
        let headDim = firstKeys.dim(3)
        let maxSeq = 2048

        // Create BatchedKVCache per layer
        var bCaches = [BatchedKVCache]()
        for _ in 0..<numLayers {
            bCaches.append(BatchedKVCache(
                maxBatch: max(B, 64), kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: firstKeys.dtype
            ))
        }

        // Copy each request's cache into the batched cache
        engine.batchSlots.removeAll()
        engine.batchTokens = Array(repeating: 0, count: max(B, 64))

        for (slotIdx, rid) in rids.enumerated() {
            guard let session = engine.sessions[rid] else { continue }
            engine.batchSlots[rid] = slotIdx

            // Copy last generated token
            let tokenId = session.iterator.y.tokens.item(Int.self)
            engine.batchTokens[slotIdx] = tokenId

            // Copy KV cache data per layer
            for layerIdx in 0..<numLayers {
                let cache = session.iterator.cache[layerIdx]
                let offset = cache.offset
                if let (k, v) = cache.peek() {
                    // k, v: [1, kvHeads, offset, headDim]
                    bCaches[layerIdx].keys[slotIdx, 0..., ..<offset, 0...] = k[0]
                    bCaches[layerIdx].values[slotIdx, 0..., ..<offset, 0...] = v[0]
                }
                bCaches[layerIdx].offsets[slotIdx] = offset
                bCaches[layerIdx].active = max(bCaches[layerIdx].active, slotIdx + 1)
            }
        }

        // Materialize the cache copies
        var toEval = [MLXArray]()
        for c in bCaches {
            toEval.append(c.keys)
            toEval.append(c.values)
        }
        eval(toEval)

        engine.batchedCaches = bCaches
        print("[vsm] Batched KV cache initialized: B=\(B), layers=\(numLayers), kvHeads=\(kvHeads), headDim=\(headDim)")
        return Int32(B)
    }
}

/// Batch decode with logprobs — same as decode_all but computes
/// log_softmax and extracts the sampled token's log-probability.
@_cdecl("vsm_engine_decode_all_logprobs")
public func vsm_engine_decode_all_logprobs(
    _ handle: UnsafeMutableRawPointer?,
    reqIds: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    outTokens: UnsafeMutablePointer<Int32>?,
    outLogprobs: UnsafeMutablePointer<Float>?,
    maxReqs: Int32
) -> Int32 {
    guard let handle, let reqIds, let outTokens, let outLogprobs else { return 0 }

    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(0) }

        let start = CFAbsoluteTimeGetCurrent()
        let rids = Array(engine.sessions.keys.prefix(Int(maxReqs)))
        guard !rids.isEmpty else { return Int32(0) }

        if let qwenModel = engine.model as? Qwen3Model,
           let bCaches = engine.batchedCaches,
           !engine.batchSlots.isEmpty
        {
            let B = engine.batchSlots.count
            let tokens = engine.batchTokens

            let inputBatch = MLXArray(tokens[0..<B]).reshaped(B, 1)
            let logitsBatch = qwenModel.fullyBatchedDecode(inputBatch, caches: bCaches)
            let lastLogits = logitsBatch[0..., -1, 0...]  // [B, vocab]

            // Compute log_softmax for logprobs
            let logSoftmax = lastLogits - MLX.logSumExp(lastLogits, axis: -1, keepDims: true)

            // Greedy sample
            let sampledTokens = argMax(lastLogits, axis: -1)  // [B]
            eval(sampledTokens, logSoftmax)

            var count: Int32 = 0
            let sortedSlots = engine.batchSlots.sorted { $0.value < $1.value }
            for (rid, slotIdx) in sortedSlots {
                let tokenId = sampledTokens[slotIdx].item(Int.self)
                let logprob = logSoftmax[slotIdx, tokenId].item(Float.self)

                engine.batchTokens[slotIdx] = tokenId
                reqIds[Int(count)] = strdup(rid)
                outTokens[Int(count)] = Int32(tokenId)
                outLogprobs[Int(count)] = logprob
                count += 1
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            engine.totalDecodeTokens += count
            engine.totalDecodeTime += elapsed
            return count
        }

        // Fallback: decode without logprobs
        return vsm_engine_decode_all(handle, reqIds: reqIds, outTokens: outTokens, maxReqs: maxReqs)
    }
}

@_cdecl("vsm_engine_finish_req")
public func vsm_engine_finish_req(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?
) {
    guard let handle, let reqId else { return }
    let rid = String(cString: reqId)

    engineQueue.sync {
        guard let engine = engines[handle] else { return }
        engine.sessions.removeValue(forKey: rid)
    }
}

@_cdecl("vsm_engine_active_requests")
public func vsm_engine_active_requests(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }
    return engineQueue.sync {
        guard let engine = engines[handle] else { return Int32(0) }
        return Int32(engine.sessions.count)
    }
}

@_cdecl("vsm_engine_decode_batch")
public func vsm_engine_decode_batch(
    _ handle: UnsafeMutableRawPointer?,
    maxTokens: Int32,
    temperature: Float,
    topP: Float,
    outputTokens: UnsafeMutablePointer<Int32>?,
    outputCapacity: Int32
) -> Int32 {
    guard let handle, let outputTokens else { return 0 }

    return engineQueue.sync {
        guard let engine = engines[handle],
              var session = engine.sessions["_default"] else { return Int32(0) }

        let limit = min(Int(maxTokens), Int(outputCapacity))
        var count: Int32 = 0

        let start = CFAbsoluteTimeGetCurrent()
        for i in 0..<limit {
            guard let token = session.iterator.next() else { break }
            outputTokens[i] = Int32(token)
            count += 1
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        engine.sessions["_default"] = session
        engine.totalDecodeTokens += count
        engine.totalDecodeTime += elapsed
        engine.peakMemoryBytes = max(
            engine.peakMemoryBytes,
            Int64(Memory.peakMemory)
        )

        return count
    }
}

@_cdecl("vsm_engine_get_logits")
public func vsm_engine_get_logits(
    _ handle: UnsafeMutableRawPointer?,
    outVocabSize: UnsafeMutablePointer<Int32>?
) -> UnsafePointer<Float>? {
    // TODO: expose raw logits from last forward pass
    outVocabSize?.pointee = 0
    return nil
}

@_cdecl("vsm_engine_reset")
public func vsm_engine_reset(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    engineQueue.sync {
        guard let engine = engines[handle] else { return }
        engine.sessions.removeAll()
    }
}

// Mirror of vsm_perf_stats_t from bridge.h
struct VsmPerfStats {
    var prefill_tokens_per_sec: Double
    var decode_tokens_per_sec: Double
    var peak_memory_bytes: Int64
    var total_tokens_generated: Int32
    var total_decode_time_sec: Double
}

@_cdecl("vsm_engine_get_stats")
public func vsm_engine_get_stats(
    _ handle: UnsafeMutableRawPointer?,
    stats: UnsafeMutableRawPointer?
) {
    guard let handle, let stats else { return }

    engineQueue.sync {
        guard let engine = engines[handle] else { return }
        let s = stats.assumingMemoryBound(to: VsmPerfStats.self)
        s.pointee = VsmPerfStats(
            prefill_tokens_per_sec: engine.prefillTokensPerSec,
            decode_tokens_per_sec: engine.totalDecodeTime > 0
                ? Double(engine.totalDecodeTokens) / engine.totalDecodeTime : 0,
            peak_memory_bytes: engine.peakMemoryBytes,
            total_tokens_generated: engine.totalDecodeTokens,
            total_decode_time_sec: engine.totalDecodeTime
        )
    }
}
