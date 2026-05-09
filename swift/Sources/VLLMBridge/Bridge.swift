// SPDX-License-Identifier: Apache-2.0
// vllm-swift C bridge implementation
//
// Wraps mlx-swift-lm's TokenIterator to expose a C API for Python ctypes.
// All GPU compute stays here in Swift/Metal — Python only drives scheduling.

import CoreImage
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@_exported import MLXLLM
import MLXVLM
// HuggingFace macros require the full HF SDK. For now, load models
// from local directories (Python downloads via huggingface_hub first).

/// Wrapper to transfer non-Sendable values across Task boundaries.
/// Safety: caller ensures no concurrent access.
struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

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
    let processor: (any UserInputProcessor)?
    let configuration: ModelConfiguration

    /// Active sessions keyed by request ID (supports concurrent requests)
    var sessions: [String: RequestSession] = [:]
    var generateParams: GenerateParameters

    /// Cap on concurrent batched-decode slots. Drives BatchedKVCache
    /// pre-allocation. Threaded from Python (vLLM scheduler_config.
    /// max_num_seqs). Default 64 retains pre-fix behavior for callers
    /// that don't set it.
    var maxConcurrentRequests: Int = 64
    /// Cap on per-slot KV depth. Pinned to max_kv_size at engine create
    /// so BatchedKVCache doesn't have to re-grow turn-to-turn.
    var maxKVSize: Int = 0

    /// Batched KV caches: one per layer, shared across all requests.
    /// Used by fullyBatchedDecode when model is Qwen3.
    var batchedCaches: [BatchedKVCache]?
    /// Polymorphic batched cache for hybrid models (attention + GDN/Mamba),
    /// used when model conforms to `BatchedHybridLLM` (e.g. Qwen3Next).
    /// Mutually exclusive with `batchedCaches` for a given engine instance.
    var batchedHybridCaches: BatchedHybridCache?
    /// Maps request ID → batch slot index in batchedCaches / batchedHybridCaches.
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
        processor: (any UserInputProcessor)? = nil,
        configuration: ModelConfiguration,
        params: GenerateParameters
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.configuration = configuration
        self.generateParams = params
    }
}

// Engine storage — nonisolated(unsafe) silences Swift 6 concurrency
// checker. Actual thread safety provided by engineQueue.
nonisolated(unsafe) private var engines: [UnsafeMutableRawPointer: InferenceEngine] = [:]
private let engineQueue = DispatchQueue(label: "vsm.engine.queue")

/// Resolve the `kv_scheme` string carried in `GenerateParameters` into the
/// `(turboKeyBits, turboValueBits)` tuple that `BatchedKVCache.init` /
/// `model.newBatchedHybridCache` accept. Returns `(nil, nil)` when no turbo
/// scheme is set or the scheme is `"none"` — preserving the legacy raw-fp16
/// batched cache path. Mirrors `parseTurboScheme` in mlx-swift-lm but lives
/// here because it's the bridge-side decision of which cache flavor to build.
private func batchedTurboBits(
    from params: GenerateParameters
) -> (Int?, Int?) {
    guard let scheme = params.kvScheme,
          scheme.hasPrefix("turbo")
    else { return (nil, nil) }
    let parsed = parseTurboScheme(scheme)
    // Symmetric form ("turbo4") returns keyBits == nil → fall back to the
    // top-level bits value for both K and V. parseTurboScheme uses
    // `params.kvBits` semantics indirectly via the top-level bits field.
    let kb = parsed.keyBits ?? parsed.bits
    let vb = parsed.valueBits ?? parsed.bits
    return (kb, vb)
}

// MARK: - C API implementations

@_cdecl("vsm_engine_create")
public func vsm_engine_create(
    modelPath: UnsafePointer<CChar>?,
    dtype: UnsafePointer<CChar>?,
    maxKVSize: Int32,
    kvScheme: UnsafePointer<CChar>?,
    kvBits: Int32,
    memoryFraction: Float,
    maxNumSeqs: Int32
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

    // Load model synchronously. Use a box to pass results across the
    // Task boundary without triggering Swift 6.2 SendingRisksDataRace.
    final class LoadResult: @unchecked Sendable {
        var context: ModelContext?
        var error: (any Error)?
    }
    let result = LoadResult()
    let semaphore = DispatchSemaphore(value: 0)
    // Resolve symlinks: MLX's qwen3 path errors out on symlinked dirs.
    let modelURL = URL(fileURLWithPath: modelId).resolvingSymlinksInPath()

    // Detect whether this model has a vision tower by inspecting config.json.
    // For dual-registered model types (e.g. `qwen3_5`), `LLMModelFactory` will
    // happily load a VL checkpoint as a text-only model, silently skipping the
    // vision tower forward pass. That breaks image inputs (image-placeholder
    // tokens get embedded as raw text). When `vision_config` (or any vision
    // marker) is present in config.json, prefer the VLM factory; otherwise
    // keep the LLM-first fast path that engages BatchedHybridLLM.
    let isVisionModel: Bool = {
        let cfgURL = modelURL.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        if obj["vision_config"] != nil { return true }
        if let archs = obj["architectures"] as? [String] {
            for a in archs {
                let s = a.lowercased()
                if s.contains("vl") || s.contains("vision") ||
                   s.contains("conditionalgeneration") {
                    return true
                }
            }
        }
        if let mt = obj["model_type"] as? String {
            let m = mt.lowercased()
            if m.contains("vl") || m.contains("vision") { return true }
        }
        return false
    }()

    Task {
        do {
            if isVisionModel {
                // VL-first path: load via VLMModelFactory so the vision tower
                // forward + image-feature merge run.
                do {
                    result.context = try await MLXVLM.VLMModelFactory.shared.load(
                        from: modelURL,
                        using: StubTokenizerLoader()
                    )
                } catch {
                    print("[vsm] VLM load failed, trying LLM: \(error.localizedDescription)")
                    result.context = try await MLXLLM.LLMModelFactory.shared.load(
                        from: modelURL,
                        using: StubTokenizerLoader()
                    )
                }
            } else {
                // LLM-first fast path (unchanged): preserves BatchedHybridLLM.
                do {
                    result.context = try await MLXLLM.LLMModelFactory.shared.load(
                        from: modelURL,
                        using: StubTokenizerLoader()
                    )
                } catch {
                    print("[vsm] LLM load failed, trying VLM: \(error.localizedDescription)")
                    result.context = try await MLXVLM.VLMModelFactory.shared.load(
                        from: modelURL,
                        using: StubTokenizerLoader()
                    )
                }
            }
        } catch {
            result.error = error
        }
        semaphore.signal()
    }
    semaphore.wait()

    guard let context = result.context else {
        let errMsg = result.error?.localizedDescription ?? "unknown"
        print("[vsm] Failed to load model \(modelId): \(errMsg)")
        return nil
    }

    let engine = InferenceEngine(
        model: context.model,
        tokenizer: context.tokenizer,
        processor: context.processor,
        configuration: ModelConfiguration(id: modelId),
        params: params
    )
    // Cap concurrent batched-decode slots from the scheduler's
    // max_num_seqs. Falls back to legacy behavior (64) if Python
    // didn't pass anything sensible.
    engine.maxConcurrentRequests = (maxNumSeqs > 0) ? Int(maxNumSeqs) : 64
    engine.maxKVSize = (maxKVSize > 0) ? Int(maxKVSize) : 0
    print("[vsm] Engine create: maxNumSeqs=\(engine.maxConcurrentRequests) "
          + "maxKVSize=\(engine.maxKVSize)")

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
    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(0) }
        // Flatten model parameters, find lm_head or embed_tokens
        let flat = engine.model.parameters().flattened()
        for (key, arr) in flat {
            if key == "lm_head.weight" || key == "model.embed_tokens.weight" {
                return Int32(arr.dim(0))
            }
        }
        return Int32(0)
    }
}

@_cdecl("vsm_engine_num_layers")
public func vsm_engine_num_layers(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return 0 }
    return engineQueue.sync { () -> Int32 in
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
    guard let handle else { return 128 }
    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(128) }
        // k_proj.weight: [num_kv_heads * head_dim, hidden_dim]
        let flat = engine.model.parameters().flattened()
        var kvDim = 0
        for (key, arr) in flat {
            if key.hasSuffix("self_attn.k_proj.weight") {
                kvDim = arr.dim(0)
                break
            }
        }
        guard kvDim > 0 else { return Int32(128) }
        // Common head_dim values — find first that divides evenly
        let candidates = [128, 96, 80, 64]
        let match = candidates.first { kvDim % $0 == 0 }
        return Int32(match ?? kvDim)
    }
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

    return engineQueue.sync { () -> Int32 in
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

/// Compute prompt logprobs: for each position i, the log-probability of token[i+1]
/// given tokens[0..i]. Returns number of logprobs written (numTokens - 1).
/// outLogprobs must have capacity for at least (numTokens - 1) floats.
@_cdecl("vsm_engine_prompt_logprobs")
public func vsm_engine_prompt_logprobs(
    _ handle: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>?,
    numTokens: Int32,
    outLogprobs: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let promptTokens, let outLogprobs, numTokens > 1 else { return 0 }

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(0) }

        let n = Int(numTokens)
        let tokens = (0..<n).map { Int(promptTokens[$0]) }
        let tokenArray = MLXArray(tokens)

        // Run model forward on full prompt with proper caches for hybrid models (GDN needs MambaCache)
        let input = LMInput(text: .init(tokens: tokenArray))
        let cache: [KVCache]? = (engine.model as? LLMModel)?.newCache(parameters: nil) ?? nil
        let result = engine.model(input.text.tokens.reshaped(1, n), cache: cache)
        // result: [1, seq_len, vocab_size]
        let logits = result.squeezed(axis: 0)  // [seq_len, vocab]

        // log_softmax over vocab dimension
        let logSoftmax = logits - MLX.logSumExp(logits, axis: -1, keepDims: true)
        eval(logSoftmax)

        // For each position i (0..n-2), extract logprob of token[i+1]
        let count = n - 1
        for i in 0..<count {
            let nextToken = tokens[i + 1]
            outLogprobs[i] = logSoftmax[i, nextToken].item(Float.self)
        }

        return Int32(count)
    }
}

@_cdecl("vsm_engine_decode_step_req")
public func vsm_engine_decode_step_req(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?
) -> Int32 {
    guard let handle, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync { () -> Int32 in
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

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(0) }

        let start = CFAbsoluteTimeGetCurrent()
        // Pull active rids from sessions, falling back to batchSlots when the
        // batched-prefill path was used (it skips per-request session setup
        // because RequestSession requires a TokenIterator we don't have).
        var rids = Array(engine.sessions.keys.prefix(Int(maxReqs)))
        if rids.isEmpty && !engine.batchSlots.isEmpty {
            rids = Array(engine.batchSlots.keys.prefix(Int(maxReqs)))
        }
        guard !rids.isEmpty else { return Int32(0) }

        // v0.5.4 fix: gate the Qwen3 batched/semi-batched decode paths off
        // when kv_scheme=turbo* is set on a dense (Qwen3) serve flow.
        //
        // v0.5.3 wired kv_scheme into the *batched-prefill* paths
        // (prefill_batched_uniform / hybrid init_batched), but the dense
        // Qwen3 serve path is prefill_req → init_batched → decode_all, and
        // both stages drop the scheme:
        //   - init_batched fails its `as? KVCacheSimple` cast (per-request
        //     caches are RotatingKVCache when kvScheme is set), returns 0
        //     silently → batchedCaches stays nil.
        //   - The Qwen3 semi-batched fallback below calls
        //     Qwen3Attention.batchedForward, whose per-request RoPE+update
        //     loop corrupts on rotating-window K/V dequant semantics. Output
        //     degenerates to "<think>1\n1\n1...".
        //
        // Until both paths are turbo-aware, force turbo-on-dense-Qwen3
        // through the sequential-stepAsync TokenIterator fallback at the
        // bottom of this function. That's the well-tested turbo decode path
        // every standalone mlx-swift-lm consumer uses. Slower than batched
        // SDPA across requests, but correct.
        //
        // Use kvScheme directly (not a `cache.first is TurboQuantKVCache`
        // type-cast): the per-request cache is actually RotatingKVCache —
        // mlx-swift-lm converts it to TurboQuantKVCache lazily inside
        // step() via maybeQuantizeKVCache, so an early type-cast misses
        // turbo'd sessions. kvScheme on generateParams is the durable signal.
        let hasTurboCache: Bool =
            engine.generateParams.kvScheme?.hasPrefix("turbo") ?? false

        // Cast ordering: Qwen3 fast path FIRST so the verified hot path
        // stays bit-identical (no extra protocol cast in the inner loop).
        // Qwen3Model and BatchedHybridLLM are disjoint conformances —
        // ordering is correctness-neutral, only perf-motivated.
        // Fully batched path for Qwen3 with BatchedKVCache
        if !hasTurboCache,
           let qwenModel = engine.model as? Qwen3Model,
           let bCaches = engine.batchedCaches,
           !engine.batchSlots.isEmpty
        {
            let B = engine.batchSlots.count
            let tokens = engine.batchTokens

            // Single batched forward: [B, 1] → [B, 1, vocab]
            let inputBatch = MLXArray(tokens[0..<B]).reshaped(B, 1)
            let logitsBatch = qwenModel.fullyBatchedDecode(inputBatch, caches: bCaches)

            let lastLogits = logitsBatch[0..., -1, 0...]  // [B, vocab]

            // Check if all requests use greedy sampling. Missing session is
            // treated as greedy (default) — happens when init_batched freed
            // the per-request session after copying its KV into batchedCaches,
            // or when prefill_batched_uniform skipped session creation.
            let sortedSlots = engine.batchSlots.sorted { $0.value < $1.value }
            let allGreedy = sortedSlots.allSatisfy { (rid, _) in
                (engine.sessions[rid]?.temperature ?? 0) == 0
            }

            // Match TokenIterator.next() pattern: return previousY, advance to next
            // TODO: temperature sampling when !allGreedy (gap #7)
            let sampledTokens: MLXArray
            if allGreedy {
                sampledTokens = argMax(lastLogits, axis: -1)
            } else {
                // Per-request temperature sampling
                var tokenList = [Int]()
                for (rid, slotIdx) in sortedSlots {
                    let temp = engine.sessions[rid]?.temperature ?? 0
                    let logits = lastLogits[slotIdx]
                    if temp > 0 {
                        let scaled = logits / temp
                        let sampled = MLXRandom.categorical(scaled)
                        tokenList.append(sampled.item(Int.self))
                    } else {
                        tokenList.append(argMax(logits, axis: -1).item(Int.self))
                    }
                }
                sampledTokens = MLXArray(tokenList)
            }
            eval(sampledTokens)

            var count: Int32 = 0
            for (rid, slotIdx) in sortedSlots {
                // Return the INPUT token (previousY pattern)
                let returnToken = engine.batchTokens[slotIdx]
                // Advance to the model's output for next step
                let nextToken = sampledTokens[slotIdx].item(Int.self)
                engine.batchTokens[slotIdx] = nextToken

                reqIds[Int(count)] = strdup(rid)
                outTokens[Int(count)] = Int32(returnToken)
                count += 1
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            engine.totalDecodeTokens += count
            engine.totalDecodeTime += elapsed
            return count
        }

        // Fully batched path for hybrid models (Qwen3Next, etc.) with
        // BatchedHybridCache. Mirrors the Qwen3 path — same sampling /
        // temperature plumbing, just dispatches through the protocol.
        if let hybridModel = engine.model as? any BatchedHybridLLM,
           let hCaches = engine.batchedHybridCaches,
           !engine.batchSlots.isEmpty
        {
            let B = engine.batchSlots.count
            let tokens = engine.batchTokens

            // Single batched forward: [B, 1] → [B, 1, vocab]
            let inputBatch = MLXArray(tokens[0..<B]).reshaped(B, 1)
            let logitsBatch = hybridModel.fullyBatchedDecode(inputBatch, caches: hCaches)

            let lastLogits = logitsBatch[0..., -1, 0...]  // [B, vocab]

            // Same greedy / per-request temperature split as the Qwen3 path.
            let sortedSlots = engine.batchSlots.sorted { $0.value < $1.value }
            let allGreedy = sortedSlots.allSatisfy { (rid, _) in
                (engine.sessions[rid]?.temperature ?? 0) == 0
            }

            // TODO: per-request temperature sampling could share a helper
            // with the Qwen3 path once we add it (gap #7).
            let sampledTokens: MLXArray
            if allGreedy {
                sampledTokens = argMax(lastLogits, axis: -1)
            } else {
                var tokenList = [Int]()
                for (rid, slotIdx) in sortedSlots {
                    let temp = engine.sessions[rid]?.temperature ?? 0
                    let logits = lastLogits[slotIdx]
                    if temp > 0 {
                        let scaled = logits / temp
                        let sampled = MLXRandom.categorical(scaled)
                        tokenList.append(sampled.item(Int.self))
                    } else {
                        tokenList.append(argMax(logits, axis: -1).item(Int.self))
                    }
                }
                sampledTokens = MLXArray(tokenList)
            }
            eval(sampledTokens)

            var count: Int32 = 0
            for (rid, slotIdx) in sortedSlots {
                let returnToken = engine.batchTokens[slotIdx]
                let nextToken = sampledTokens[slotIdx].item(Int.self)
                engine.batchTokens[slotIdx] = nextToken

                reqIds[Int(count)] = strdup(rid)
                outTokens[Int(count)] = Int32(returnToken)
                count += 1
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            engine.totalDecodeTokens += count
            engine.totalDecodeTime += elapsed
            return count
        }

        // Semi-batched path for Qwen3 with per-request caches.
        // Skipped when any per-request cache is TurboQuantKVCache — see the
        // top-of-function `hasTurboCache` comment. Falls through to the
        // sequential-stepAsync fallback which IS validated for turbo.
        if !hasTurboCache, let qwenModel = engine.model as? Qwen3Model {
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

/// VLM prefill: tokens + preprocessed pixel tensor from Python.
/// Python (vLLM) handles model-specific image preprocessing.
/// Swift receives ready-to-use pixel data.
@_cdecl("vsm_engine_prefill_vlm")
public func vsm_engine_prefill_vlm(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?,
    promptTokens: UnsafePointer<Int32>?,
    numTokens: Int32,
    pixels: UnsafePointer<Float>?,
    pixelCount: Int32,
    pixelDims: UnsafePointer<Int32>?,
    numPixelDims: Int32,
    gridTHW: UnsafePointer<Int32>?,
    temperature: Float,
    topP: Float
) -> Int32 {
    guard let handle, let promptTokens, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(-1) }

        let tokens = (0..<Int(numTokens)).map { Int(promptTokens[$0]) }
        let tokenArray = MLXArray(tokens)

        var params = engine.generateParams
        params.temperature = temperature
        params.topP = topP

        // Build LMInput with preprocessed pixel data
        let input: LMInput
        if let pixels, pixelCount > 0, let pixelDims, numPixelDims > 0 {
            let pixelData = Array(UnsafeBufferPointer(start: pixels, count: Int(pixelCount)))
            let shape = (0..<Int(numPixelDims)).map { Int(pixelDims[$0]) }

            let pixelArray = MLXArray(pixelData).reshaped(shape)

            // Use grid_thw for frames if provided, else infer from shape
            let frames: [THW]
            if let gridTHW {
                let t = Int(gridTHW[0])
                let h = Int(gridTHW[1])
                let w = Int(gridTHW[2])
                frames = [THW(t, h, w)]
            } else if shape.count >= 4 {
                frames = [THW(1, shape[shape.count - 2], shape[shape.count - 1])]
            } else {
                frames = [THW(1, shape.last ?? 1, 1)]
            }

            let processedImage = LMInput.ProcessedImage(
                pixels: pixelArray, frames: frames
            )
            input = LMInput(text: .init(tokens: tokenArray), image: processedImage)
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

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return Int32(0) }

        let rids = Array(engine.sessions.keys)
        let B = rids.count
        guard B > 0 else { return Int32(0) }

        // Cast ordering: try Qwen3 first (preserves the verified hot path),
        // then BatchedHybridLLM. Qwen3Model and BatchedHybridLLM are disjoint
        // so order is correctness-neutral. Models that match neither return -1.
        if engine.model is Qwen3Model {
            // Fall through to the existing Qwen3 init path below.
        } else if let hybridModel = engine.model as? any BatchedHybridLLM {
            return initBatchedHybrid(engine: engine, model: hybridModel, rids: rids)
        } else {
            return Int32(-1)
        }

        // Get model dimensions from first session's cache
        guard let firstSession = engine.sessions[rids[0]] else { return Int32(0) }
        let numLayers = firstSession.iterator.cache.count
        guard numLayers > 0 else { return Int32(0) }

        // Determine KV heads and head dim from first cache
        guard let firstCache = firstSession.iterator.cache[0] as? KVCacheSimple,
              let firstKeys = firstCache.peek()?.0 else { return Int32(0) }
        let kvHeads = firstKeys.dim(1)
        let headDim = firstKeys.dim(3)
        // Size cache from longest actual prefill + decode margin. Pin to
        // engine.maxKVSize when known so subsequent prefills can't force
        // a re-grow (which on Metal stacks the old backing in the heap
        // and shows up as a GB/turn unified-mem leak).
        let maxPrefillOffset = rids.compactMap {
            engine.sessions[$0]?.iterator.cache.first?.offset
        }.max() ?? 0
        let decodeMargin = 512
        let maxSeq = engine.maxKVSize > 0
            ? engine.maxKVSize
            : max(2048, maxPrefillOffset + decodeMargin)
        // Respect scheduler max_num_seqs instead of forcing 64 slots.
        // 64-slot pre-alloc on max_num_seqs=1 wastes (64-1)× per-layer
        // KV bytes — at maxSeq=64K that's ~60 GB unused-but-allocated.
        let maxBatch = max(B, engine.maxConcurrentRequests)

        var bCaches = [BatchedKVCache]()
        for _ in 0..<numLayers {
            bCaches.append(BatchedKVCache(
                maxBatch: maxBatch, kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: firstKeys.dtype
            ))
        }

        // Copy per-request cache into batched cache, then free the session.
        // Per-request KVCacheSimples hold full prompt-length K/V (~1 GB/req at
        // 4B/8K). Holding them across the copy doubles KV memory and OOMs at
        // long-ctx high-B cells.
        engine.batchSlots.removeAll()
        engine.batchTokens = Array(repeating: 0, count: maxBatch)

        for (slotIdx, rid) in rids.enumerated() {
            guard let session = engine.sessions[rid] else { continue }
            engine.batchSlots[rid] = slotIdx

            let tokenId = session.iterator.y.tokens.item(Int.self)
            engine.batchTokens[slotIdx] = tokenId

            for layerIdx in 0..<numLayers {
                let cache = session.iterator.cache[layerIdx]
                let offset = cache.offset
                if let (k, v) = cache.peek() {
                    bCaches[layerIdx].keys[slotIdx, 0..., ..<offset, 0...] = k[0]
                    bCaches[layerIdx].values[slotIdx, 0..., ..<offset, 0...] = v[0]
                }
                bCaches[layerIdx].offsets[slotIdx] = offset
                bCaches[layerIdx].active = max(bCaches[layerIdx].active, slotIdx + 1)
            }

            // Materialize this slot's writes, then drop the per-req cache.
            // Otherwise per-req K/V accumulate alongside the growing batched
            // cache, peaking at 2× total KV memory.
            var slotEval = [MLXArray]()
            for c in bCaches {
                slotEval.append(c.keys)
                slotEval.append(c.values)
            }
            eval(slotEval)
            engine.sessions[rid] = nil
        }

        // Materialize the final cache state.
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

/// Add a single request to the batched KV cache without full reinit.
/// Must be called after prefill_req for this request.
/// Returns the slot index, or -1 on failure.
@_cdecl("vsm_engine_add_batch_slot")
public func vsm_engine_add_batch_slot(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?
) -> Int32 {
    guard let handle, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle],
              let session = engine.sessions[rid] else { return Int32(-1) }

        // Hybrid path: copy per-layer cache (KVCacheSimple OR MambaCache) into
        // the matching BatchedHybridCache layer, then addSlot() to advance
        // active counts in lockstep across all layers.
        if let hCaches = engine.batchedHybridCaches {
            return addBatchSlotHybrid(
                engine: engine, hCaches: hCaches, rid: rid, session: session)
        }

        guard let bCaches = engine.batchedCaches else { return Int32(-1) }

        // Find next available slot
        let slotIdx = engine.batchSlots.count
        guard slotIdx < bCaches[0].maxBatch else { return Int32(-1) }

        let numLayers = session.iterator.cache.count
        guard numLayers == bCaches.count else { return Int32(-1) }

        // Copy this request's KV cache into the batch slot
        let tokenId = session.iterator.y.tokens.item(Int.self)
        engine.batchTokens[slotIdx] = tokenId
        engine.batchSlots[rid] = slotIdx

        for layerIdx in 0..<numLayers {
            let cache = session.iterator.cache[layerIdx]
            let offset = cache.offset
            if let (k, v) = cache.peek() {
                bCaches[layerIdx].keys[slotIdx, 0..., ..<offset, 0...] = k[0]
                bCaches[layerIdx].values[slotIdx, 0..., ..<offset, 0...] = v[0]
            }
            bCaches[layerIdx].offsets[slotIdx] = offset
            bCaches[layerIdx].active = max(bCaches[layerIdx].active, slotIdx + 1)
        }

        // Materialize and drop per-req cache — avoid doubled KV memory.
        var toEval = [MLXArray]()
        for c in bCaches { toEval.append(c.keys); toEval.append(c.values) }
        eval(toEval)
        engine.sessions[rid] = nil

        engine.batchedCaches = bCaches
        return Int32(slotIdx)
    }
}

/// Remove a request from the batched KV cache.
/// Swaps the last active slot into the removed slot to keep dense packing.
@_cdecl("vsm_engine_remove_batch_slot")
public func vsm_engine_remove_batch_slot(
    _ handle: UnsafeMutableRawPointer?,
    reqId: UnsafePointer<CChar>?
) -> Int32 {
    guard let handle, let reqId else { return -1 }
    let rid = String(cString: reqId)

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle],
              let slotIdx = engine.batchSlots[rid] else { return Int32(-1) }

        // Hybrid path: delegate the swap-from-end to BatchedHybridCache.
        if let hCaches = engine.batchedHybridCaches {
            let lastSlot = engine.batchSlots.count - 1
            if slotIdx < lastSlot {
                guard let lastRid = engine.batchSlots.first(where: { $0.value == lastSlot })?.key
                else { return Int32(-1) }
                hCaches.removeSlot(slotIdx)
                engine.batchTokens[slotIdx] = engine.batchTokens[lastSlot]
                engine.batchSlots[lastRid] = slotIdx
            } else {
                hCaches.removeSlot(slotIdx)
            }
            engine.batchSlots.removeValue(forKey: rid)
            return Int32(0)
        }

        guard let bCaches = engine.batchedCaches else { return Int32(-1) }

        let lastSlot = engine.batchSlots.count - 1

        if slotIdx < lastSlot {
            // Swap last slot into removed slot
            guard let lastRid = engine.batchSlots.first(where: { $0.value == lastSlot })?.key
            else { return Int32(-1) }

            for layerIdx in 0..<bCaches.count {
                let offset = bCaches[layerIdx].offsets[lastSlot]
                bCaches[layerIdx].keys[slotIdx, 0..., ..<offset, 0...] =
                    bCaches[layerIdx].keys[lastSlot, 0..., ..<offset, 0...]
                bCaches[layerIdx].values[slotIdx, 0..., ..<offset, 0...] =
                    bCaches[layerIdx].values[lastSlot, 0..., ..<offset, 0...]
                bCaches[layerIdx].offsets[slotIdx] = offset
            }

            engine.batchTokens[slotIdx] = engine.batchTokens[lastSlot]
            engine.batchSlots[lastRid] = slotIdx
        }

        // Clear last slot
        engine.batchSlots.removeValue(forKey: rid)
        for layerIdx in 0..<bCaches.count {
            bCaches[layerIdx].active = max(0, bCaches[layerIdx].active - 1)
        }

        engine.batchedCaches = bCaches
        return Int32(0)
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

    return engineQueue.sync { () -> Int32 in
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

        // Mirror of the Qwen3 path for hybrid models (Qwen3Next, etc.).
        if let hybridModel = engine.model as? any BatchedHybridLLM,
           let hCaches = engine.batchedHybridCaches,
           !engine.batchSlots.isEmpty
        {
            let B = engine.batchSlots.count
            let tokens = engine.batchTokens

            let inputBatch = MLXArray(tokens[0..<B]).reshaped(B, 1)
            let logitsBatch = hybridModel.fullyBatchedDecode(inputBatch, caches: hCaches)
            let lastLogits = logitsBatch[0..., -1, 0...]  // [B, vocab]

            let logSoftmax = lastLogits - MLX.logSumExp(lastLogits, axis: -1, keepDims: true)

            let sampledTokens = argMax(lastLogits, axis: -1)
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
    return engineQueue.sync { () -> Int32 in
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

    return engineQueue.sync { () -> Int32 in
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

/// Test-only: run B sequential single-prompt prefills and capture the top-K
/// logits from the last prefill forward of each. Mirrors the chunked
/// `prepare` + `step` pattern that `TokenIterator` uses, but NOT the extra
/// `iterator.next()` advance — we want the prefill-exit logits, not the
/// next-decode-step logits, so the comparison vs `prefill_batched_uniform`
/// is at the same conceptual point in the model's compute graph.
///
/// Buffer layout: `outIndices`/`outValues` are flat `[B*K]` row-major
/// (slot i's top-K starts at `i*K`). `outValues` is float32 logits.
@_cdecl("vsm_engine_prefill_seq_uniform_topk")
public func vsm_engine_prefill_seq_uniform_topk(
    _ handle: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>?,
    numReqs: Int32,
    promptLen: Int32,
    K: Int32,
    outIndices: UnsafeMutablePointer<Int32>?,
    outValues: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let promptTokens, let outIndices, let outValues else { return -1 }
    let B = Int(numReqs)
    let T = Int(promptLen)
    let k = Int(K)
    guard B > 0, T > 0, k > 0 else { return -1 }

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return -1 }
        // Same dual-cast as the batched variant — hybrid models route
        // through their LanguageModel surface. See note above.
        let lmModel: any LanguageModel
        let cacheSource: (GenerateParameters?) -> [KVCache]
        if let qwenModel = engine.model as? Qwen3Model {
            lmModel = qwenModel
            cacheSource = qwenModel.newCache
        } else if let hybridModel = engine.model as? any BatchedHybridLLM,
                  let lm = hybridModel as? any LanguageModel {
            lmModel = lm
            cacheSource = lm.newCache
        } else {
            return -2
        }

        let intArr = UnsafeBufferPointer(start: promptTokens, count: B * T)

        for slot in 0..<B {
            let slotTokens: [Int] = (0..<T).map { Int(intArr[slot * T + $0]) }
            let slotInput = MLXArray(slotTokens).reshaped(1, T)
            let caches = cacheSource(nil)
            if T > 1 {
                let prefillChunk = slotInput[0..., ..<(T - 1)]
                _ = lmModel(LMInput.Text(tokens: prefillChunk), cache: caches, state: nil)
            }
            let lastTok = slotInput[0..., (T - 1)..<T]
            let stepOut = lmModel(LMInput.Text(tokens: lastTok), cache: caches, state: nil)
            let lastLogits = stepOut.logits[0..., -1, 0...]  // [1, V]

            let sorted = MLX.argSort(lastLogits, axis: -1)
            let vocab = lastLogits.dim(1)
            let topIdx = sorted[0..., (vocab - k)..<vocab]   // [1, K]
            let topVal = takeAlong(lastLogits, topIdx, axis: -1)  // [1, K]
            eval(topIdx, topVal)

            for j in 0..<k {
                outIndices[slot * k + j] = topIdx[0, j].item(Int32.self)
                outValues[slot * k + j] = topVal[0, j].item(Float.self)
            }
        }
        return Int32(B)
    }
}

/// Test-only batched analogue of `prefill_seq_uniform_topk`. Same chunked
/// `[B, T-1]` + `[B, 1]` pattern, single forward per chunk for all B
/// requests. Captures top-K logits at the same prefill-exit point so the
/// two functions can be compared apples-to-apples.
@_cdecl("vsm_engine_prefill_batched_uniform_topk")
public func vsm_engine_prefill_batched_uniform_topk(
    _ handle: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>?,
    numReqs: Int32,
    promptLen: Int32,
    K: Int32,
    outIndices: UnsafeMutablePointer<Int32>?,
    outValues: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let promptTokens, let outIndices, let outValues else { return -1 }
    let B = Int(numReqs)
    let T = Int(promptLen)
    let k = Int(K)
    guard B > 0, T > 0, k > 0 else { return -1 }

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return -1 }
        // Accept either Qwen3Model (dense) or any BatchedHybridLLM
        // (Qwen3Next/3.5/3.6). Both support batched [B, T] forward through
        // standard `callAsFunction` with a fresh per-layer cache. The test
        // harness uses this function as the batched correctness oracle for
        // the prefill_batched_uniform path, so it has to match the model
        // gating in that function.
        let lmModel: any LanguageModel
        let cacheSource: (GenerateParameters?) -> [KVCache]
        if let qwenModel = engine.model as? Qwen3Model {
            lmModel = qwenModel
            cacheSource = qwenModel.newCache
        } else if let hybridModel = engine.model as? any BatchedHybridLLM,
                  let lm = hybridModel as? any LanguageModel {
            lmModel = lm
            cacheSource = lm.newCache
        } else {
            return -2
        }

        let intArr = UnsafeBufferPointer(start: promptTokens, count: B * T)
        let tokens: [Int] = intArr.map { Int($0) }
        let inputBatch = MLXArray(tokens).reshaped(B, T)

        let caches = cacheSource(nil)
        if T > 1 {
            let prefillChunk = inputBatch[0..., ..<(T - 1)]
            _ = lmModel(LMInput.Text(tokens: prefillChunk), cache: caches, state: nil)
        }
        let lastTok = inputBatch[0..., (T - 1)..<T]  // [B, 1]
        let stepOut = lmModel(LMInput.Text(tokens: lastTok), cache: caches, state: nil)
        let lastLogits = stepOut.logits[0..., -1, 0...]  // [B, V]

        let sorted = MLX.argSort(lastLogits, axis: -1)
        let vocab = lastLogits.dim(1)
        let topIdx = sorted[0..., (vocab - k)..<vocab]   // [B, K]
        let topVal = takeAlong(lastLogits, topIdx, axis: -1)  // [B, K]
        eval(topIdx, topVal)

        for slot in 0..<B {
            for j in 0..<k {
                outIndices[slot * k + j] = topIdx[slot, j].item(Int32.self)
                outValues[slot * k + j] = topVal[slot, j].item(Float.self)
            }
        }
        return Int32(B)
    }
}

/// Batched prefill — uniform prompt length only (M1).
///
/// Replaces the sequential pattern of `B × prefill_req` + `init_batched`
/// with a single `[B, T]` forward through the model. The mlx-lm Python
/// equivalent and vllm-swift sequential both take ~23-27s for B=64/T=2048
/// on 4B (compute-bound on per-request prefill). This path collapses the
/// 64 sequential forwards into one batched forward.
///
/// `reqIds`: array of B null-terminated request ID strings.
/// `promptTokens`: flattened [B*T] int32 buffer, row-major (request i at offset i*T).
/// All requests use the same prompt length T (variable-length deferred to M4).
///
/// Returns 0 on success, negative on failure.
@_cdecl("vsm_engine_prefill_batched_uniform")
public func vsm_engine_prefill_batched_uniform(
    _ handle: UnsafeMutableRawPointer?,
    reqIds: UnsafePointer<UnsafePointer<CChar>?>?,
    promptTokens: UnsafePointer<Int32>?,
    numReqs: Int32,
    promptLen: Int32,
    temperature: Float,
    topP: Float
) -> Int32 {
    guard let handle, let reqIds, let promptTokens else { return -1 }
    let B = Int(numReqs)
    let T = Int(promptLen)
    guard B > 0, T > 0 else { return -1 }

    return engineQueue.sync { () -> Int32 in
        guard let engine = engines[handle] else { return -1 }

        // Read req IDs out before any compute (clearer error path).
        var rids = [String]()
        rids.reserveCapacity(B)
        for i in 0..<B {
            guard let cstr = reqIds[i] else { return -3 }
            rids.append(String(cString: cstr))
        }

        // Build [B, T] input from flattened prompt buffer.
        let totalTokens = B * T
        let intArr = UnsafeBufferPointer(start: promptTokens, count: totalTokens)
        let tokens: [Int] = intArr.map { Int($0) }
        let inputBatch = MLXArray(tokens).reshaped(B, T)

        // Cast ordering: Qwen3 fast path FIRST so the verified hot path
        // stays bit-identical (no extra protocol cast / branch in the hot
        // section). Qwen3Model and BatchedHybridLLM are disjoint, so order
        // is correctness-neutral; perf-motivated only.
        if let qwenModel = engine.model as? Qwen3Model {
            return prefillBatchedUniformQwen3(
                engine: engine, model: qwenModel,
                inputBatch: inputBatch, rids: rids, B: B, T: T)
        } else if let hybridModel = engine.model as? any BatchedHybridLLM {
            return prefillBatchedUniformHybrid(
                engine: engine, model: hybridModel,
                inputBatch: inputBatch, rids: rids, B: B, T: T)
        }
        return -2
    }
}

/// Qwen3 (dense, all-attention) batched prefill — bit-identical to the
/// pre-P5 hot path. Pulled into its own function so the hybrid variant
/// can sit alongside without reordering casts in the inner loop.
private func prefillBatchedUniformQwen3(
    engine: InferenceEngine,
    model: Qwen3Model,
    inputBatch: MLXArray,
    rids: [String],
    B: Int,
    T: Int
) -> Int32 {
    // Fresh per-layer caches. KVCacheSimple holds [B, kvHeads, T, headDim]
    // after a single batched forward — same shape as the existing
    // BatchedKVCache layout, so the copy step below is a slice assign.
    guard let caches = model.newCache(parameters: nil) as [KVCache]? else { return -4 }
    let numLayers = caches.count

    // Mirror TokenIterator's prepare + step + iterator.next() so batchTokens
    // matches sequential prefill_req: forward [B, T-1] for prefill, [B, 1]
    // for the last prompt token (first sampled), then [B, 1] of the first
    // sampled token (second sampled — what init_batched stashes).
    let lmModel: any LanguageModel = model
    if T > 1 {
        let prefillChunk = inputBatch[0..., ..<(T - 1)]
        _ = lmModel(LMInput.Text(tokens: prefillChunk), cache: caches, state: nil)
    }
    let lastPromptTokens = inputBatch[0..., (T - 1)..<T]  // [B, 1]
    let firstStepOut = lmModel(
        LMInput.Text(tokens: lastPromptTokens), cache: caches, state: nil)
    let firstLogits = firstStepOut.logits[0..., -1, 0...]  // [B, V]
    let firstSampled = argMax(firstLogits, axis: -1)       // [B]
    eval(firstSampled)
    // iterator.next() equivalent — second sampled token is what init_batched stashes.
    let secondInput = firstSampled.reshaped(B, 1)
    let secondStepOut = lmModel(
        LMInput.Text(tokens: secondInput), cache: caches, state: nil)
    let secondLogits = secondStepOut.logits[0..., -1, 0...]
    let firstTokens = argMax(secondLogits, axis: -1)        // [B]
    eval(firstTokens)

    // Read K/V from prefilled caches and pull dimensions for BatchedKVCache.
    // After prepare + step + iterator.next() equivalent, the cache holds
    // T+1 tokens (prompt + first sampled token).
    guard let firstSimple = caches[0] as? KVCacheSimple,
          let firstPeek = firstSimple.peek() else { return -5 }
    let kvHeads = firstPeek.0.dim(1)
    let headDim = firstPeek.0.dim(3)
    let cacheLen = firstSimple.offset  // = T + 1

    // Allocate BatchedKVCache sized for prefill + decode margin.
    // Pin to engine.maxKVSize when set so we don't re-grow per turn.
    let decodeMargin = 512
    let maxSeq = engine.maxKVSize > 0
        ? engine.maxKVSize
        : max(2048, cacheLen + decodeMargin)
    let maxBatch = max(B, engine.maxConcurrentRequests)
    var bCaches = [BatchedKVCache]()
    bCaches.reserveCapacity(numLayers)
    // TODO(v0.5.4): the Qwen3 dense prefill→batched transition still copies
    // raw fp16 K/V into the BatchedKVCache slot below — kv_scheme is dropped
    // for dense Qwen3 models on this path. The hybrid path (Qwen3.5/3.6 +
    // Qwen3Next via newBatchedHybridCache) does honor kv_scheme as of v0.5.3.
    // Wiring turbo here means bulk-encoding the prefilled K/V (one fusedEncode
    // dispatch over [B*H*cacheLen, headDim]) into a turbo BatchedKVCache.
    // Use a wrapped caches array we can nil out per-layer — drops the
    // [B, kvHeads, cacheLen, headDim] KVCacheSimple as soon as its layer
    // is copied into the BatchedKVCache slot. Without this, both caches
    // are alive across all 36 layers (≈ 144 GB at 4B/p8K/B=64) and OOM.
    var transientCaches: [KVCache?] = caches.map { $0 }
    for layer in 0..<numLayers {
        let bc = BatchedKVCache(
            maxBatch: maxBatch, kvHeads: kvHeads, headDim: headDim,
            maxSeq: maxSeq, dtype: firstPeek.0.dtype
        )
        guard let simple = transientCaches[layer] as? KVCacheSimple,
              let (k, v) = simple.peek() else { return -6 }
        bc.keys[..<B, 0..., ..<cacheLen, 0...] = k
        bc.values[..<B, 0..., ..<cacheLen, 0...] = v
        for i in 0..<B { bc.offsets[i] = cacheLen }
        bc.active = B
        // Materialize this layer's copy, then drop the prefill cache.
        eval(bc.keys, bc.values)
        transientCaches[layer] = nil
        bCaches.append(bc)
    }

    // Final eval to ensure all bCaches are materialized before bench timing.
    var toEval = [MLXArray]()
    for c in bCaches {
        toEval.append(c.keys)
        toEval.append(c.values)
    }
    eval(toEval)

    // Set up engine state for batched decode. Mirrors init_batched.
    engine.batchedCaches = bCaches
    engine.batchedHybridCaches = nil
    engine.batchSlots.removeAll()
    engine.batchTokens = Array(repeating: 0, count: maxBatch)

    // Stash sampled first tokens — they're returned by the first decode_all
    // (same previousY pattern as sequential prefill_req).
    let firstTokensArr: [Int32] = (0..<B).map { firstTokens[$0].item(Int32.self) }
    for (slot, rid) in rids.enumerated() {
        engine.batchSlots[rid] = slot
        engine.batchTokens[slot] = Int(firstTokensArr[slot])
    }
    // Note: RequestSession requires a TokenIterator (sequential-prefill path
    // produces one). Batched prefill skips that — decode_all's per-request
    // temperature loop uses `?? 0` default for missing sessions, which is
    // correct for greedy. Adding a TokenIterator-less session variant is
    // M5 polish work, not a correctness issue here.

    return 0
}

/// Hybrid (attention + GDN) batched prefill. Same chunked `[B, T-1]` + `[B, 1]`
/// + `[B, 1]` pattern as the Qwen3 path — works because the hybrid model's
/// standard `callAsFunction` accepts mixed `[KVCacheSimple, MambaCache, …]`
/// caches and propagates the leading-B dimension through both attention and
/// GDN layers. After the chunks land, copy each layer's cache into the right
/// `BatchedHybridCache` slot range (attention → BatchedKVCache, GDN →
/// BatchedMambaCache). Per-layer eager release mirrors the Qwen3 path's
/// doubled-KV-peak fix from `00e7538` (extended to GDN state here too).
private func prefillBatchedUniformHybrid(
    engine: InferenceEngine,
    model: any BatchedHybridLLM,
    inputBatch: MLXArray,
    rids: [String],
    B: Int,
    T: Int
) -> Int32 {
    // Hybrid models conform to LanguageModel by way of LLMModel — pull the
    // protocol surface for the chunked forward calls.
    guard let lmModel = model as? any LanguageModel else { return -7 }

    // Fresh per-layer caches: mixed [KVCacheSimple, MambaCache, …]. The
    // standard hybrid forward writes [B, ...] state into both kinds.
    let caches = lmModel.newCache(parameters: nil)
    let numLayers = caches.count

    // Same chunked pattern as the Qwen3 path so batchTokens matches what
    // sequential prefill_req would have stashed (second sampled token).
    if T > 1 {
        let prefillChunk = inputBatch[0..., ..<(T - 1)]
        _ = lmModel(LMInput.Text(tokens: prefillChunk), cache: caches, state: nil)
    }
    let lastPromptTokens = inputBatch[0..., (T - 1)..<T]  // [B, 1]
    let firstStepOut = lmModel(
        LMInput.Text(tokens: lastPromptTokens), cache: caches, state: nil)
    let firstLogits = firstStepOut.logits[0..., -1, 0...]  // [B, V]
    let firstSampled = argMax(firstLogits, axis: -1)       // [B]
    eval(firstSampled)
    let secondInput = firstSampled.reshaped(B, 1)
    let secondStepOut = lmModel(
        LMInput.Text(tokens: secondInput), cache: caches, state: nil)
    let secondLogits = secondStepOut.logits[0..., -1, 0...]
    let firstTokens = argMax(secondLogits, axis: -1)        // [B]
    eval(firstTokens)

    // Build a fresh BatchedHybridCache sized to the scheduler's
    // max_num_seqs (was hardcoded 64 — over-alloc on small concurrency).
    let maxBatch = max(B, engine.maxConcurrentRequests)
    // Honor `--additional-config kv_scheme=turbo*` on the batched-decode
    // path. Pre v0.5.3 this flag was silently dropped here — newBatched-
    // HybridCache only knew how to construct raw-fp16 BatchedKVCache, so
    // the kvScheme set in GenerateParameters never reached the attention
    // layers' batched cache. Buddy's v0.5.1 alpha report (Qwen3.6 +
    // turbo4v2 → ".2.2.2.2..." drift) was a symptom of that silent bypass.
    let (turboKB, turboVB) = batchedTurboBits(from: engine.generateParams)
    let hCaches = model.newBatchedHybridCache(
        maxBatch: maxBatch, parameters: engine.generateParams,
        turboKeyBits: turboKB, turboValueBits: turboVB)

    guard numLayers == hCaches.layers.count else {
        print("[vsm] prefill_batched_uniform hybrid: layer count mismatch — req=\(numLayers) hybrid=\(hCaches.layers.count)")
        return -8
    }

    // Per-layer copy + eager release. Mirrors the Qwen3 path's
    // `transientCaches[layer] = nil` defense; doubled-state would otherwise
    // pin both prefill caches and BatchedHybridCache across all layers.
    var transientCaches: [KVCache?] = caches.map { $0 }
    for layerIdx in 0..<numLayers {
        let dstLayer = hCaches.layers[layerIdx]
        switch dstLayer {
        case .attention(let bkv):
            guard let simple = transientCaches[layerIdx] as? KVCacheSimple,
                  copyBatchedAttentionLayer(src: simple, dst: bkv, B: B)
            else {
                print("[vsm] prefill_batched_uniform hybrid: layer \(layerIdx) expected KVCacheSimple")
                return -9
            }
            eval(bkv.keys, bkv.values)
        case .gdn(let bma):
            guard let mamba = transientCaches[layerIdx] as? MambaCache,
                  copyBatchedMambaLayer(src: mamba, dst: bma, B: B)
            else {
                print("[vsm] prefill_batched_uniform hybrid: layer \(layerIdx) expected MambaCache")
                return -10
            }
            eval(bma.convState, bma.recState)
        }
        transientCaches[layerIdx] = nil
    }

    // Final eval across all layers before returning.
    var toEval = [MLXArray]()
    for layer in hCaches.layers {
        switch layer {
        case .attention(let c):
            toEval.append(c.keys); toEval.append(c.values)
        case .gdn(let c):
            toEval.append(c.convState); toEval.append(c.recState)
        }
    }
    eval(toEval)

    // Engine state mirrors init_batched hybrid path: batchedHybridCaches
    // holds the polymorphic cache, batchedCaches stays nil.
    engine.batchedHybridCaches = hCaches
    engine.batchedCaches = nil
    engine.batchSlots.removeAll()
    engine.batchTokens = Array(repeating: 0, count: maxBatch)

    let firstTokensArr: [Int32] = (0..<B).map { firstTokens[$0].item(Int32.self) }
    for (slot, rid) in rids.enumerated() {
        engine.batchSlots[rid] = slot
        engine.batchTokens[slot] = Int(firstTokensArr[slot])
    }

    return 0
}

// MARK: - Hybrid (Qwen3Next-class) batched cache helpers

/// Copy a single per-request `KVCacheSimple` layer into the matching
/// `BatchedKVCache` slot. Mirrors the Qwen3 init path's per-layer copy.
private func copyAttentionLayerIntoSlot(
    src: KVCacheSimple, dst: BatchedKVCache, slot: Int
) -> Bool {
    let offset = src.offset
    guard let (k, v) = src.peek() else { return false }
    dst.keys[slot, 0..., ..<offset, 0...] = k[0]
    dst.values[slot, 0..., ..<offset, 0...] = v[0]
    dst.offsets[slot] = offset
    dst.active = max(dst.active, slot + 1)
    return true
}

/// Copy a single per-request `MambaCache` layer (conv + recurrent state)
/// into the matching `BatchedMambaCache` slot. Per-request MambaCache holds
/// `state[0]` as `[1, kernel-1, convDim]` conv state and `state[1]` as
/// `[1, Hv, Dv, Dk]` fp32 recurrent state — slice off the leading 1-dim
/// before writing into the batched [maxBatch, ...] tensors.
private func copyMambaLayerIntoSlot(
    src: MambaCache, dst: BatchedMambaCache, slot: Int
) -> Bool {
    let s = src.state
    // src may be empty (zero-length prompt) — leave the destination zeroed.
    if s.isEmpty {
        dst.active = max(dst.active, slot + 1)
        return true
    }
    guard s.count >= 2 else { return false }
    let conv = s[0]   // [1, kernel-1, convDim]
    let rec = s[1]    // [1, Hv, Dv, Dk] fp32
    dst.convState[slot, 0..., 0...] = conv[0]
    let recCast = (rec.dtype == .float32) ? rec[0] : rec[0].asType(.float32)
    dst.recState[slot, 0..., 0..., 0...] = recCast
    dst.active = max(dst.active, slot + 1)
    return true
}

/// Copy a batched `KVCacheSimple` layer (already shaped `[B, kvHeads, T, headDim]`
/// after a single batched forward) into `[..<B]` slots of a `BatchedKVCache`.
/// Used by `vsm_engine_prefill_batched_uniform` hybrid path — the input is
/// already batched, so we slice-assign the whole `[..<B]` range in one op
/// instead of looping per-slot.
private func copyBatchedAttentionLayer(
    src: KVCacheSimple, dst: BatchedKVCache, B: Int
) -> Bool {
    let offset = src.offset
    guard let (k, v) = src.peek() else { return false }
    // k, v are [B, kvHeads, offset, headDim] from a batched forward.
    dst.keys[..<B, 0..., ..<offset, 0...] = k
    dst.values[..<B, 0..., ..<offset, 0...] = v
    for i in 0..<B { dst.offsets[i] = offset }
    dst.active = max(dst.active, B)
    return true
}

/// Copy a batched `MambaCache` layer into `[..<B]` slots of a
/// `BatchedMambaCache`. After a batched forward, the per-request
/// `MambaCache.state` holds `state[0]: [B, kernel-1, convDim]` and
/// `state[1]: [B, Hv, Dv, Dk]` (fp32) — same leading-B layout as the
/// destination, so this is a single slice-assign per state tensor.
private func copyBatchedMambaLayer(
    src: MambaCache, dst: BatchedMambaCache, B: Int
) -> Bool {
    let s = src.state
    if s.isEmpty {
        // Fresh cache (zero-length input would not produce state); nothing
        // to copy. Slot range stays at the BatchedMambaCache zero-init.
        dst.active = max(dst.active, B)
        return true
    }
    guard s.count >= 2 else { return false }
    let conv = s[0]    // [B, kernel-1, convDim]
    let rec = s[1]     // [B, Hv, Dv, Dk] (fp32 after gatedDeltaUpdate)
    dst.convState[..<B, 0..., 0...] = conv
    let recCast = (rec.dtype == .float32) ? rec : rec.asType(.float32)
    dst.recState[..<B, 0..., 0..., 0...] = recCast
    dst.active = max(dst.active, B)
    return true
}

/// Hybrid analogue of `vsm_engine_init_batched`'s Qwen3 path. Walks each
/// session's per-layer cache list and copies into the matching
/// `BatchedHybridCache.layers[i]` (attention or GDN). Frees per-request
/// caches eagerly after each slot so total KV memory doesn't double.
private func initBatchedHybrid(
    engine: InferenceEngine,
    model: any BatchedHybridLLM,
    rids: [String]
) -> Int32 {
    let B = rids.count

    // Build the BatchedHybridCache. Respect scheduler max_num_seqs to avoid
    // over-allocating slots no one will use (drove the GB/turn unified-mem
    // leak when running with max_num_seqs=1).
    let maxBatch = max(B, engine.maxConcurrentRequests)
    // Mirror the prefill_batched_uniform path: thread kvScheme into the
    // batched-hybrid cache factory so turbo* schemes actually take effect.
    let (turboKB, turboVB) = batchedTurboBits(from: engine.generateParams)
    let hCaches = model.newBatchedHybridCache(
        maxBatch: maxBatch, parameters: engine.generateParams,
        turboKeyBits: turboKB, turboValueBits: turboVB)

    // Sanity: layer count must match the per-request cache layer count.
    guard let firstSession = engine.sessions[rids[0]] else { return Int32(0) }
    let numLayers = firstSession.iterator.cache.count
    guard numLayers == hCaches.layers.count else {
        print("[vsm] init_batched_hybrid: layer count mismatch — req=\(numLayers) hybrid=\(hCaches.layers.count)")
        return Int32(-1)
    }

    engine.batchSlots.removeAll()
    engine.batchTokens = Array(repeating: 0, count: maxBatch)

    for (slotIdx, rid) in rids.enumerated() {
        guard let session = engine.sessions[rid] else { continue }
        engine.batchSlots[rid] = slotIdx

        let tokenId = session.iterator.y.tokens.item(Int.self)
        engine.batchTokens[slotIdx] = tokenId

        for layerIdx in 0..<numLayers {
            let srcCache = session.iterator.cache[layerIdx]
            let dstLayer = hCaches.layers[layerIdx]
            switch dstLayer {
            case .attention(let bkv):
                guard let simple = srcCache as? KVCacheSimple,
                      copyAttentionLayerIntoSlot(src: simple, dst: bkv, slot: slotIdx)
                else {
                    print("[vsm] init_batched_hybrid: layer \(layerIdx) expected KVCacheSimple")
                    return Int32(-1)
                }
            case .gdn(let bma):
                guard let mamba = srcCache as? MambaCache,
                      copyMambaLayerIntoSlot(src: mamba, dst: bma, slot: slotIdx)
                else {
                    print("[vsm] init_batched_hybrid: layer \(layerIdx) expected MambaCache")
                    return Int32(-1)
                }
            }
        }

        // Materialize this slot's writes, then drop the per-req cache to
        // avoid the doubled-KV peak the Qwen3 path also defends against.
        var slotEval = [MLXArray]()
        for layer in hCaches.layers {
            switch layer {
            case .attention(let c):
                slotEval.append(c.keys); slotEval.append(c.values)
            case .gdn(let c):
                slotEval.append(c.convState); slotEval.append(c.recState)
            }
        }
        eval(slotEval)
        engine.sessions[rid] = nil
    }

    // Final eval to materialize the full batched cache before returning.
    var toEval = [MLXArray]()
    for layer in hCaches.layers {
        switch layer {
        case .attention(let c):
            toEval.append(c.keys); toEval.append(c.values)
        case .gdn(let c):
            toEval.append(c.convState); toEval.append(c.recState)
        }
    }
    eval(toEval)

    engine.batchedHybridCaches = hCaches
    print("[vsm] Batched hybrid cache initialized: B=\(B), layers=\(numLayers), maxBatch=\(maxBatch)")
    return Int32(B)
}

/// Hybrid analogue of `vsm_engine_add_batch_slot`. Copies one new request's
/// per-layer caches into the next free `BatchedHybridCache` slot, then evals
/// + drops the per-request cache to avoid doubled KV memory.
private func addBatchSlotHybrid(
    engine: InferenceEngine,
    hCaches: BatchedHybridCache,
    rid: String,
    session: RequestSession
) -> Int32 {
    let slotIdx = engine.batchSlots.count
    let numLayers = session.iterator.cache.count
    guard numLayers == hCaches.layers.count else { return Int32(-1) }

    let tokenId = session.iterator.y.tokens.item(Int.self)
    engine.batchTokens[slotIdx] = tokenId
    engine.batchSlots[rid] = slotIdx

    for layerIdx in 0..<numLayers {
        let srcCache = session.iterator.cache[layerIdx]
        let dstLayer = hCaches.layers[layerIdx]
        switch dstLayer {
        case .attention(let bkv):
            guard let simple = srcCache as? KVCacheSimple,
                  copyAttentionLayerIntoSlot(src: simple, dst: bkv, slot: slotIdx)
            else { return Int32(-1) }
        case .gdn(let bma):
            guard let mamba = srcCache as? MambaCache,
                  copyMambaLayerIntoSlot(src: mamba, dst: bma, slot: slotIdx)
            else { return Int32(-1) }
        }
    }

    var toEval = [MLXArray]()
    for layer in hCaches.layers {
        switch layer {
        case .attention(let c):
            toEval.append(c.keys); toEval.append(c.values)
        case .gdn(let c):
            toEval.append(c.convState); toEval.append(c.recState)
        }
    }
    eval(toEval)
    engine.sessions[rid] = nil

    return Int32(slotIdx)
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
