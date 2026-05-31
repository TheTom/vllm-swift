// SPDX-License-Identifier: Apache-2.0
//
// FFAI-backed engine path: serves the GGUF DeepSeek-V4-Flash model through
// FFAI (Swift) → metaltile (Metal kernels). This is the `vllm-swift → FFAI
// → metaltile` route — native Swift→Swift, no C glue (the only C-ABI edge
// stays at the Python↔dylib boundary in Bridge.swift's @_cdecl funcs).
//
// The MLX path (InferenceEngine, mlx-swift-lm) is untouched: DSv4-GGUF
// models register an FFAIDsv4Engine in a parallel handle map and the
// @_cdecl entry points branch to it when present. FFAI today is
// single-stream, sliding-window-128 decode (CSA/HCA long-context is the
// separate Track B); so only the single-stream calls route here — batched
// paths stay on MLX.

import FFAI
import Foundation

/// Single-stream FFAI decode engine for DeepSeek-V4-Flash GGUF.
final class FFAIDsv4Engine: @unchecked Sendable {
    let model: DeepSeekV4Model
    let bundle: GGUFTensorBundle
    var state: DeepSeekV4Model.DecodeState
    let vocabSize: Int
    let numLayers: Int
    /// Stable storage for the last forward's logits, so `vsm_engine_get_logits`
    /// can hand Python a pointer that stays valid until the next step.
    let logitsBuf: UnsafeMutableBufferPointer<Float>

    private init(model: DeepSeekV4Model, bundle: GGUFTensorBundle, vocab: Int, layers: Int) {
        self.model = model
        self.bundle = bundle
        self.vocabSize = vocab
        self.numLayers = layers
        self.logitsBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: vocab)
        self.logitsBuf.initialize(repeating: 0)
        model.keepLayersResident = true
        self.state = model.makeDecodeState()
    }

    deinit { logitsBuf.deallocate() }

    /// Run one forward, copy logits into the stable buffer, return greedy argmax.
    private func forwardAndArgmax(_ tokenId: Int) -> Int32 {
        guard let logits = try? model.forwardAllLayers(inputTokenId: tokenId, state: state) else {
            return -1
        }
        let host = logits.toFloatArray()
        let n = min(host.count, vocabSize)
        if n > 0 { host.withUnsafeBufferPointer { logitsBuf.baseAddress!.update(from: $0.baseAddress!, count: n) } }
        var maxIdx = 0
        var maxVal = -Float.infinity
        for i in 0..<n where host[i] > maxVal { maxVal = host[i]; maxIdx = i }
        return Int32(maxIdx)
    }

    /// Last sampled token — `decode_step` (which takes no token arg) feeds
    /// this, mirroring the MLX session iterator's internal carry.
    var lastToken: Int32 = -1

    /// Prefill: feed the prompt sequentially (FFAI has no batched prefill);
    /// the final token's forward yields the first sampled token.
    func prefill(tokens: [Int]) -> Int32 {
        guard !tokens.isEmpty else { return -1 }
        for t in tokens.dropLast() { _ = forwardAndArgmax(t) }
        lastToken = forwardAndArgmax(tokens.last!)
        return lastToken
    }

    /// Returns an engine iff `modelPath` is a DeepSeek-V4 GGUF directory (or
    /// .gguf file). Returns nil for everything else so the caller falls
    /// through to the MLX factory path. Cheap — only reads the GGUF header.
    static func tryLoad(modelPath: String) -> FFAIDsv4Engine? {
        let url = URL(fileURLWithPath: modelPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir) else { return nil }
        let bundle: GGUFTensorBundle
        do {
            bundle = isDir.boolValue
                ? try GGUFTensorBundle(directory: url)
                : try GGUFTensorBundle(url: url)
        } catch {
            return nil  // not a GGUF — MLX path handles it
        }
        // Only claim DeepSeek-V4 architectures.
        let arch = bundle.architecture ?? ""
        guard arch == "deepseek4" || arch.contains("DeepSeekV4") || arch.contains("DeepseekV4") else {
            return nil
        }
        do {
            let model = try DeepSeekV4Flash.loadFlashFromGGUF(gguf: bundle, device: .shared)
            let vocab = Int(bundle.reader.metadataUInt32("deepseek4.vocab_size")
                ?? bundle.reader.metadataUInt32("vocab_size") ?? 129_280)
            // 43 layers for Flash; pull from metadata if present.
            let layers = Int(bundle.reader.metadataUInt32("deepseek4.block_count")
                ?? bundle.reader.metadataUInt32("block_count") ?? 43)
            print("[vsm-ffai] loaded DSv4-Flash GGUF: vocab=\(vocab) layers=\(layers)")
            return FFAIDsv4Engine(model: model, bundle: bundle, vocab: vocab, layers: layers)
        } catch {
            print("[vsm-ffai] DSv4 GGUF load failed: \(error)")
            return nil
        }
    }

    /// One decode step: feed the last sampled token, return greedy argmax
    /// (logits also land in `logitsBuf` for `get_logits`).
    func decodeStep() -> Int32 {
        guard lastToken >= 0 else { return -1 }
        lastToken = forwardAndArgmax(Int(lastToken))
        return lastToken
    }

    /// Reset decode state for a fresh sequence.
    func reset() {
        state = model.makeDecodeState()
        lastToken = -1
    }
}

/// Parallel handle map for FFAI engines (kept separate from the MLX
/// `engines` map so the MLX path is entirely unaffected).
let ffaiEngineQueue = DispatchQueue(label: "vsm.ffai.engines")
nonisolated(unsafe) var ffaiEngines: [UnsafeMutableRawPointer: FFAIDsv4Engine] = [:]
