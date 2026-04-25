// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Based on DFlash (arXiv:2602.06036)
// vllm-swift DFlash implementation with extensible abstractions

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DFlash Target Model Protocol

/// Protocol that target models can conform to in order to expose their
/// internal structure for DFlash speculative decoding.
///
/// The DFlash runtime needs to:
/// 1. Access the embedding layer for draft noise embeddings
/// 2. Access the lm_head for draft logits
/// 3. Run a custom forward pass that captures intermediate hidden states
/// 4. Determine if the model has hybrid GDN layers
public protocol DFlashTargetModel: LanguageModel {
    /// Embed token IDs and return the embedding vectors.
    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray

    /// Compute logits from hidden states (via lm_head or tied weights).
    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray

    /// Run a forward pass capturing hidden states at the specified layer indices.
    ///
    /// - Parameters:
    ///   - inputIDs: Input token IDs [1, seqLen]
    ///   - cache: The KV cache array
    ///   - captureLayerIDs: Set of 0-based layer indices whose output to capture
    /// - Returns: Tuple of (logits, captured hidden states keyed by layerID+1)
    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray])

    /// Whether the model contains hybrid GatedDeltaNet layers.
    var dflashIsHybridGDN: Bool { get }

    /// Whether the hybrid GDN layers should use full innovation-tape rollback
    /// (RecurrentRollbackCache) vs lightweight snapshot-only rollback.
    /// Default: true (tape rollback).
    var dflashUseTapeRollback: Bool { get }
}

public extension DFlashTargetModel {
    var dflashUseTapeRollback: Bool { true }
}

// MARK: - DFlash Draft Model Protocol

/// Protocol for DFlash draft models.
///
/// Draft models take noise token embeddings (from the target model's embed_tokens)
/// and target hidden states, and produce draft logits for block-diffusion speculative decoding.
public protocol DFlashDraftModelProtocol {
    /// Number of tokens per draft block
    var blockSize: Int { get }

    /// Mask token ID used during drafting
    var maskTokenID: Int { get }

    /// Target layer indices used for context feature extraction
    var targetLayerIDs: [Int] { get }

    /// Run the draft model forward pass.
    func forwardDraft(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        cache: [any DFlashDraftCacheProtocol]?
    ) -> MLXArray
}

// MARK: - DFlash Draft Cache Protocol

/// Protocol for DFlash draft model KV caches.
/// Draft caches store context keys/values for cross-attention during drafting.
public protocol DFlashDraftCacheProtocol: AnyObject {
    /// Current cache length
    var cacheLength: Int { get }

    /// Append context keys/values to the cache.
    func appendContext(
        contextKeys: MLXArray,
        contextValues: MLXArray,
        numPositions: Int
    )

    /// Fetch cached keys and values.
    func fetch() -> (MLXArray?, MLXArray?)
}

// MARK: - DFlash Rollback Cache Protocol

/// Protocol for rollback-capable caches used in hybrid GDN models.
public protocol DFlashRollbackCacheProtocol: AnyObject {
    var isArmed: Bool { get }
    func armRollback(prefixLen: Int)
    func rollback(nAccepted: Int)
    func clearTransients()
}

// MARK: - DFlash Engine Protocol

/// Protocol for DFlash verify/rollback engines.
public protocol DFlashEngineProtocol: Sendable {
    /// Arm the target model's cache for rollback before verification.
    func armRollback(targetCache: [KVCache], prefixLen: Int)

    /// Roll back the target cache after partial acceptance.
    func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int
}

// MARK: - DFlash Generation Event

/// Events emitted during DFlash generation.
public enum DFlashEvent: Sendable {
    /// Prefill completed
    case prefill(promptTokenCount: Int, prefillUs: Double)
    /// Prefill progress (chunked)
    case prefillProgress(tokensProcessed: Int, tokensTotal: Int)
    /// A token was generated
    case token(tokenID: Int, generatedTokens: Int, acceptanceRatio: Double, cyclesCompleted: Int)
    /// Generation summary
    case summary(DFlashSummary)
}

/// Summary statistics for a DFlash generation run.
public struct DFlashSummary: Sendable {
    public let elapsedUs: Double
    public let promptTokenCount: Int
    public let generatedTokenIDs: [Int]
    public let acceptedFromDraft: Int
    public let acceptanceRatio: Double
    public let blockTokens: Int
    public let cyclesCompleted: Int
    public let phaseTimingsUs: PhaseTimings

    public struct PhaseTimings: Sendable {
        public let prefill: Double
        public let draft: Double
        public let verify: Double
        public let replay: Double
    }

    public var generationTokens: Int { generatedTokenIDs.count }
    public var tokensPerSecond: Double {
        let genUs = elapsedUs - phaseTimingsUs.prefill
        return genUs > 0 ? Double(generationTokens) / (genUs / 1_000_000.0) : 0
    }
}

// MARK: - DFlash Configuration

/// Configuration for DFlash speculative decoding.
public struct DFlashConfiguration: Sendable {
    /// Number of tokens per draft block (default: from draft model)
    public var blockTokens: Int?

    /// Stop token IDs that signal end of generation
    public var stopTokenIDs: [Int] = []

    /// Token IDs to suppress during generation
    public var suppressTokenIDs: [Int]?

    /// Sink tokens to keep in draft cache
    public var draftSinkSize: Int = 64

    /// Sliding window size for draft cache
    public var draftWindowSize: Int = 1024

    /// Use tape-based rollback for hybrid GDN models (more accurate, ~30% slower)
    public var useTapeRollback: Bool = true

    public init(
        blockTokens: Int? = nil,
        stopTokenIDs: [Int] = [],
        suppressTokenIDs: [Int]? = nil,
        draftSinkSize: Int = 64,
        draftWindowSize: Int = 1024,
        useTapeRollback: Bool = true
    ) {
        self.blockTokens = blockTokens
        self.stopTokenIDs = stopTokenIDs
        self.suppressTokenIDs = suppressTokenIDs
        self.draftSinkSize = draftSinkSize
        self.draftWindowSize = draftWindowSize
        self.useTapeRollback = useTapeRollback
    }
}

// MARK: - Context Feature Extraction

/// Extract and concatenate hidden states at the specified layer IDs.
/// The layer IDs are 0-indexed into the model's layers, and we take
/// `hiddenStates[layerID + 1]` because index 0 is the embedding output.
public func extractContextFeature(
    hiddenStates: [MLXArray],
    layerIDs: [Int]
) -> MLXArray {
    let selected = layerIDs.map { hiddenStates[$0 + 1] }
    return concatenated(selected, axis: -1)
}

/// Extract context feature from a dictionary of captured hidden states.
public func extractContextFeatureFromDict(
    capturedDict: [Int: MLXArray],
    targetLayerIDs: [Int]
) -> MLXArray {
    let selected = targetLayerIDs.map { capturedDict[$0 + 1]! }
    return concatenated(selected, axis: -1)
}
