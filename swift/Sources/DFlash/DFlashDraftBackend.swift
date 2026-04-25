// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Based on DFlash (arXiv:2602.06036)
// vllm-swift DFlash draft generation backend

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Draft Backend

/// Backend for generating draft tokens using the DFlash draft model.
public final class DFlashDraftBackend: @unchecked Sendable {

    public init() {}

    /// Create the draft cache (one `ContextOnlyDraftKVCache` per layer).
    public func makeCache(
        draftModel: any DFlashDraftModelProtocol,
        sinkSize: Int = 64,
        windowSize: Int = 1024
    ) -> [ContextOnlyDraftKVCache] {
        // Get the number of layers from the draft model
        var numLayers = 0
        if let dflashModel = draftModel as? DFlashDraftModel {
            numLayers = dflashModel.layers.count
        }
        return (0 ..< numLayers).map { _ in
            ContextOnlyDraftKVCache(sinkSize: sinkSize, windowSize: windowSize)
        }
    }

    /// Generate draft tokens greedily using the DFlash draft model.
    ///
    /// - Parameters:
    ///   - targetModel: The target model (must conform to DFlashTargetModel for embed/lm_head access)
    ///   - draftModel: The DFlash draft model
    ///   - draftCache: The draft model's KV caches
    ///   - stagedFirst: The first token (already verified by the target)
    ///   - targetHidden: The target model's hidden states for context
    ///   - blockLen: Number of tokens to draft
    ///   - maskTokenTail: Mask token IDs for positions 1..blockLen-1
    ///   - suppressTokenMask: Optional mask to suppress certain tokens
    /// - Returns: Draft token IDs [blockLen-1]
    public func draftGreedy(
        targetModel: any DFlashTargetModel,
        draftModel: any DFlashDraftModelProtocol,
        draftCache: [ContextOnlyDraftKVCache],
        stagedFirst: MLXArray,
        targetHidden: MLXArray,
        blockLen: Int,
        maskTokenTail: MLXArray,
        suppressTokenMask: MLXArray? = nil
    ) -> MLXArray {
        precondition(blockLen > 1, "draftGreedy requires blockLen > 1")

        let blockTokenIDs = concatenated(
            [stagedFirst[..<1], maskTokenTail[..<(blockLen - 1)]],
            axis: 0
        )

        // Get noise embedding from target model's embed_tokens
        let noiseEmbedding = targetModel.dflashEmbedTokens(blockTokenIDs[.newAxis])

        // Run the draft model
        let draftHidden = draftModel.forwardDraft(
            noiseEmbedding: noiseEmbedding,
            targetHidden: targetHidden,
            cache: draftCache
        )

        // Get draft logits via the target model's lm_head
        let draftLogits = targetModel.dflashLmHeadLogits(
            draftHidden[.ellipsis, 1..., 0...]
        )

        // Greedy decode
        let drafted = DFlashRuntime.greedyTokensWithMask(
            logits: draftLogits,
            suppressTokenMask: suppressTokenMask
        ).squeezed(axis: 0)

        asyncEval(drafted)
        return drafted
    }
}
