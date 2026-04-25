// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// DFlashTargetModel conformance for MLXLLM models
//
// This file provides documentation and helpers for adding DFlash support to MLXLLM models.
// Due to Swift access control, model internals (embedTokens, layers, norm, lmHead) 
// are internal to the MLXLLM package. The conformance extensions must be added to
// the MLXLLM package itself.
//
// MARK: - Forward with Capture Protocol

/// Protocol for models that support capturing intermediate hidden states.
/// Models implementing this protocol can be used with DFlash for speculative decoding.
public protocol DFlashForwardWithCapture: LanguageModel {
    /// Run a forward pass that captures hidden states at specified layer indices.
    func forwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache?],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray])
}

// MARK: - Model Conformance Template

/// Template for adding DFlash conformance to a model.
/// Copy this template to the model's Swift file in MLXLLM and fill in the specifics.
///
/// ## Usage
///
/// Add the following extension to any model file in MLXLLM/Models/:
///
/// ```swift
/// // For pure attention models:
/// extension YourModel: DFlashTargetModel {
///     public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
///         model.embedTokens(tokens)
///     }
///
///     public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
///         if let lmHead { return lmHead(hiddenStates) }
///         return model.embedTokens.asLinear(hiddenStates)
///     }
///
///     public func dflashForwardWithCapture(
///         inputIDs: MLXArray,
///         cache: [KVCache],
///         captureLayerIDs: Set<Int>
///     ) -> (MLXArray, [Int: MLXArray]) {
///         var h = model.embedTokens(inputIDs)
///         var captured: [Int: MLXArray] = [:]
///         for (i, layer) in model.layers.enumerated() {
///             h = layer(h, cache: cache[i])
///             if captureLayerIDs.contains(i + 1) {
///                 captured[i + 1] = h
///             }
///         }
///         let normed = model.norm(h)
///         let logits: MLXArray
///         if let head = lmHead { logits = head(normed) }
///         else { logits = model.embedTokens.asLinear(normed) }
///         return (logits, captured)
///     }
///
///     public var dflashIsHybridGDN: Bool { false }
/// }
///
/// extension YourModel: DFlashForwardWithCapture {
///     public func forwardWithCapture(
///         inputIDs: MLXArray,
///         cache: [KVCache?],
///         captureLayerIDs: Set<Int>
///     ) -> (MLXArray, [Int: MLXArray]) {
///         // Same implementation as dflashForwardWithCapture but with optional cache
///         var h = model.embedTokens(inputIDs)
///         var captured: [Int: MLXArray] = [:]
///         for (i, layer) in model.layers.enumerated() {
///             h = layer(h, cache: cache[i])
///             if captureLayerIDs.contains(i + 1) {
///                 captured[i + 1] = h
///             }
///         }
///         let normed = model.norm(h)
///         let logits: MLXArray
///         if let head = lmHead { logits = head(normed) }
///         else { logits = model.embedTokens.asLinear(normed) }
///         return (logits, captured)
///     }
/// }
/// ```
public struct DFlashModelConformanceTemplate {
    // This is a documentation struct - see above for usage
    
    /// Pure attention models (set dflashIsHybridGDN = false)
    public static let pureAttentionModels: [String] = [
        "Qwen3Model",
        "Qwen2Model",
        "LlamaModel",
        "GemmaModel",
        "Gemma2Model",
        "Gemma3Model",
        "Gemma4Model",
        "PhiModel",
        "Phi3Model",
        "CohereModel",
        "Starcoder2Model",
        "SmolLMModel",
        "NanoChatModel",
        "Internlm2Model",
        "BaichuanM1Model",
        "Mistral3TextModel",
    ]
    
    /// Hybrid GDN models (set dflashIsHybridGDN = true)
    public static let hybridModels: [String] = [
        "Qwen35Model",          // Qwen3.5 MoE
        "Qwen3MoEModel",
        "Qwen3NextModel",
        "DeepseekV3Model",
        "MiniMaxModel",
        "MiniMaxM2Model",
        "GraniteMoeHybridModel",
        "LFM2Model",
        "LFM2MoEModel",
        "AfMoEModel",
        "GLM4MoEModel",
        "GLM4MoELiteModel",
    ]
}

import MLXNN

// MARK: - Helper Extension

extension Embedding {
    /// Convert embeddings to logits using tied weights.
    public func asLinear(_ x: MLXArray) -> MLXArray {
        let weightT = transposed(weight, axes: [1, 0])
        return matmul(x, weightT)
    }
}
