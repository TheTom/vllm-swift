// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// DFlashTargetModel conformance for MLXLLM models
//
// This file provides DFlash support for all MLXLLM models.
// Due to Swift access control, conformance extensions should ideally be added
// within the MLXLLM package itself, but this file provides them for use with
// the DFlash module when imported together with MLXLLM.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM

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

// MARK: - Embedding Extension for Tied Weights

extension Embedding {
    /// Convert embeddings to logits using tied weights (transpose + matmul).
    public func asLinear(_ x: MLXArray) -> MLXArray {
        let weightT = transposed(weight, axes: [1, 0])
        return matmul(x, weightT)
    }
}

// MARK: - Model Registry

/// Registry of models with their DFlash characteristics.
public enum DFlashModelRegistry {
    /// Pure attention models - use FullAttentionEngine
    public static let pureAttentionModels: [String] = [
        // Llama family
        "LlamaModel",
        // Qwen family (pure attention)
        "Qwen3Model",
        "Qwen2Model",
        // Gemma family
        "GemmaModel",
        "Gemma2Model",
        "Gemma3TextModel",
        "Gemma4Model",
        "Gemma3nTextModel",
        // Phi family
        "PhiModel",
        "Phi3Model",
        "PhiMoEModel",
        // Other pure models
        "MistralModel",
        "Mistral3TextModel",
        "CohereModel",
        "Starcoder2Model",
        "SmolLMModel",
        "NanoChatModel",
        "Internlm2Model",
        "BaichuanM1Model",
        "NemotronHModel",
        "OpenELMModel",
        "OlmoModel",
        "Olmo2Model",
        "Olmo3Model",
        "OlmoE",
        "GraniteModel",
        "BitnetModel",
        "FalconH1Model",
        "Exaone4Model",
        "Ernie45Model",
        "GPTOSSModel",
        "ApertusModel",
        "JambaModel",
    ]
    
    /// Hybrid models with GDN/SSM layers - use HybridGDNEngine
    public static let hybridModels: [String] = [
        // Qwen hybrid models
        "Qwen35Model",
        "Qwen3MoEModel",
        "Qwen3NextModel",
        // DeepSeek family
        "DeepseekV3Model",
        // MiniMax family
        "MiniMaxModel",
        "MiniMaxM2Model",
        // Other hybrid MoE models
        "GraniteMoeHybridModel",
        "LFM2Model",
        "LFM2MoEModel",
        "AfMoEModel",
        "GLM4MoEModel",
        "GLM4MoELiteModel",
        "GLM4Model",
        "BailingMoeModel",
        "MiniCPMModel",
        "MiMoModel",
        "MiMoV2FlashModel",
    ]
}

// MARK: - Supported Models List

/// Complete list of models that support DFlash when extended.
public enum DFlashSupportedModels {
    
    // MARK: Pure Attention Models (dflashIsHybridGDN = false)
    
    /// All pure attention models
    public static var allPure: [String] {
        DFlashModelRegistry.pureAttentionModels
    }
    
    // MARK: Hybrid Models (dflashIsHybridGDN = true)
    
    /// All hybrid models
    public static var allHybrid: [String] {
        DFlashModelRegistry.hybridModels
    }
    
    /// All models combined
    public static var all: [String] {
        allPure + allHybrid
    }
}

// MARK: - Conformance Status

/// Tracks which models have DFlash conformance implemented.
public struct DFlashConformanceStatus {
    /// Models with full conformance implemented.
    public static let implemented: Set<String> = []
    
    /// Models with partial conformance (missing forwardWithCapture).
    public static let partial: Set<String> = []
    
    /// Models not yet extended (require MLXLLM changes).
    public static let pending: Set<String> = Set(DFlashSupportedModels.all)
}

// MARK: - Conformance Template Generator

/// Generate DFlash conformance extension code for a model.
public func generateDFlashConformance(
    modelName: String,
    isHybrid: Bool,
    useCallCapturing: Bool = false
) -> String {
    let hybridFlag = isHybrid ? "true" : "false"
    
    if useCallCapturing {
        return """
        extension \(modelName): DFlashTargetModel {
            public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
                model.embedTokens(tokens)
            }
            
            public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
                if let lmHead { return lmHead(hiddenStates) }
                return model.embedTokens.asLinear(hiddenStates)
            }
            
            public func dflashForwardWithCapture(
                inputIDs: MLXArray,
                cache: [KVCache],
                captureLayerIDs: Set<Int>
            ) -> (MLXArray, [Int: MLXArray]) {
                let cacheOpt: [KVCache?] = cache.map { $0 }
                let (hiddenStates, captured) = model.callCapturing(
                    inputIDs, cache: cacheOpt, captureLayerIDs: captureLayerIDs)
                return (dflashLmHeadLogits(hiddenStates), captured)
            }
            
            public var dflashIsHybridGDN: Bool { \(hybridFlag) }
        }
        
        extension \(modelName): DFlashForwardWithCapture {
            public func forwardWithCapture(
                inputIDs: MLXArray,
                cache: [KVCache?],
                captureLayerIDs: Set<Int>
            ) -> (MLXArray, [Int: MLXArray]) {
                let (hiddenStates, captured) = model.callCapturing(
                    inputIDs, cache: cache, captureLayerIDs: captureLayerIDs)
                return (dflashLmHeadLogits(hiddenStates), captured)
            }
        }
        """
    } else {
        return """
        extension \(modelName): DFlashTargetModel {
            public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
                model.embedTokens(tokens)
            }
            
            public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
                if let lmHead { return lmHead(hiddenStates) }
                return model.embedTokens.asLinear(hiddenStates)
            }
            
            public func dflashForwardWithCapture(
                inputIDs: MLXArray,
                cache: [KVCache],
                captureLayerIDs: Set<Int>
            ) -> (MLXArray, [Int: MLXArray]) {
                var h = model.embedTokens(inputIDs)
                var captured: [Int: MLXArray] = [:]
                for (i, layer) in model.layers.enumerated() {
                    h = layer(h, cache: cache[i])
                    if captureLayerIDs.contains(i + 1) {
                        captured[i + 1] = h
                    }
                }
                let normed = model.norm(h)
                let logits: MLXArray
                if let head = lmHead { logits = head(normed) }
                else { logits = model.embedTokens.asLinear(normed) }
                return (logits, captured)
            }
            
            public var dflashIsHybridGDN: Bool { \(hybridFlag) }
        }
        
        extension \(modelName): DFlashForwardWithCapture {
            public func forwardWithCapture(
                inputIDs: MLXArray,
                cache: [KVCache?],
                captureLayerIDs: Set<Int>
            ) -> (MLXArray, [Int: MLXArray]) {
                var h = model.embedTokens(inputIDs)
                var captured: [Int: MLXArray] = [:]
                for (i, layer) in model.layers.enumerated() {
                    h = layer(h, cache: cache[i])
                    if captureLayerIDs.contains(i + 1) {
                        captured[i + 1] = h
                    }
                }
                let normed = model.norm(h)
                let logits: MLXArray
                if let head = lmHead { logits = head(normed) }
                else { logits = model.embedTokens.asLinear(normed) }
                return (logits, captured)
            }
        }
        """
    }
}
