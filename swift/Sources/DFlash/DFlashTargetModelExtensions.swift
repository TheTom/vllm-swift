// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Model conformance examples for DFlashTargetModel
//
// To enable DFlash on your model, extend it with DFlashTargetModel conformance:

/*
 ## Example: Qwen3Model Conformance

 Add this to your model extension file (requires MLXLLM):

 ```swift
 import MLXLLM

 extension Qwen3Model: DFlashTargetModel {
     public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
         model.embedTokens(tokens)
     }

     public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
         if let lmHead {
             return lmHead(hiddenStates)
         }
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

     public var dflashIsHybridGDN: Bool { false }
 }
 ```

 ## Example: DeepseekV3DFlashModel (Hybrid GDN)

 For hybrid GDN models, set `dflashIsHybridGDN = true`:

 ```swift
 extension DeepseekV3DFlashModel: DFlashTargetModel {
     public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
         model.embedTokens(tokens)
     }

     public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
         model.lmHead(hiddenStates)
     }

     public func dflashForwardWithCapture(
         inputIDs: MLXArray,
         cache: [KVCache],
         captureLayerIDs: Set<Int>
     ) -> (MLXArray, [Int: MLXArray]) {
         // Use GDN-aware forward pass
         let (logits, captured) = model.forwardWithCaptureGDN(
             inputIDs: inputIDs,
             cache: cache,
             captureLayerIDs: captureLayerIDs
         )
         return (logits, captured)
     }

     public var dflashIsHybridGDN: Bool { true }
 }
 ```
*/

// No concrete implementations here - see the documentation above
// and model-specific files (Qwen3+DFlash.swift, DeepseekV3DFlash.swift, etc.)