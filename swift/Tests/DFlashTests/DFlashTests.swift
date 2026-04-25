// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Tests for DFlash speculative decoding module

import XCTest
@testable import DFlash
import MLX
import MLXLMCommon
import MLXNN

final class DFlashTests: XCTestCase {

    // MARK: - Token Utilities Tests

    func testGreedyTokensWithMask() {
        // Create logits [1.0, 4.0, 2.0, 3.0] - token 1 has highest logit
        let logits = MLXArray([1.0, 5.0, 2.0, 3.0]).reshaped([1, 4])
        let result = DFlashRuntime.greedyTokensWithMask(logits: logits)
        XCTAssertEqual(result.item(Int.self), 1) // argmax of [1,5,2,3] is index 1
    }

    func testGreedyTokensWithSuppressMask() {
        // Create logits [1.0, 4.0, 2.0, 3.0] - token 1 has highest logit
        let logits = MLXArray([1.0, 5.0, 2.0, 3.0])
        // Suppress token 1
        let suppressMask = MLXArray([false, true, false, false])
        let result = DFlashRuntime.greedyTokensWithMask(
            logits: logits,
            suppressTokenMask: suppressMask
        )
        // Token 3 should be selected (3.0) instead of token 1 (5.0)
        XCTAssertEqual(result.item(Int.self), 3)
    }

    func testMatchAcceptanceLength() {
        // drafted = [1, 2, 3], posterior = [1, 2, 5]
        let drafted = MLXArray([1, 2, 3])
        let posterior = MLXArray([1, 2, 5])
        let result = DFlashRuntime.matchAcceptanceLength(
            draftedTokens: drafted,
            posteriorTokens: posterior
        )
        XCTAssertEqual(result.item(Int.self), 2) // first two tokens match
    }

    func testMatchAcceptanceLengthFullMatch() {
        let drafted = MLXArray([1, 2, 3])
        let posterior = MLXArray([1, 2, 3])
        let result = DFlashRuntime.matchAcceptanceLength(
            draftedTokens: drafted,
            posteriorTokens: posterior
        )
        XCTAssertEqual(result.item(Int.self), 3) // all match
    }

    func testMatchAcceptanceLengthNoMatch() {
        let drafted = MLXArray([1, 2, 3])
        let posterior = MLXArray([4, 5, 6])
        let result = DFlashRuntime.matchAcceptanceLength(
            draftedTokens: drafted,
            posteriorTokens: posterior
        )
        XCTAssertEqual(result.item(Int.self), 0) // no match
    }

    func testMatchAcceptanceLengthEmpty() {
        // Create empty arrays with explicit shape
        let emptyArr = MLXArray.zeros([0], dtype: .int32)
        let drafted = emptyArr
        let posterior = emptyArr
        let result = DFlashRuntime.matchAcceptanceLength(
            draftedTokens: drafted,
            posteriorTokens: posterior
        )
        XCTAssertEqual(result.item(Int.self), 0)
    }

    // MARK: - Build Suppress Token Mask

    func testBuildSuppressTokenMask() {
        let vocabSize = 100
        let suppressIDs: [Int] = [5, 10, 15]
        let mask = DFlashRuntime.buildSuppressTokenMask(
            vocabSize: vocabSize,
            suppressTokenIDs: suppressIDs
        )
        XCTAssertNotNil(mask)
        let arr = mask!.asArray(Bool.self)
        XCTAssertTrue(arr[5])
        XCTAssertTrue(arr[10])
        XCTAssertTrue(arr[15])
        XCTAssertFalse(arr[0])
        XCTAssertFalse(arr[50])
    }

    func testBuildSuppressTokenMaskEmpty() {
        let mask = DFlashRuntime.buildSuppressTokenMask(
            vocabSize: 100,
            suppressTokenIDs: nil
        )
        XCTAssertNil(mask)
    }

    func testBuildSuppressTokenMaskOutOfBounds() {
        // Out of bounds IDs should be filtered out
        let mask = DFlashRuntime.buildSuppressTokenMask(
            vocabSize: 100,
            suppressTokenIDs: [5, 150, 200, 10] // 150 and 200 out of bounds
        )
        XCTAssertNotNil(mask)
        let arr = mask!.asArray(Bool.self)
        XCTAssertTrue(arr[5])
        XCTAssertTrue(arr[10])
        XCTAssertFalse(arr[150])
        XCTAssertFalse(arr[200])
    }

    // MARK: - Draft Configuration

    func testBuildTargetLayerIDs() {
        let ids = buildTargetLayerIDs(numTargetLayers: 36, numDraftLayers: 4)
        XCTAssertEqual(ids.count, 4)
        // First should be near 1, last should be near 33
        XCTAssertGreaterThan(ids[0], 0)
        XCTAssertLessThan(ids[3], 36)
        // Should be roughly evenly spaced
        XCTAssertTrue(ids[0] < ids[1])
        XCTAssertTrue(ids[1] < ids[2])
        XCTAssertTrue(ids[2] < ids[3])
    }

    func testBuildTargetLayerIDsSingleLayer() {
        let ids = buildTargetLayerIDs(numTargetLayers: 36, numDraftLayers: 1)
        XCTAssertEqual(ids.count, 1)
        XCTAssertEqual(ids[0], 18) // middle of 36
    }

    // MARK: - DFlash Configuration

    func testDFlashConfigurationDefaults() {
        let config = DFlashConfiguration()
        XCTAssertNil(config.blockTokens)
        XCTAssertTrue(config.stopTokenIDs.isEmpty)
        XCTAssertNil(config.suppressTokenIDs)
        XCTAssertEqual(config.draftSinkSize, 64)
        XCTAssertEqual(config.draftWindowSize, 1024)
        XCTAssertTrue(config.useTapeRollback)
    }

    func testDFlashConfigurationCustom() {
        let config = DFlashConfiguration(
            blockTokens: 8,
            stopTokenIDs: [2, 3],
            suppressTokenIDs: [100, 200],
            draftSinkSize: 32,
            draftWindowSize: 512,
            useTapeRollback: false
        )
        XCTAssertEqual(config.blockTokens, 8)
        XCTAssertEqual(config.stopTokenIDs, [2, 3])
        XCTAssertEqual(config.suppressTokenIDs, [100, 200])
        XCTAssertEqual(config.draftSinkSize, 32)
        XCTAssertEqual(config.draftWindowSize, 512)
        XCTAssertFalse(config.useTapeRollback)
    }

    // MARK: - Draft Cache Tests

    func testContextOnlyDraftKVCache() {
        let cache = ContextOnlyDraftKVCache(sinkSize: 4, windowSize: 8)
        XCTAssertEqual(cache.cacheLength, 0)
        XCTAssertEqual(cache.offset, 0)

        // Append some context
        let keys = MLXArray.zeros([1, 2, 1, 128])
        let values = MLXArray.zeros([1, 2, 1, 128])
        cache.appendContext(contextKeys: keys, contextValues: values, numPositions: 2)

        XCTAssertEqual(cache.cacheLength, 2)
        XCTAssertEqual(cache.offset, 2)
    }

    func testContextOnlyDraftKVCacheWindowing() {
        let cache = ContextOnlyDraftKVCache(sinkSize: 4, windowSize: 8)

        // Append enough context to trigger windowing
        let keys = MLXArray.zeros([1, 20, 1, 128])
        let values = MLXArray.zeros([1, 20, 1, 128])
        cache.appendContext(contextKeys: keys, contextValues: values, numPositions: 20)

        // Should have sink + window
        XCTAssertEqual(cache.cacheLength, 12) // 4 + 8
    }

    // MARK: - Context Feature Extraction

    func testExtractContextFeature() {
        // Simulate hidden states: [embedding, layer0, layer1, layer2, ...]
        let h0 = MLXArray.zeros([1, 1, 512])
        let h1 = MLXArray.ones([1, 1, 512])
        let h2 = MLXArray.full([1, 1, 512], values: MLXArray(2.0))
        let h3 = MLXArray.full([1, 1, 512], values: MLXArray(3.0))
        let hiddenStates = [h0, h1, h2, h3]

        let result = extractContextFeature(hiddenStates: hiddenStates, layerIDs: [1, 2])
        XCTAssertEqual(result.dim(-1), 1024) // 512 * 2 concatenated
    }

    func testExtractContextFeatureFromDict() {
        var captured: [Int: MLXArray] = [:]
        captured[1] = MLXArray.ones([1, 1, 256])
        captured[2] = MLXArray.full([1, 1, 256], values: MLXArray(2.0))
        captured[3] = MLXArray.full([1, 1, 256], values: MLXArray(3.0))

        let result = extractContextFeatureFromDict(
            capturedDict: captured,
            targetLayerIDs: [1, 2]
        )
        XCTAssertEqual(result.dim(-1), 512) // 256 * 2
    }

    // MARK: - Events

    func testDFlashSummary() {
        let timings = DFlashSummary.PhaseTimings(
            prefill: 100.0,
            draft: 50.0,
            verify: 200.0,
            replay: 10.0
        )
        let summary = DFlashSummary(
            elapsedUs: 500.0,
            promptTokenCount: 100,
            generatedTokenIDs: [1, 2, 3, 4, 5],
            acceptedFromDraft: 3,
            acceptanceRatio: 0.6,
            blockTokens: 4,
            cyclesCompleted: 2,
            phaseTimingsUs: timings
        )

        XCTAssertEqual(summary.generationTokens, 5)
        XCTAssertEqual(summary.promptTokenCount, 100)
        XCTAssertEqual(summary.acceptedFromDraft, 3)
    }
}

// MARK: - Integration Tests

#if canImport(MLXLLM)
import MLXLLM

final class DFlashIntegrationTests: XCTestCase {
    func testFullAttentionEngineCreation() {
        let engine = FullAttentionEngine()
        // Verify engine can be created and used
        let cache: [KVCache] = []
        engine.armRollback(targetCache: cache, prefixLen: 0)
        let replayNs = engine.rollback(
            targetCache: cache,
            targetLen: 10,
            acceptanceLength: 3,
            draftedTokens: 5
        )
        XCTAssertGreaterThanOrEqual(replayNs, 0)
    }

    func testHybridGDNEngineCreation() {
        let engine = HybridGDNEngine()
        let cache: [KVCache] = []
        engine.armRollback(targetCache: cache, prefixLen: 0)
        let replayNs = engine.rollback(
            targetCache: cache,
            targetLen: 10,
            acceptanceLength: 3,
            draftedTokens: 5
        )
        XCTAssertGreaterThanOrEqual(replayNs, 0)
    }
}
#endif