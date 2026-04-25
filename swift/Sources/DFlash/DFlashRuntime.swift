// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Based on DFlash (arXiv:2602.06036)
// vllm-swift DFlash speculative decoding runtime

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DFlash Runtime

/// The main DFlash speculative decoding runtime.
///
/// Orchestrates the block-diffusion draft → verify → accept/reject → rollback
/// cycle for lossless speculative decoding on Apple Silicon.

/// Wrapper to transfer non-Sendable values across Task boundaries.
/// Safety: caller ensures no concurrent access.
struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

public enum DFlashRuntime {

    // MARK: - Token Utilities

    /// Build a suppress token mask from a list of token IDs.
    public static func buildSuppressTokenMask(
        vocabSize: Int,
        suppressTokenIDs: [Int]?
    ) -> MLXArray? {
        let ids = Set((suppressTokenIDs ?? []).filter { $0 >= 0 && $0 < vocabSize })
        guard !ids.isEmpty else { return nil }
        var mask = [Bool](repeating: false, count: vocabSize)
        for id in ids { mask[id] = true }
        return MLXArray(mask)
    }

    /// Greedy token selection with optional suppress mask.
    public static func greedyTokensWithMask(
        logits: MLXArray,
        suppressTokenMask: MLXArray? = nil
    ) -> MLXArray {
        if let mask = suppressTokenMask {
            let floor = MLXArray(-1e9, dtype: logits.dtype)
            let maskedLogits = MLX.where(mask, floor, logits)
            return argMax(maskedLogits, axis: -1).asType(.uint32)
        }
        return argMax(logits, axis: -1).asType(.uint32)
    }

    /// Match the acceptance length between drafted and posterior tokens.
    /// Returns the number of consecutive matches starting from position 0.
    /// E.g. if drafted=[1,2,3] and posterior=[1,2,5], returns 2.
    public static func matchAcceptanceLength(
        draftedTokens: MLXArray,
        posteriorTokens: MLXArray
    ) -> MLXArray {
        let count = draftedTokens.dim(0)
        guard count > 0 else { return MLXArray(0, dtype: .int32) }
        let matches = (draftedTokens .== posteriorTokens).asType(.int32)
        // cumprod: [1,1,0,...] for consecutive matches, then sum counts them
        return cumprod(matches, axis: 0).sum(axis: 0, keepDims: false)
    }

    // MARK: - Target Cache Management

    /// Create the appropriate cache entries for the target model.
    /// For hybrid GDN models, replaces MambaCache with a rollback-capable variant.
    public static func makeTargetCache(
        targetModel: any DFlashTargetModel,
        useTapeRollback: Bool = true
    ) -> [KVCache] {
        let cache = targetModel.newCache(parameters: nil)
        if targetModel.dflashIsHybridGDN {
            // Note: MambaSnapshotCache/RecurrentRollbackCache would be used here
            // if we have the full GDN implementation available
        }
        return cache
    }

    // MARK: - Main Generation Loop

    /// Generate tokens using DFlash speculative decoding.
    ///
    /// - Parameters:
    ///   - targetModel: The target (large) language model (must conform to DFlashTargetModel)
    ///   - draftModel: The DFlash block-diffusion draft model
    ///   - promptTokens: Pre-tokenized prompt token IDs
    ///   - maxNewTokens: Maximum number of new tokens to generate
    ///   - config: DFlash configuration options
    /// - Returns: AsyncStream of DFlashEvent values
    public static func generate(
        targetModel: any DFlashTargetModel,
        draftModel: any DFlashDraftModelProtocol,
        promptTokens: [Int],
        maxNewTokens: Int,
        config: DFlashConfiguration = DFlashConfiguration()
    ) -> AsyncStream<DFlashEvent> {
        // Use UnsafeSendable to pass non-Sendable values across Task boundaries
        // Safety: caller ensures no concurrent access to the models
        let targetWrapper = UnsafeSendable(targetModel)
        let draftWrapper = UnsafeSendable(draftModel)

        return AsyncStream(bufferingPolicy: .unbounded) { continuation in
            let task = Task {
                let target = targetWrapper.value
                let draft = draftWrapper.value
                generateStreaming(
                    targetModel: target,
                    draftModel: draft,
                    promptTokens: promptTokens,
                    maxNewTokens: maxNewTokens,
                    config: config,
                    yield: { event in
                        guard !Task.isCancelled else { return }
                        continuation.yield(event)
                    }
                )
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Synchronous generation that returns all events at once.
    public static func generateSync(
        targetModel: any DFlashTargetModel,
        draftModel: any DFlashDraftModelProtocol,
        promptTokens: [Int],
        maxNewTokens: Int,
        config: DFlashConfiguration = DFlashConfiguration()
    ) -> [DFlashEvent] {
        var events: [DFlashEvent] = []
        generateStreaming(
            targetModel: targetModel,
            draftModel: draftModel,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            config: config,
            yield: { events.append($0) }
        )
        return events
    }

    /// Core streaming generation loop.
    private static func generateStreaming(
        targetModel: any DFlashTargetModel,
        draftModel: any DFlashDraftModelProtocol,
        promptTokens: [Int],
        maxNewTokens: Int,
        config: DFlashConfiguration,
        yield: (DFlashEvent) -> Void
    ) {
        let promptLen = promptTokens.count
        guard promptLen > 0 && maxNewTokens > 0 else { return }

        let tokensInt32 = promptTokens.map { Int32($0) }
        let promptArray = MLXArray(tokensInt32).reshaped(1, -1).asType(.uint32)

        // Detect engine and create caches
        let engine: any DFlashEngineProtocol = targetModel.dflashIsHybridGDN
            ? HybridGDNEngine()
            : FullAttentionEngine()

        let draftBackend = DFlashDraftBackend()

        let targetCache = makeTargetCache(
            targetModel: targetModel,
            useTapeRollback: config.useTapeRollback
        )

        let draftCache = draftBackend.makeCache(
            draftModel: draftModel,
            sinkSize: config.draftSinkSize,
            windowSize: config.draftWindowSize
        )

        let targetLayerIDList = draftModel.targetLayerIDs
        let captureLayerIDs = Set(targetLayerIDList.map { $0 + 1 })
        let maskTokenID = draftModel.maskTokenID

        let startNanos = DispatchTime.now().uptimeNanoseconds

        // ── Prefill ────────────────────────────────────────────────
        let prefillStepSize = 2048
        var targetHidden: MLXArray?
        var prefillLogits: MLXArray!

        for chunkStart in stride(from: 0, to: promptLen, by: prefillStepSize) {
            let chunkEnd = min(chunkStart + prefillStepSize, promptLen)
            let chunkIDs = promptArray[0..., chunkStart ..< chunkEnd]

            let (chunkLogits, chunkHidden) = targetModel.dflashForwardWithCapture(
                inputIDs: chunkIDs,
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )

            // Batched asyncEval: enqueue everything without blocking
            asyncEval(chunkLogits)
            for (_, v) in chunkHidden { asyncEval(v) }

            let feat = extractContextFeatureFromDict(
                capturedDict: chunkHidden,
                targetLayerIDs: targetLayerIDList
            )

            if targetHidden == nil {
                targetHidden = MLXArray.zeros(
                    [feat.dim(0), promptLen, feat.dim(-1)],
                    dtype: feat.dtype
                )
            }
            targetHidden![0..., chunkStart ..< chunkEnd, 0...] = feat
            eval(targetHidden!)

            prefillLogits = chunkLogits

            yield(.prefillProgress(
                tokensProcessed: chunkEnd,
                tokensTotal: promptLen
            ))
        }

        MLX.Memory.clearCache()

        let prefillNanos = Int(DispatchTime.now().uptimeNanoseconds) - Int(startNanos)

        let suppressTokenMask = buildSuppressTokenMask(
            vocabSize: Int(prefillLogits.dim(-1)),
            suppressTokenIDs: config.suppressTokenIDs
        )

        var stagedFirst = greedyTokensWithMask(
            logits: prefillLogits[0..., -1, 0...],
            suppressTokenMask: suppressTokenMask
        ).reshaped(-1)

        yield(.prefill(
            promptTokenCount: promptLen,
            prefillUs: Double(prefillNanos) / 1000.0
        ))

        // Yield the first token
        let firstTokenID = Int(stagedFirst.item(Int.self))
        yield(.token(
            tokenID: firstTokenID,
            generatedTokens: 1,
            acceptanceRatio: 0.0,
            cyclesCompleted: 0
        ))

        // ── Generation Loop ───────────────────────────────────────
        let draftBlockSize = draftModel.blockSize
        let requestedBlockTokens = config.blockTokens ?? draftBlockSize
        let effectiveBlockTokens = max(1, min(requestedBlockTokens, draftBlockSize))
        let verifyLenCap = effectiveBlockTokens

        var generatedTokenIDs: [Int] = []
        var acceptedFromDraft = 0
        var cyclesCompleted = 0
        var start = promptLen
        var firstTokenYielded = false

        generatedTokenIDs.append(firstTokenID)
        firstTokenYielded = true

        let maskTokenTail = MLXArray.full(
            [max(0, effectiveBlockTokens - 1)],
            values: MLXArray(Int32(maskTokenID), dtype: .uint32)
        )

        var verifyNsTotal: Int = 0
        var draftNsTotal: Int = 0
        var replayNsTotal: Int = 0

        // Precompute stop token set for O(1) lookup
        let stopTokenSet = Set(config.stopTokenIDs)

        // Prefetch state: the draft for the NEXT cycle can be overlapped
        // with the current cycle's rollback.
        var prefetchedDraft: MLXArray?
        var prefetchedBlockLen: Int?

        while generatedTokenIDs.count < maxNewTokens {
            let remaining = maxNewTokens - generatedTokenIDs.count
            let blockLen = max(1, min(effectiveBlockTokens, remaining))

            // ── Draft Phase ──────────────────────────────────────
            var drafted: MLXArray?
            let currentStagedFirst = stagedFirst
            if blockLen > 1 {
                if let pf = prefetchedDraft, prefetchedBlockLen == blockLen {
                    drafted = pf
                    prefetchedDraft = nil
                    prefetchedBlockLen = nil
                } else {
                    let draftStart = Int(DispatchTime.now().uptimeNanoseconds)
                    drafted = draftBackend.draftGreedy(
                        targetModel: targetModel,
                        draftModel: draftModel,
                        draftCache: draftCache,
                        stagedFirst: stagedFirst,
                        targetHidden: targetHidden!,
                        blockLen: blockLen,
                        maskTokenTail: maskTokenTail,
                        suppressTokenMask: suppressTokenMask
                    )
                    draftNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - draftStart
                }
            }

            // ── Verify Phase ────────────────────────────────────
            let verifyTokenCount = min(blockLen, verifyLenCap)
            let verifyTokenIDs: MLXArray
            if blockLen <= 1 {
                verifyTokenIDs = currentStagedFirst[..<1]
            } else if let drafted = drafted, verifyTokenCount > 1 {
                verifyTokenIDs = concatenated(
                    [currentStagedFirst[..<1], drafted[..<(verifyTokenCount - 1)]],
                    axis: 0
                )
            } else {
                verifyTokenIDs = currentStagedFirst[..<1]
            }
            let verifyIDs = verifyTokenIDs[.newAxis]

            armTargetRollback(targetCache: targetCache, prefixLen: start)

            let verifyStart = Int(DispatchTime.now().uptimeNanoseconds)
            let (verifyLogits, verifyHiddenStates) = targetModel.dflashForwardWithCapture(
                inputIDs: verifyIDs,
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )
            asyncEval(verifyLogits)
            for v in verifyHiddenStates.values { asyncEval(v) }
            verifyNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - verifyStart

            // ── Accept/Reject ──────────────────────────────────
            let posterior = greedyTokensWithMask(
                logits: verifyLogits[0],
                suppressTokenMask: suppressTokenMask
            )

            let acceptanceLen: Int
            if verifyTokenIDs.dim(0) > 1 {
                acceptanceLen = Int(
                    matchAcceptanceLength(
                        draftedTokens: verifyTokenIDs[1...],
                        posteriorTokens: posterior[..<(verifyTokenIDs.dim(0) - 1)]
                    ).item(Int.self)
                )
            } else {
                acceptanceLen = 0
            }

            let committedHidden = extractContextFeatureFromDict(
                capturedDict: verifyHiddenStates,
                targetLayerIDs: targetLayerIDList
            )[0..., ..<(1 + acceptanceLen), 0...]
            asyncEval(committedHidden)

            let commitCount = 1 + acceptanceLen
            let committedSegment = verifyTokenIDs[..<(commitCount)]

            let stagedFirstNext = posterior[acceptanceLen ..< (acceptanceLen + 1)]

            // ── Prefetch next draft (overlaps with rollback on GPU) ──
            let nextRemaining = maxNewTokens - generatedTokenIDs.count - commitCount
            let nextBlockLen = max(1, min(effectiveBlockTokens, nextRemaining))
            if nextBlockLen > 1 && generatedTokenIDs.count + commitCount < maxNewTokens {
                prefetchedDraft = draftBackend.draftGreedy(
                    targetModel: targetModel,
                    draftModel: draftModel,
                    draftCache: draftCache,
                    stagedFirst: stagedFirstNext,
                    targetHidden: committedHidden,
                    blockLen: nextBlockLen,
                    maskTokenTail: maskTokenTail,
                    suppressTokenMask: suppressTokenMask
                )
                prefetchedBlockLen = nextBlockLen
                asyncEval(prefetchedDraft!)
            } else {
                prefetchedDraft = nil
                prefetchedBlockLen = nil
            }

            // ── Rollback ───────────────────────────────────────
            start += commitCount
            targetHidden = committedHidden
            let replayNs = engine.rollback(
                targetCache: targetCache,
                targetLen: start,
                acceptanceLength: acceptanceLen,
                draftedTokens: blockLen - 1
            )
            replayNsTotal += replayNs
            cyclesCompleted += 1
            acceptedFromDraft += acceptanceLen

            // ── Emit tokens ───────────────────────────────────
            let committedIDs = committedSegment.asArray(Int.self)
            for tokenID in committedIDs {
                guard generatedTokenIDs.count < maxNewTokens else { break }

                if firstTokenYielded {
                    firstTokenYielded = false
                    continue
                }

                generatedTokenIDs.append(tokenID)

                let acceptanceRatio = generatedTokenIDs.count > 0
                    ? Double(acceptedFromDraft) / Double(generatedTokenIDs.count)
                    : 0.0
                yield(.token(
                    tokenID: tokenID,
                    generatedTokens: generatedTokenIDs.count,
                    acceptanceRatio: acceptanceRatio,
                    cyclesCompleted: cyclesCompleted
                ))
            }

            // Check for stop tokens
            let hit = committedIDs.contains { stopTokenSet.contains($0) }
            if hit { break }

            stagedFirst = stagedFirstNext
        }

        // ── Summary ────────────────────────────────────────────
        let elapsedNanos = Int(DispatchTime.now().uptimeNanoseconds) - Int(startNanos)
        let acceptanceRatio = generatedTokenIDs.count > 0
            ? Double(acceptedFromDraft) / Double(generatedTokenIDs.count)
            : 0.0

        yield(.summary(DFlashSummary(
            elapsedUs: Double(elapsedNanos) / 1000.0,
            promptTokenCount: promptLen,
            generatedTokenIDs: generatedTokenIDs,
            acceptedFromDraft: acceptedFromDraft,
            acceptanceRatio: acceptanceRatio,
            blockTokens: effectiveBlockTokens,
            cyclesCompleted: cyclesCompleted,
            phaseTimingsUs: .init(
                prefill: Double(prefillNanos) / 1000.0,
                draft: Double(draftNsTotal) / 1000.0,
                verify: Double(verifyNsTotal) / 1000.0,
                replay: Double(replayNsTotal) / 1000.0
            )
        )))
    }
}

// MARK: - Hybrid GDN Engine (Stub for future implementation)

/// Engine for hybrid GatedDeltaNet + attention target models.
/// Uses rollback caches for recurrent layers with tape replay.
public final class HybridGDNEngine: DFlashEngineProtocol, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {
        for cache in targetCache {
            if let rollbackCache = cache as? (any DFlashRollbackCacheProtocol) {
                rollbackCache.armRollback(prefixLen: prefixLen)
            }
        }
    }

    public func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        restoreTargetCacheAfterAcceptance(
            targetCache,
            targetLen: targetLen,
            acceptanceLength: acceptanceLength,
            draftedTokens: draftedTokens
        )
    }
}