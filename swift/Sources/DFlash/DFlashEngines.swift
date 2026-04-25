// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Based on DFlash (arXiv:2602.06036)
// vllm-swift DFlash verify/rollback engines

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Full Attention Engine

/// Engine for pure-attention target models (no recurrent layers).
/// Rollback is just KV cache trimming.
public final class FullAttentionEngine: DFlashEngineProtocol, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {
        // Pure attention: no arming needed
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

// MARK: - Cache Restoration Utilities

/// Restore the target cache after partial acceptance of draft tokens.
///
/// For KVCacheSimple: trim to remove rejected tokens' KV entries.
/// For rollback-aware caches: delegates to their rollback method.
///
/// - Returns: Time spent on replay in nanoseconds
@discardableResult
public func restoreTargetCacheAfterAcceptance(
    _ cacheEntries: [KVCache],
    targetLen: Int,
    acceptanceLength: Int,
    draftedTokens: Int
) -> Int {
    let fullyAccepted = draftedTokens > 0 && acceptanceLength == draftedTokens
    var replayNs: Int = 0

    for cache in cacheEntries {
        if let rollbackCache = cache as? (any DFlashRollbackCacheProtocol) {
            if fullyAccepted {
                rollbackCache.clearTransients()
                continue
            }
            let startNs = Int(DispatchTime.now().uptimeNanoseconds)
            rollbackCache.rollback(nAccepted: acceptanceLength)
            replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
        } else if cache.isTrimmable {
            let offset = cache.offset
            if offset > targetLen {
                let startNs = Int(DispatchTime.now().uptimeNanoseconds)
                cache.trim(offset - targetLen)
                replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
            }
        }
    }

    return replayNs
}

/// Arm all rollback-capable caches in the target model.
public func armTargetRollback(targetCache: [KVCache], prefixLen: Int) {
    for cache in targetCache {
        if let rollbackCache = cache as? (any DFlashRollbackCacheProtocol) {
            rollbackCache.armRollback(prefixLen: prefixLen)
        }
    }
}

// MARK: - KVCache Extension for Trimmability

extension KVCache {
    /// Whether this cache type supports trimming.
    public var isTrimmable: Bool {
        self is KVCacheSimple
    }
}
