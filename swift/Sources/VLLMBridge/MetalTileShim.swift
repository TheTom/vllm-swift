// MetalTileShim
//
// Replaces MLX's quantizedMatmul with a metaltile mt_qmm_mma dispatch.
// MLXArray ↔ MTLBuffer bridging uses MLXArray.asMTLBuffer (mlx-swift public
// API). Dispatch runs on MetalTileLibrary.shared.commandQueue, results
// reconstructed into MLXArray via mlx_array_new_data from MTLBuffer.contents().
//
// Gated by env var `VSM_USE_METALTILE`. When unset/0, callers MUST fall
// back to the MLX path — this shim is opt-in.

import Cmlx
import Foundation
import MLX
import Metal
import MetalTileSwift

public enum MetalTileShim {

    /// One-time env probe. Cached at first read; subsequent reads are atomic.
    public static let isEnabled: Bool = {
        (ProcessInfo.processInfo.environment["VSM_USE_METALTILE"] ?? "0") != "0"
    }()

    /// metaltile mt_qmm_mma dispatch — quantized int4 matmul Y = X @ W^T.
    ///
    /// Shapes:
    ///   x:        [M, K]                 fp16
    ///   w:        [N, K/8]               packed int4 (8 nibbles per uint32)
    ///   scales:   [N, K/groupSize]       fp16
    ///   biases:   [N, K/groupSize]       fp16
    ///   returns:  [M, N]                 fp16
    ///
    /// Constraints (mt_qmm_mma):
    ///   M  % 32 == 0  (BM=32, 4 SGs in 2x2 warp grid)
    ///   N  % 32 == 0  (BN=32)
    ///   K  % 32 == 0  (BK=32)
    ///   bits = 4, groupSize = 64, dtype = fp16
    ///
    /// Returns nil if any shape fails the constraint — caller must fall back.
    /// Synchronous: commits + waitUntilCompleted before returning.
    public static func qmmMma(
        x: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int = 64,
        bits: Int = 4
    ) -> MLXArray? {
        // Constraints
        guard bits == 4, groupSize == 64 else { return nil }
        guard x.dtype == .float16, scales.dtype == .float16, biases.dtype == .float16
        else { return nil }
        guard x.ndim == 2 else { return nil }
        let m = x.shape[0]
        let k = x.shape[1]
        guard w.ndim == 2, w.shape[1] == k / 8 else { return nil }
        let n = w.shape[0]
        guard m % 32 == 0, n % 32 == 0, k % 32 == 0 else { return nil }
        let gsPerRow = k / groupSize

        let lib = MetalTileLibrary.shared
        let device = lib.device

        // MLXArray → MTLBuffer (eval implicit). noCopy=false to avoid lifetime
        // issues with the temporary MLXArrays during a single dispatch.
        guard let bX = x.asMTLBuffer(device: device),
              let bW = w.asMTLBuffer(device: device),
              let bScales = scales.asMTLBuffer(device: device),
              let bBiases = biases.asMTLBuffer(device: device)
        else { return nil }

        // Output: [M, N] fp16 = M*N*2 bytes
        let outBytes = m * n * 2
        guard let bOut = device.makeBuffer(length: outBytes, options: .storageModeShared)
        else { return nil }

        // mt_qmm_mma grid: [N/32, M/32, 1], tpg=[128,1,1] (4 SGs)
        let grid = MTLSize(width: n / 32, height: m / 32, depth: 1)
        let tpg = MTLSize(width: 128, height: 1, depth: 1)

        guard let cmd = lib.commandQueue.makeCommandBuffer() else { return nil }

        MetalTileKernels.mt_qmm_mma_f16(
            w: bW,
            scales: bScales,
            biases: bBiases,
            x: bX,
            out: bOut,
            k: UInt32(k),
            n: UInt32(n),
            gs_per_row: UInt32(gsPerRow),
            gridSize: grid,
            threadgroupSize: tpg,
            on: cmd
        )
        cmd.commit()
        cmd.waitUntilCompleted()

        // Rebuild MLXArray from output MTLBuffer.contents().
        // mlx_array_new_data copies bytes; freeing bOut after the call is safe.
        let ptr = bOut.contents().assumingMemoryBound(to: Int8.self)
        var arr = mlx_array_new()
        let shape: [Int32] = [Int32(m), Int32(n)]
        shape.withUnsafeBufferPointer { sp in
            arr = mlx_array_new_data(
                UnsafeRawPointer(ptr),
                sp.baseAddress!,
                Int32(2),
                MLX.DType.float16.cmlxDtype
            )
        }
        return MLXArray(arr)
    }
}
