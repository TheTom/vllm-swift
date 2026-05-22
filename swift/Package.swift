// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "VLLMSwiftMetal",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "VLLMBridge",
            type: .dynamic,
            targets: ["VLLMBridge"]
        ),
    ],
    dependencies: [
        // Frozen snapshot of mlx-swift-lm. The `vllm-swift-stable` branch on
        // TheTom/mlx-swift-lm is force-pushed at release time to capture the
        // exact alpha tip (plus any local cherry-picks) we tested. Alpha keeps
        // moving; vllm-swift-stable does not until the next release.
        // For local dev: .package(path: "/Users/tom/dev/mlx-swift-lm")
        // .package(url: "https://github.com/TheTom/mlx-swift-lm.git", branch: "vllm-swift-stable"),
        .package(path: "/Users/tom/dev/mlx-swift-lm"),
        // metaltile path is moving to `vllm-swift → FFAI → metaltile`. The
        // standalone `~/dev/MetalTileSwift` package dep is removed for the
        // moment; once vllm-swift takes a dep on FFAI, kernels flow through
        // FFAI's inner MetalTileSwift target. The shim wrapper that used
        // this dep (`MetalTileShim.swift`) is gone alongside the dep.
    ],
    targets: [
        .target(
            name: "VLLMBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
            ],
            path: "Sources/VLLMBridge",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"]),
            ]
        ),
        .testTarget(
            name: "VLLMBridgeTests",
            dependencies: [],
            path: "Tests/VLLMBridgeTests"
        ),
    ]
)
