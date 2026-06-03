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
        // `vllm-swift → FFAI → metaltile`: FFAI carries the GGUF DSv4-Flash
        // engine (Swift) + the metaltile-generated Metal kernels (its inner
        // MetalTileSwift target). Bridging to it is native Swift→Swift — the
        // only C-ABI edge stays at the Python↔dylib boundary (`@_cdecl`).
        .package(path: "/Users/tom/dev/FFAI"),
    ],
    targets: [
        .target(
            name: "VLLMBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "FFAI", package: "FFAI"),
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
