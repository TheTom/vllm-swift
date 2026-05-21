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
        // metaltile kernels (Rust DSL → MSL → kernels.metallib + Swift wrappers).
        // Local path so we pick up new kernels as they're added to the registry.
        .package(path: "/Users/tom/dev/MetalTileSwift"),
    ],
    targets: [
        .target(
            name: "VLLMBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MetalTileSwift", package: "MetalTileSwift"),
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
