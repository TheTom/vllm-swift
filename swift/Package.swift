// swift-tools-version: 6.0
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
        // Pinned snapshot of alpha branch with BatchedKVCache + TurboQuant+
        // For local dev: swap to .package(path: "/Users/tom/dev/mlx-swift-lm")
        .package(url: "https://github.com/TheTom/mlx-swift-lm.git", branch: "vllm-swift-stable"),
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
    ]
)
