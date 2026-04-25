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
        .library(
            name: "DFlash",
            type: .static,
            targets: ["DFlash"]
        ),
    ],
    dependencies: [
        // Pinned snapshot of alpha branch with BatchedKVCache + TurboQuant+
        .package(url: "https://github.com/TheTom/mlx-swift-lm.git", branch: "vllm-swift-stable"),
        // MLX from the mlx-swift-lm dependency chain
        .package(url: "https://github.com/ekryski/mlx-swift.git", branch: "alpha"),
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
        .target(
            name: "DFlash",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/DFlash"
        ),
        .testTarget(
            name: "DFlashTests",
            dependencies: ["DFlash"]
        ),
    ]
)
