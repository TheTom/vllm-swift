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
        // Dev: local path for fast iteration (no network fetches)
        // Release: swap to .package(url: "https://github.com/ekryski/mlx-swift-lm.git", branch: "ek/tom-eric-moe-tuning")
        .package(path: "/Users/tom/dev/mlx-swift-lm"),
    ],
    targets: [
        .target(
            name: "VLLMBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/VLLMBridge",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"]),
            ]
        ),
    ]
)
