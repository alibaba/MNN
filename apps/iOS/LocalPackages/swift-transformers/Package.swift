// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
        .executable(name: "transformers", targets: ["TransformersCLI"]),
        .executable(name: "hub-cli", targets: ["HubCLI"]),
    ],
    dependencies: [
        .package(path: "../swift-argument-parser"),
        .package(path: "../Jinja"),
    ],
    targets: [
        .executableTarget(
            name: "TransformersCLI",
            dependencies: [
                "Models", "Generation", "Tokenizers",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(name: "HubCLI", dependencies: ["Hub", .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .target(name: "Hub", resources: [.process("FallbackConfigs")]),
        .target(name: "Tokenizers", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")]),
        .target(name: "TensorUtils"),
        .target(name: "Generation", dependencies: ["Tokenizers", "TensorUtils"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation", "TensorUtils"]),
        .testTarget(name: "TokenizersTests", dependencies: ["Tokenizers", "Models", "Hub"], resources: [.process("Resources"), .process("Vocabs")]),
        .testTarget(name: "HubTests", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")]),
        .testTarget(name: "PreTokenizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(name: "TensorUtilsTests", dependencies: ["TensorUtils", "Models", "Hub"], resources: [.process("Resources")]),
        .testTarget(name: "NormalizerTests", dependencies: ["Tokenizers", "Hub"]),
        .testTarget(name: "PostProcessorTests", dependencies: ["Tokenizers", "Hub"]),
    ]
)
