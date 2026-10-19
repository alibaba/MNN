// swift-tools-version: 5.9
import PackageDescription

// MNNChatCore — shared chat/inference code for the MNN iOS demo apps
// (MNNLLMChat, MNNPocket, ...). App shells provide only the entry point,
// signing, resources (Assets.xcassets, Localizable.xcstrings) and bundled
// models under LocalModel/.
//
// Dual-engine: the internal (AliNNPrivate) and public MNN builds use
// incompatible OpType numberings, so each app links its OWN engine
// xcframework (Vendor/public or Vendor/internal) via its Xcode project.
// CMNNInference below compiles header-only against staged MNN headers
// (mnn-include/, from the public engine = common API subset); the MNN and
// sherpa-mnn symbols are resolved when the app links its engine. Regenerate
// staged headers and xcframeworks with ../Scripts/build-xcframeworks.sh
// (all gitignored). Device-only (arm64); simulator builds unsupported.

let package = Package(
    name: "MNNChatCore",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "MNNChatCore", targets: ["MNNChatCore"]),
    ],
    dependencies: [
        .package(path: "../LocalPackages/Chat"),
        .package(path: "../LocalPackages/MarkdownView"),
        .package(path: "../LocalPackages/MediaPicker"),
        .package(path: "../LocalPackages/SQLite.swift"),
        .package(path: "../LocalPackages/swift-transformers"),
        .package(path: "../LocalPackages/Jinja"),
        .package(path: "../LocalPackages/NetworkImage"),
        .package(path: "../LocalPackages/LRUCache"),
    ],
    targets: [
        .target(
            name: "CMNNInference",
            path: "Sources/CMNNInference",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("mnn-include"),
                .headerSearchPath("sherpa-include"),
            ]
        ),
        .target(
            name: "MNNChatCore",
            dependencies: [
                "CMNNInference",
                .product(name: "ExyteChat", package: "Chat"),
                .product(name: "ExyteMediaPicker", package: "MediaPicker"),
                .product(name: "MarkdownView", package: "MarkdownView"),
                .product(name: "SQLite", package: "SQLite.swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            resources: [.process("Resources")]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
