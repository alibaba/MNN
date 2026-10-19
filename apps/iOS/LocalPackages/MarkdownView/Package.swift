// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MarkdownView",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v13),
        .macCatalyst(.v13),
    ],
    products: [
        .library(name: "MarkdownView", targets: ["MarkdownView"]),
        .library(name: "MarkdownParser", targets: ["MarkdownParser"]),
    ],
    dependencies: [
        .package(path: "../swift-collections"),
        .package(path: "../SwiftMath"),
        .package(path: "../Splash"),
        .package(path: "../swift-cmark"),
        .package(path: "../LRUCache"),
    ],
    targets: [
        .target(name: "MarkdownView", dependencies: [
            "Litext",
            "Splash",
            "MarkdownParser",
            "SwiftMath",
            "LRUCache",
            .product(name: "DequeModule", package: "swift-collections"),
            .product(name: "OrderedCollections", package: "swift-collections"),
        ]),
        .target(name: "MarkdownParser", dependencies: [
            .product(name: "cmark-gfm", package: "swift-cmark"),
            .product(name: "cmark-gfm-extensions", package: "swift-cmark"),
        ]),
        .target(name: "Litext"),
    ]
)
