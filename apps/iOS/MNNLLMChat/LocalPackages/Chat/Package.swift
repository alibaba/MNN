// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Chat",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "ExyteChat",
            targets: ["ExyteChat"]),
    ],
    dependencies: [
        .package(path: "../swiftui-introspect"),
        .package(path: "../swift-markdown-ui"),
        .package(path: "../MarkdownView"),
        .package(path: "../MediaPicker"),
        .package(path: "../FloatingButton"),
        .package(path: "../ActivityIndicatorView"),
    ],
    targets: [
        .target(
            name: "ExyteChat",
            dependencies: [
                .product(name: "SwiftUIIntrospect", package: "swiftui-introspect"),
                .product(name: "MarkdownView", package: "MarkdownView"),
                .product(name: "MarkdownParser", package: "MarkdownView"),
                .product(name: "ExyteMediaPicker", package: "MediaPicker"),
                .product(name: "FloatingButton", package: "FloatingButton"),
                .product(name: "ActivityIndicatorView", package: "ActivityIndicatorView")
            ]
        ),
        .testTarget(
            name: "ExyteChatTests",
            dependencies: ["ExyteChat"]),
    ]
)
