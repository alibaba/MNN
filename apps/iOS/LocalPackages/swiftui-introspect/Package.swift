// swift-tools-version:5.7

import PackageDescription

let package = Package(
    name: "swiftui-introspect",
    platforms: [
        .iOS(.v13),
        .tvOS(.v13),
        .macOS(.v10_15),
    ],
    products: [
        .library(name: "SwiftUIIntrospect", targets: ["SwiftUIIntrospect"]),
        .library(name: "SwiftUIIntrospect-Static", type: .static, targets: ["SwiftUIIntrospect"]),
        .library(name: "SwiftUIIntrospect-Dynamic", type: .dynamic, targets: ["SwiftUIIntrospect"]),
    ],
    targets: [
        .target(
            name: "SwiftUIIntrospect",
            path: "Sources"
        ),
    ]
)
