// swift-tools-version:5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "test",
    platforms: [
        .iOS(.v11),
        .macOS(.v10_13),
        .watchOS(.v4),
        .tvOS(.v11)
    ],
    dependencies: [
        // for testing from same repository
        .package(path: "../..")
        // normally this would be:
        // .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.15.3")
    ],
    targets: [
        .executableTarget(name: "test", dependencies: [.product(name: "SQLite", package: "SQLite.swift")])
    ]
)
