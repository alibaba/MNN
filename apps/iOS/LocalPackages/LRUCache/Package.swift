// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "LRUCache",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
        .tvOS(.v13),
        .watchOS(.v6),
    ],
    products: [
        .library(name: "LRUCache", targets: ["LRUCache"]),
    ],
    targets: [
        .target(name: "LRUCache", path: "Sources"),
        .testTarget(name: "LRUCacheTests", dependencies: ["LRUCache"], path: "Tests"),
    ]
)
