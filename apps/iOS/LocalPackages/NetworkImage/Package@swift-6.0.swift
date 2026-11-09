// swift-tools-version:6.0

import PackageDescription

let package = Package(
  name: "NetworkImage",
  platforms: [
    .macOS(.v11),
    .iOS(.v14),
    .tvOS(.v14),
    .watchOS(.v7),
  ],
  products: [
    .library(name: "NetworkImage", targets: ["NetworkImage"])
  ],
  dependencies: [],
  targets: [
    .target(name: "NetworkImage"),
    .testTarget(
      name: "NetworkImageTests",
      dependencies: ["NetworkImage"]
    ),
  ]
)
