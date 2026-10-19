// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SQLite.swift",
    platforms: [
        .iOS(.v11),
        .macOS(.v10_13),
        .watchOS(.v4),
        .tvOS(.v11),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "SQLite",
            targets: ["SQLite"]
        )
    ],
    targets: [
        .target(
            name: "SQLite",
            exclude: [
                "Info.plist"
            ]
        ),
        .testTarget(
            name: "SQLiteTests",
            dependencies: [
                "SQLite"
            ],
            path: "Tests/SQLiteTests",
            exclude: [
                "Info.plist"
            ],
            resources: [
                .copy("Resources")
            ]
        )
    ]
)

#if os(Linux)
package.dependencies = [
    .package(path: "../CSQLite")
]
package.targets.first?.dependencies += [
    .product(name: "CSQLite", package: "CSQLite")
]
#endif
