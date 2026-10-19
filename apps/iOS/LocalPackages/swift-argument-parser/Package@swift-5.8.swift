// swift-tools-version:5.8
//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import PackageDescription

var package = Package(
    name: "swift-argument-parser",
    products: [
        .library(
            name: "ArgumentParser",
            targets: ["ArgumentParser"]),
        .plugin(
            name: "GenerateManual",
            targets: ["GenerateManual"]),
    ],
    dependencies: [],
    targets: [
        // Core Library
        .target(
            name: "ArgumentParser",
            dependencies: ["ArgumentParserToolInfo"],
            exclude: ["CMakeLists.txt"],
            swiftSettings: [
              .enableExperimentalFeature("StrictConcurrency"),
            ]),
        .target(
            name: "ArgumentParserTestHelpers",
            dependencies: ["ArgumentParser", "ArgumentParserToolInfo"],
            exclude: ["CMakeLists.txt"]),
        .target(
            name: "ArgumentParserToolInfo",
            dependencies: [ ],
            exclude: ["CMakeLists.txt"]),

        // Plugins
        .plugin(
            name: "GenerateManual",
            capability: .command(
                intent: .custom(
                    verb: "generate-manual",
                    description: "Generate a manual entry for a specified target.")),
            dependencies: ["generate-manual"]),

        // Examples
        .executableTarget(
            name: "roll",
            dependencies: ["ArgumentParser"],
            path: "Examples/roll"),
        .executableTarget(
            name: "math",
            dependencies: ["ArgumentParser"],
            path: "Examples/math"),
        .executableTarget(
            name: "repeat",
            dependencies: ["ArgumentParser"],
            path: "Examples/repeat"),

        // Tools
        .executableTarget(
            name: "generate-manual",
            dependencies: ["ArgumentParser", "ArgumentParserToolInfo"],
            path: "Tools/generate-manual"),
    
        // Tests
        .testTarget(
            name: "ArgumentParserEndToEndTests",
            dependencies: ["ArgumentParser", "ArgumentParserTestHelpers"],
            exclude: ["CMakeLists.txt"]),
        .testTarget(
            name: "ArgumentParserExampleTests",
            dependencies: ["ArgumentParserTestHelpers"],
            resources: [.copy("CountLinesTest.txt")]),
        .testTarget(
            name: "ArgumentParserGenerateManualTests",
            dependencies: ["ArgumentParserTestHelpers"]),
        .testTarget(
            name: "ArgumentParserPackageManagerTests",
            dependencies: ["ArgumentParser", "ArgumentParserTestHelpers"],
            exclude: ["CMakeLists.txt"]),
        .testTarget(
            name: "ArgumentParserUnitTests",
            dependencies: ["ArgumentParser", "ArgumentParserTestHelpers"],
            exclude: ["CMakeLists.txt"]),
    ]
)

#if os(macOS)
package.targets.append(contentsOf: [
    // Examples
    .executableTarget(
        name: "count-lines",
        dependencies: ["ArgumentParser"],
        path: "Examples/count-lines"),

    // Tools
    .executableTarget(
        name: "changelog-authors",
        dependencies: ["ArgumentParser"],
        path: "Tools/changelog-authors"),
    ])
#endif
