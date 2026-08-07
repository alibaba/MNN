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

import ArgumentParser

extension Package {
  /// Generates an Xcode project
  struct GenerateXcodeProject: ParsableCommand {
    static let configuration =
      CommandConfiguration(commandName: "generate-xcodeproj")

    @OptionGroup()
    var options: Options

    @Flag(help: "Enable code coverage in the generated project")
    var enableCodeCoverage: Bool = false

    @Flag(help: "Use the legacy scheme generator")
    var legacySchemeGenerator: Bool = false

    @Option(help: "Path where the Xcode project should be generated")
    var output: String?

    @Flag(help: "Do not add file references for extra files to the generated Xcode project")
    var skipExtraFiles: Bool = false

    @Flag(help: "Watch for changes to the Package manifest to regenerate the Xcode project")
    var watch: Bool = false

    @Option(help: "Path to xcconfig file")
    var xcconfigOverrides: String?

    mutating func run() {
      print("Generating Xcode Project.......")
    }
  }
}
