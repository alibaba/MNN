//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import PackagePlugin

@main
struct GenerateManualPlugin: CommandPlugin {
  func performCommand(
    context: PluginContext,
    arguments: [String]
  ) async throws {
    // Locate generation tool.
    let generationToolFile = try context.tool(named: "generate-manual").path

    // Create an extractor to extract plugin-only arguments from the `arguments`
    // array.
    var extractor = ArgumentExtractor(arguments)

    // Run generation tool once if help is requested.
    if extractor.helpRequest() {
      try generationToolFile.exec(arguments: ["--help"])
      print("""
        ADDITIONAL OPTIONS:
          --configuration <configuration>
                                  Tool build configuration used to generate the
                                  manual. (default: release)

        NOTE: The "GenerateManual" plugin handles passing the "<tool>" and
        "--output-directory <output-directory>" arguments. Manually supplying
        these arguments will result in a runtime failure.
        """)
      return
    }

    // Extract configuration argument before making it to the
    // "generate-manual" tool.
    let configuration = try extractor.configuration()

    // Build all products first.
    print("Building package in \(configuration) mode...")
    let buildResult = try packageManager.build(
      .all(includingTests: false),
      parameters: .init(configuration: configuration))

    guard buildResult.succeeded else {
      throw GenerateManualPluginError.buildFailed(buildResult.logText)
    }
    print("Built package in \(configuration) mode")

    // Run generate-manual on all executable artifacts.
    for builtArtifact in buildResult.builtArtifacts {
      // Skip non-executable targets
      guard builtArtifact.kind == .executable else { continue }

      // Skip executables without a matching product.
      guard let product = builtArtifact.matchingProduct(context: context)
      else { continue }

      // Skip products without a dependency on ArgumentParser.
      guard product.hasDependency(named: "ArgumentParser") else { continue }

      // Get the artifacts name.
      let executableName = builtArtifact.path.lastComponent
      print("Generating manual for \(executableName)...")

      // Create output directory.
      let outputDirectory = context
        .pluginWorkDirectory
        .appending(executableName)
      try outputDirectory.createOutputDirectory()

      // Create generation tool arguments.
      var generationToolArguments = [
        builtArtifact.path.string,
        "--output-directory",
        outputDirectory.string
      ]
      generationToolArguments.append(
        contentsOf: extractor.remainingArguments)

      // Spawn generation tool.
      try generationToolFile.exec(arguments: generationToolArguments)
      print("Generated manual in '\(outputDirectory)'")
    }
  }
}
