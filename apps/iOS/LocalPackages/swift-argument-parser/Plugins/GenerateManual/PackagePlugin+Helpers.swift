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

import Foundation
import PackagePlugin

extension ArgumentExtractor {
  mutating func helpRequest() -> Bool {
      self.extractFlag(named: "help") > 0
  }

  mutating func configuration() throws -> PackageManager.BuildConfiguration {
    switch self.extractOption(named: "configuration").first {
    case .some(let configurationString):
      switch configurationString {
      case "debug":
        return .debug
      case "release":
        return .release
      default:
        throw GenerateManualPluginError
          .unknownBuildConfiguration(configurationString)
      }
    case .none:
      return .release
    }
  }
}

extension Path {
  func createOutputDirectory() throws {
    do {
      try FileManager.default.createDirectory(
        atPath: self.string,
        withIntermediateDirectories: true)
    } catch {
      throw GenerateManualPluginError.createOutputDirectoryFailed(error)
    }
  }

  func exec(arguments: [String]) throws {
    do {
      let process = Process()
      process.executableURL = URL(fileURLWithPath: self.string)
      process.arguments = arguments
      try process.run()
      process.waitUntilExit()
      guard
        process.terminationReason == .exit,
        process.terminationStatus == 0
      else {
        throw GenerateManualPluginError.subprocessFailedNonZeroExit(
          self, process.terminationStatus)
      }
    } catch {
      throw GenerateManualPluginError.subprocessFailedError(self, error)
    }
  }
}

extension PackageManager.BuildResult.BuiltArtifact {
  func matchingProduct(context: PluginContext) -> Product? {
    context
      .package
      .products
      .first { $0.name == self.path.lastComponent }
  }
}

extension Product {
  func hasDependency(named name: String) -> Bool {
    recursiveTargetDependencies
      .contains { $0.name == name }
  }

  var recursiveTargetDependencies: [Target] {
    var dependencies = [Target.ID: Target]()
    for target in self.targets {
      for dependency in target.recursiveTargetDependencies {
        dependencies[dependency.id] = dependency
      }
    }
    return Array(dependencies.values)
  }
}
