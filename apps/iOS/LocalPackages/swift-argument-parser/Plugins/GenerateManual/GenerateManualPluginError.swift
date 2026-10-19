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

enum GenerateManualPluginError: Error {
  case unknownBuildConfiguration(String)
  case buildFailed(String)
  case createOutputDirectoryFailed(Error)
  case subprocessFailedNonZeroExit(Path, Int32)
  case subprocessFailedError(Path, Error)
}

extension GenerateManualPluginError: CustomStringConvertible {
  var description: String {
    switch self {
    case .unknownBuildConfiguration(let configuration):
      return "Build failed: Unknown build configuration '\(configuration)'."
    case .buildFailed(let logText):
      return "Build failed: \(logText)."
    case .createOutputDirectoryFailed(let error):
      return """
        Failed to create output directory: '\(error.localizedDescription)'
        """
    case .subprocessFailedNonZeroExit(let tool, let exitCode):
      return """
        '\(tool.lastComponent)' invocation failed with a nonzero exit code: \
        '\(exitCode)'.
        """
    case .subprocessFailedError(let tool, let error):
      return """
        '\(tool.lastComponent)' invocation failed: \
        '\(error.localizedDescription)'
        """
    }
  }
}

extension GenerateManualPluginError: LocalizedError {
  var errorDescription: String? { self.description }
}
