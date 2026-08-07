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

enum SubprocessError: Swift.Error, LocalizedError, CustomStringConvertible {
  case missingExecutable(url: URL)
  case failedToLaunch(error: Swift.Error)
  case nonZeroExitCode(code: Int)

  var description: String {
    switch self {
    case .missingExecutable(let url):
      return "No executable at '\(url.standardizedFileURL.path)'."
    case .failedToLaunch(let error):
      return "Couldn't run command process. \(error.localizedDescription)"
    case .nonZeroExitCode(let code):
      return "Process returned non-zero exit code '\(code)'."
    }
  }

  var errorDescription: String? { description }
}

func executeCommand(
  executable: URL,
  arguments: [String]
) throws -> String {
  guard (try? executable.checkResourceIsReachable()) ?? false else {
    throw SubprocessError.missingExecutable(url: executable)
  }

  let process = Process()
  if #available(macOS 10.13, *) {
    process.executableURL = executable
  } else {
    process.launchPath = executable.path
  }
  process.arguments = arguments

  let output = Pipe()
  process.standardOutput = output
  process.standardError = FileHandle.nullDevice

  if #available(macOS 10.13, *) {
    do {
      try process.run()
    } catch {
      throw SubprocessError.failedToLaunch(error: error)
    }
  } else {
    process.launch()
  }
  let outputData = output.fileHandleForReading.readDataToEndOfFile()
  process.waitUntilExit()

  guard process.terminationStatus == 0 else {
    throw SubprocessError.nonZeroExitCode(code: Int(process.terminationStatus))
  }

  let outputActual = String(data: outputData, encoding: .utf8)?
    .trimmingCharacters(in: .whitespacesAndNewlines)
    ?? ""

  return outputActual
}
