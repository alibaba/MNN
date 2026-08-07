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

#if os(macOS)

import XCTest
import ArgumentParserTestHelpers

final class CountLinesExampleTests: XCTestCase {
  override func setUp() {
    #if !os(Windows) && !os(WASI)
    unsetenv("COLUMNS")
    #endif
  }
  
  func testCountLines() throws {
    guard #available(macOS 12, *) else { return }
    let testFile = try XCTUnwrap(Bundle.module.url(forResource: "CountLinesTest", withExtension: "txt"))
    try AssertExecuteCommand(command: "count-lines \(testFile.path)", expected: "20")
    try AssertExecuteCommand(command: "count-lines \(testFile.path) --prefix al", expected: "4")
  }
  
  func testCountLinesHelp() throws {
    guard #available(macOS 12, *) else { return }
    let helpText = """
        USAGE: count-lines [<input-file>] [--prefix <prefix>] [--verbose]

        ARGUMENTS:
          <input-file>            A file to count lines in. If omitted, counts the
                                  lines of stdin.

        OPTIONS:
          --prefix <prefix>       Only count lines with this prefix.
          --verbose               Include extra information in the output.
          -h, --help              Show help information.
        """
    try AssertExecuteCommand(command: "count-lines -h", expected: helpText)
  }
}

#endif
