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

import XCTest
import ArgumentParserTestHelpers

final class RepeatExampleTests: XCTestCase {
  override func setUp() {
    #if !os(Windows) && !os(WASI)
    unsetenv("COLUMNS")
    #endif
  }

  func testRepeat() throws {
    try AssertExecuteCommand(command: "repeat hello", expected: """
        hello
        hello
        """)
  }
  
  func testRepeat_include_counter() throws {
    try AssertExecuteCommand(command: "repeat --include-counter hello", expected: """
      1: hello
      2: hello
      """)
  }
  
  func testRepeat_Count() throws {
    try AssertExecuteCommand(command: "repeat hello --count 6", expected: """
        hello
        hello
        hello
        hello
        hello
        hello
        """)
  }
  
  func testRepeat_Help() throws {
    let helpText = """
        USAGE: repeat [--count <count>] [--include-counter] <phrase>

        ARGUMENTS:
          <phrase>                The phrase to repeat.

        OPTIONS:
          --count <count>         The number of times to repeat 'phrase'.
          --include-counter       Include a counter with each repetition.
          -h, --help              Show help information.
        """
    
    try AssertExecuteCommand(command: "repeat -h", expected: helpText)
    try AssertExecuteCommand(command: "repeat --help", expected: helpText)
  }
  
  func testRepeat_Fail() throws {
    try AssertExecuteCommand(
      command: "repeat",
      expected: """
            Error: Missing expected argument '<phrase>'

            USAGE: repeat [--count <count>] [--include-counter] <phrase>

            ARGUMENTS:
              <phrase>                The phrase to repeat.

            OPTIONS:
              --count <count>         The number of times to repeat 'phrase'.
              --include-counter       Include a counter with each repetition.
              -h, --help              Show help information.
            """,
      exitCode: .validationFailure)

    try AssertExecuteCommand(
      command: "repeat hello --count",
      expected: """
            Error: Missing value for '--count <count>'
            Help:  --count <count>  The number of times to repeat 'phrase'.
            Usage: repeat [--count <count>] [--include-counter] <phrase>
              See 'repeat --help' for more information.
            """,
      exitCode: .validationFailure)
    
    try AssertExecuteCommand(
      command: "repeat hello --count ZZZ",
      expected: """
            Error: The value 'ZZZ' is invalid for '--count <count>'
            Help:  --count <count>  The number of times to repeat 'phrase'.
            Usage: repeat [--count <count>] [--include-counter] <phrase>
              See 'repeat --help' for more information.
            """,
      exitCode: .validationFailure)
    
    try AssertExecuteCommand(
      command: "repeat --version hello",
      expected: """
            Error: Unknown option '--version'
            Usage: repeat [--count <count>] [--include-counter] <phrase>
              See 'repeat --help' for more information.
            """,
      exitCode: .validationFailure)
  }
}
