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
import ArgumentParser

final class LongNameWithSingleDashEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Flag(name: .customLong("file", withSingleDash: true))
  var file: Bool = false

  @Flag(name: .short)
  var force: Bool = false

  @Flag(name: .short)
  var input: Bool = false
}

extension LongNameWithSingleDashEndToEndTests {
  func testParsing_empty() throws {
    AssertParse(Bar.self, []) { options in
      XCTAssertEqual(options.file, false)
      XCTAssertEqual(options.force, false)
      XCTAssertEqual(options.input, false)
    }
  }

  func testParsing_singleOption_1() {
    AssertParse(Bar.self, ["-file"]) { options in
      XCTAssertEqual(options.file, true)
      XCTAssertEqual(options.force, false)
      XCTAssertEqual(options.input, false)
    }
  }

  func testParsing_singleOption_2() {
    AssertParse(Bar.self, ["-f"]) { options in
      XCTAssertEqual(options.file, false)
      XCTAssertEqual(options.force, true)
      XCTAssertEqual(options.input, false)
    }
  }

  func testParsing_singleOption_3() {
    AssertParse(Bar.self, ["-i"]) { options in
      XCTAssertEqual(options.file, false)
      XCTAssertEqual(options.force, false)
      XCTAssertEqual(options.input, true)
    }
  }

  func testParsing_combined_1() {
    AssertParse(Bar.self, ["-f", "-i"]) { options in
      XCTAssertEqual(options.file, false)
      XCTAssertEqual(options.force, true)
      XCTAssertEqual(options.input, true)
    }
  }

  func testParsing_combined_2() {
    AssertParse(Bar.self, ["-fi"]) { options in
      XCTAssertEqual(options.file, false)
      XCTAssertEqual(options.force, true)
      XCTAssertEqual(options.input, true)
    }
  }

  func testParsing_combined_3() {
    AssertParse(Bar.self, ["-file", "-f"]) { options in
      XCTAssertEqual(options.file, true)
      XCTAssertEqual(options.force, true)
      XCTAssertEqual(options.input, false)
    }
  }

  func testParsing_combined_4() {
    AssertParse(Bar.self, ["-file", "-i"]) { options in
      XCTAssertEqual(options.file, true)
      XCTAssertEqual(options.force, false)
      XCTAssertEqual(options.input, true)
    }
  }

  func testParsing_combined_5() {
    AssertParse(Bar.self, ["-file", "-fi"]) { options in
      XCTAssertEqual(options.file, true)
      XCTAssertEqual(options.force, true)
      XCTAssertEqual(options.input, true)
    }
  }

  func testParsing_invalid() throws {
    //XCTAssertThrowsError(try Bar.parse(["-fil"]))
    XCTAssertThrowsError(try Bar.parse(["--file"]))
  }
}

extension LongNameWithSingleDashEndToEndTests {
  private struct Issue327: ParsableCommand {
    @Option(name: .customLong("argWithAnH", withSingleDash: true),
            parsing: .upToNextOption)
    var args: [String]
  }

  func testIssue327() {
    AssertParse(Issue327.self, ["-argWithAnH", "03ade86c0", "8f2058e3ade86c84ec5b"]) { issue327 in
      XCTAssertEqual(issue327.args, ["03ade86c0", "8f2058e3ade86c84ec5b"])
    }
  }
  
  private struct JoinedItem: ParsableCommand {
    @Option(name: .customLong("argWithAnH", withSingleDash: true))
    var arg: String
  }

  func testJoinedItem_Issue327() {
    AssertParse(JoinedItem.self, ["-argWithAnH=foo"]) { joinedItem in
      XCTAssertEqual(joinedItem.arg, "foo")
    }
  }
}
