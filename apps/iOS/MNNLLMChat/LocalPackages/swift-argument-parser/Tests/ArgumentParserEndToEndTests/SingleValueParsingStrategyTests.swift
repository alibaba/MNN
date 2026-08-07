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

final class SingleValueParsingStrategyTests: XCTestCase {
}

// MARK: Scanning for Value

fileprivate struct Bar: ParsableArguments {
  @Option(parsing: .scanningForValue) var name: String
  @Option(parsing: .scanningForValue) var format: String
  @Option(parsing: .scanningForValue) var input: String
}

extension SingleValueParsingStrategyTests {
  func testParsing_scanningForValue_1() throws {
    AssertParse(Bar.self, ["--name", "Foo", "--format", "Bar", "--input", "Baz"]) { bar in
      XCTAssertEqual(bar.name, "Foo")
      XCTAssertEqual(bar.format, "Bar")
      XCTAssertEqual(bar.input, "Baz")
    }
  }
  
  func testParsing_scanningForValue_2() throws {
    AssertParse(Bar.self, ["--name", "--format", "Foo", "Bar", "--input", "Baz"]) { bar in
      XCTAssertEqual(bar.name, "Foo")
      XCTAssertEqual(bar.format, "Bar")
      XCTAssertEqual(bar.input, "Baz")
    }
  }
  
  func testParsing_scanningForValue_3() throws {
    AssertParse(Bar.self, ["--name", "--format", "--input", "Foo", "Bar", "Baz"]) { bar in
      XCTAssertEqual(bar.name, "Foo")
      XCTAssertEqual(bar.format, "Bar")
      XCTAssertEqual(bar.input, "Baz")
    }
  }
}

// MARK: Unconditional

fileprivate struct Baz: ParsableArguments {
  @Option(parsing: .unconditional) var name: String
  @Option(parsing: .unconditional) var format: String
  @Option(parsing: .unconditional) var input: String
}

extension SingleValueParsingStrategyTests {
  func testParsing_unconditional_1() throws {
    AssertParse(Baz.self, ["--name", "Foo", "--format", "Bar", "--input", "Baz"]) { bar in
      XCTAssertEqual(bar.name, "Foo")
      XCTAssertEqual(bar.format, "Bar")
      XCTAssertEqual(bar.input, "Baz")
    }
  }
  
  func testParsing_unconditional_2() throws {
    AssertParse(Baz.self, ["--name", "--name", "--format", "--format", "--input", "--input"]) { bar in
      XCTAssertEqual(bar.name, "--name")
      XCTAssertEqual(bar.format, "--format")
      XCTAssertEqual(bar.input, "--input")
    }
  }
  
  func testParsing_unconditional_3() throws {
    AssertParse(Baz.self, ["--name", "-Foo", "--format", "-Bar", "--input", "-Baz"]) { bar in
      XCTAssertEqual(bar.name, "-Foo")
      XCTAssertEqual(bar.format, "-Bar")
      XCTAssertEqual(bar.input, "-Baz")
    }
  }
}
