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

final class RawRepresentableEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  struct Identifier: RawRepresentable, Equatable, ExpressibleByArgument {
    var rawValue: Int
  }
  
  @Option() var identifier: Identifier
}

extension RawRepresentableEndToEndTests {
  func testParsing_SingleOption() throws {
    AssertParse(Bar.self, ["--identifier", "123"]) { bar in
      XCTAssertEqual(bar.identifier, Bar.Identifier(rawValue: 123))
    }
  }
  
  func testParsing_SingleOptionMultipleTimes() throws {
    AssertParse(Bar.self, ["--identifier", "123", "--identifier", "456"]) { bar in
      XCTAssertEqual(bar.identifier, Bar.Identifier(rawValue: 456))
    }
  }
  
  func testParsing_SingleOption_Fails() throws {
    XCTAssertThrowsError(try Bar.parse([]))
    XCTAssertThrowsError(try Bar.parse(["--identifier"]))
    XCTAssertThrowsError(try Bar.parse(["--identifier", "not a number"]))
    XCTAssertThrowsError(try Bar.parse(["--identifier", "123.456"]))
  }
}
