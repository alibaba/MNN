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

final class ParsingEndToEndTests: XCTestCase {
}

struct Name {
  var rawValue: String

  init(rawValue: String) throws {
    if rawValue == "bad" {
      throw ValidationError("Bad input for name")
    }
    self.rawValue = rawValue
  }
}

extension Array where Element == Name {
  var rawValues: [String] {
    map { $0.rawValue }
  }
}

// MARK: -

fileprivate struct Foo: ParsableCommand {
  enum Subgroup: Equatable {
    case first(Int)
    case second(Int)

    static func makeFirst(_ str: String) throws -> Subgroup {
      guard let value = Int(str) else {
        throw ValidationError("Not a valid integer for 'first'")
      }
      return .first(value)
    }

    static func makeSecond(_ str: String) throws -> Subgroup {
      guard let value = Int(str) else {
        throw ValidationError("Not a valid integer for 'second'")
      }
      return .second(value)
    }
  }

  @Option(transform: Subgroup.makeFirst)
  var first: Subgroup

  @Argument(transform: Subgroup.makeSecond)
  var second: Subgroup
}

extension ParsingEndToEndTests {
  func testParsing() throws {
    AssertParse(Foo.self, ["--first", "1", "2"]) { foo in
      XCTAssertEqual(foo.first, .first(1))
      XCTAssertEqual(foo.second, .second(2))
    }
  }

  func testParsing_Fails() throws {
    // Failure inside custom parser
    XCTAssertThrowsError(try Foo.parse(["--first", "1", "bad"]))
    XCTAssertThrowsError(try Foo.parse(["--first", "bad", "2"]))
    XCTAssertThrowsError(try Foo.parse(["--first", "bad", "bad"]))

    // Missing argument failures
    XCTAssertThrowsError(try Foo.parse(["--first", "1"]))
    XCTAssertThrowsError(try Foo.parse(["5"]))
    XCTAssertThrowsError(try Foo.parse([]))
  }
}

// MARK: -

fileprivate struct Bar: ParsableCommand {
  @Option(transform: { try Name(rawValue: $0) })
  var firstName: Name = try! Name(rawValue: "none")

  @Argument(transform: { try Name(rawValue: $0) })
  var lastName: Name?
}

extension ParsingEndToEndTests {
  func testParsing_Defaults() throws {
    AssertParse(Bar.self, ["--first-name", "A", "B"]) { bar in
      XCTAssertEqual(bar.firstName.rawValue, "A")
      XCTAssertEqual(bar.lastName?.rawValue, "B")
    }

    AssertParse(Bar.self, ["B"]) { bar in
      XCTAssertEqual(bar.firstName.rawValue, "none")
      XCTAssertEqual(bar.lastName?.rawValue, "B")
    }

    AssertParse(Bar.self, ["--first-name", "A"]) { bar in
      XCTAssertEqual(bar.firstName.rawValue, "A")
      XCTAssertNil(bar.lastName)
    }

    AssertParse(Bar.self, []) { bar in
      XCTAssertEqual(bar.firstName.rawValue, "none")
      XCTAssertNil(bar.lastName)
    }
  }

  func testParsing_Defaults_Fails() throws {
    XCTAssertThrowsError(try Bar.parse(["--first-name", "bad"]))
    XCTAssertThrowsError(try Bar.parse(["bad"]))
  }
}

// MARK: -

fileprivate struct Qux: ParsableCommand {
  @Option(transform: { try Name(rawValue: $0) })
  var firstName: [Name] = []

  @Argument(transform: { try Name(rawValue: $0) })
  var lastName: [Name] = []
}

extension ParsingEndToEndTests {
  func testParsing_Array() throws {
    AssertParse(Qux.self, ["--first-name", "A", "B"]) { qux in
      XCTAssertEqual(qux.firstName.rawValues, ["A"])
      XCTAssertEqual(qux.lastName.rawValues, ["B"])
    }

    AssertParse(Qux.self, ["--first-name", "A", "--first-name", "B", "C", "D"]) { qux in
      XCTAssertEqual(qux.firstName.rawValues, ["A", "B"])
      XCTAssertEqual(qux.lastName.rawValues, ["C", "D"])
    }

    AssertParse(Qux.self, ["--first-name", "A", "--first-name", "B"]) { qux in
      XCTAssertEqual(qux.firstName.rawValues, ["A", "B"])
      XCTAssertEqual(qux.lastName.rawValues, [])
    }

    AssertParse(Qux.self, ["C", "D"]) { qux in
      XCTAssertEqual(qux.firstName.rawValues, [])
      XCTAssertEqual(qux.lastName.rawValues, ["C", "D"])
    }

    AssertParse(Qux.self, []) { qux in
      XCTAssertEqual(qux.firstName.rawValues, [])
      XCTAssertEqual(qux.lastName.rawValues, [])
    }
  }

  func testParsing_Array_Fails() {
    XCTAssertThrowsError(try Qux.parse(["--first-name", "A", "--first-name", "B", "C", "D", "bad"]))
    XCTAssertThrowsError(try Qux.parse(["--first-name", "A", "--first-name", "B", "--first-name", "bad", "C", "D"]))
  }
}

