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

final class SimpleEndToEndTests: XCTestCase {
}

// MARK: Single value String

fileprivate struct Bar: ParsableArguments {
  @Option() var name: String
}

extension SimpleEndToEndTests {
  func testParsing_SingleOption() throws {
    AssertParse(Bar.self, ["--name", "Bar"]) { bar in
      XCTAssertEqual(bar.name, "Bar")
    }
    AssertParse(Bar.self, ["--name", " foo "]) { bar in
      XCTAssertEqual(bar.name, " foo ")
    }
  }
  
  func testParsing_SingleOption_Fails() throws {
    XCTAssertThrowsError(try Bar.parse([]))
    XCTAssertThrowsError(try Bar.parse(["--name"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "--foo"]))
    XCTAssertThrowsError(try Bar.parse(["Bar"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "Bar", "Baz"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "Bar", "--foo"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "Bar", "--foo", "Foo"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "Bar", "-f"]))
    XCTAssertThrowsError(try Bar.parse(["--foo", "--name", "Bar"]))
    XCTAssertThrowsError(try Bar.parse(["--foo", "Foo", "--name", "Bar"]))
    XCTAssertThrowsError(try Bar.parse(["-f", "--name", "Bar"]))
  }
}

// MARK: Single value Int

fileprivate struct Foo: ParsableArguments {
  @Option() var count: Int
}

extension SimpleEndToEndTests {
  func testParsing_SingleOption_Int() throws {
    AssertParse(Foo.self, ["--count", "42"]) { foo in
      XCTAssertEqual(foo.count, 42)
    }
  }
  
  func testParsing_SingleOption_Int_Fails() throws {
    XCTAssertThrowsError(try Foo.parse([]))
    XCTAssertThrowsError(try Foo.parse(["--count"]))
    XCTAssertThrowsError(try Foo.parse(["--count", "a"]))
    XCTAssertThrowsError(try Foo.parse(["Bar"]))
    XCTAssertThrowsError(try Foo.parse(["--count", "42", "Baz"]))
    XCTAssertThrowsError(try Foo.parse(["--count", "42", "--foo"]))
    XCTAssertThrowsError(try Foo.parse(["--count", "42", "--foo", "Foo"]))
    XCTAssertThrowsError(try Foo.parse(["--count", "42", "-f"]))
    XCTAssertThrowsError(try Foo.parse(["--foo", "--count", "42"]))
    XCTAssertThrowsError(try Foo.parse(["--foo", "Foo", "--count", "42"]))
    XCTAssertThrowsError(try Foo.parse(["-f", "--count", "42"]))
  }
}

// MARK: Two values

fileprivate struct Baz: ParsableArguments {
  @Option() var name: String
  @Option() var format: String
}

extension SimpleEndToEndTests {
  func testParsing_TwoOptions_1() throws {
    AssertParse(Baz.self, ["--name", "Bar", "--format", "Foo"]) { baz in
      XCTAssertEqual(baz.name, "Bar")
      XCTAssertEqual(baz.format, "Foo")
    }
  }
  
  func testParsing_TwoOptions_2() throws {
    AssertParse(Baz.self, ["--format", "Foo", "--name", "Bar"]) { baz in
      XCTAssertEqual(baz.name, "Bar")
      XCTAssertEqual(baz.format, "Foo")
    }
  }
  
  func testParsing_TwoOptions_Fails() throws {
    XCTAssertThrowsError(try Baz.parse(["--nam", "Bar", "--format", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "Bar", "--forma", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "Bar"]))
    XCTAssertThrowsError(try Baz.parse(["--format", "Foo"]))
    
    XCTAssertThrowsError(try Baz.parse(["--name", "--format", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "Bar", "--format"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "Bar", "--format", "Foo", "Baz"]))
    XCTAssertThrowsError(try Baz.parse(["Bar", "--name", "--format", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["Bar", "--name", "Foo", "--format"]))
    XCTAssertThrowsError(try Baz.parse(["Bar", "Foo", "--name", "--format"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "--name", "Bar", "--format", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "Bar", "--format", "--format", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--format", "--name", "Bar", "Foo"]))
    XCTAssertThrowsError(try Baz.parse(["--name", "--format", "Bar", "Foo"]))
  }
}
