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

final class OptionalEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Foo: ParsableArguments {
  struct Name: RawRepresentable, ExpressibleByArgument {
    var rawValue: String
  }
  @Option() var name: Name?
  @Option() var max: Int?
}

extension OptionalEndToEndTests {
  func testParsing_Optional() throws {
    AssertParse(Foo.self, []) { foo in
      XCTAssertNil(foo.name)
      XCTAssertNil(foo.max)
    }
    
    AssertParse(Foo.self, ["--name", "A"]) { foo in
      XCTAssertEqual(foo.name?.rawValue, "A")
      XCTAssertNil(foo.max)
    }
    
    AssertParse(Foo.self, ["--max", "3"]) { foo in
      XCTAssertNil(foo.name)
      XCTAssertEqual(foo.max, 3)
    }
    
    AssertParse(Foo.self, ["--max", "3", "--name", "A"]) { foo in
      XCTAssertEqual(foo.name?.rawValue, "A")
      XCTAssertEqual(foo.max, 3)
    }
  }
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  enum Format: String, ExpressibleByArgument {
    case A
    case B
    case C
  }
  @Option() var name: String? = nil
  @Option() var format: Format? = nil
  @Option() var foo: String
  @Argument() var bar: String? = nil
}

extension OptionalEndToEndTests {
  func testParsing_Optional_WithAllValues_1() {
    AssertParse(Bar.self, ["--name", "A", "--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithAllValues_2() {
    AssertParse(Bar.self, ["D", "--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithAllValues_3() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithMissingValues_1() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithMissingValues_2() {
    AssertParse(Bar.self, ["D", "--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithMissingValues_3() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }
  
  func testParsing_Optional_WithMissingValues_4() {
    AssertParse(Bar.self, ["--name", "A", "--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_5() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format,.B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_6() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_7() {
    AssertParse(Bar.self, ["--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, nil)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_8() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_9() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_10() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, nil)
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_WithMissingValues_11() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }
  
  func testParsing_Optional_Fails() throws {
    XCTAssertThrowsError(try Bar.parse([]))
    XCTAssertThrowsError(try Bar.parse(["--format", "ZZ", "--foo", "C"]))
    XCTAssertThrowsError(try Bar.parse(["--fooz", "C"]))
    XCTAssertThrowsError(try Bar.parse(["--nam", "A", "--foo", "C"]))
    XCTAssertThrowsError(try Bar.parse(["--name"]))
    XCTAssertThrowsError(try Bar.parse(["A"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "D"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "--foo"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "--format", "B"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "-f"]))
    XCTAssertThrowsError(try Bar.parse(["D", "--name", "A"]))
    XCTAssertThrowsError(try Bar.parse(["-f", "--name", "A"]))
  }
}

extension OptionalEndToEndTests {
  // Compilation test: https://github.com/apple/swift-argument-parser/issues/618
  private struct Command: ParsableCommand {
    struct MyError: Error {}
    struct Foo {
      init?(string: String) { return nil }
    }

    @Option(transform: {
      guard let foo = Foo(string: $0) else {
        throw MyError()
      }
      return foo
    })
    var testOption: Foo?
    
    @Argument(transform: {
      guard let foo = Foo(string: $0) else {
        throw MyError()
      }
      return foo
    })
    var testArgument: Foo?
  }
}
