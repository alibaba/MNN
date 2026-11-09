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

final class ShortNameEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Flag(name: [.long, .short])
  var verbose: Bool = false

  @Option(name: [.long, .short])
  var file: String?

  @Argument()
  var name: String
}

extension ShortNameEndToEndTests {
  func testParsing_withLongNames() throws {
    AssertParse(Bar.self, ["foo"]) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertNil(options.file)
      XCTAssertEqual(options.name, "foo")
    }

    AssertParse(Bar.self, ["--verbose", "--file", "myfile", "foo"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }
  }

  func testParsing_simple() throws {
    AssertParse(Bar.self, ["-v", "foo"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertNil(options.file)
      XCTAssertEqual(options.name, "foo")
    }

    AssertParse(Bar.self, ["-f", "myfile", "foo"]) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }

    AssertParse(Bar.self, ["-v", "-f", "myfile", "foo"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }
  }

  func testParsing_combined() throws {
    AssertParse(Bar.self, ["-vf", "myfile", "foo"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }

    AssertParse(Bar.self, ["-fv", "myfile", "foo"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }

    AssertParse(Bar.self, ["foo", "-fv", "myfile"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.file, "myfile")
      XCTAssertEqual(options.name, "foo")
    }
  }
}

// MARK: -

fileprivate struct Foo: ParsableArguments {
  @Option(name: [.long, .short])
  var name: String

  @Option(name: [.long, .short])
  var file: String

  @Option(name: [.long, .short])
  var city: String
}

extension ShortNameEndToEndTests {
  func testParsing_combinedShortNames() throws {
    AssertParse(Foo.self, ["-nfc", "name", "file", "city"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }

    AssertParse(Foo.self, ["-ncf", "name", "city", "file"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }

    AssertParse(Foo.self, ["-fnc", "file", "name", "city"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }

    AssertParse(Foo.self, ["-fcn", "file", "city", "name"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }

    AssertParse(Foo.self, ["-cnf", "city", "name", "file"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }

    AssertParse(Foo.self, ["-cfn", "city", "file", "name"]) { options in
      XCTAssertEqual(options.name, "name")
      XCTAssertEqual(options.file, "file")
      XCTAssertEqual(options.city, "city")
    }
  }
}
