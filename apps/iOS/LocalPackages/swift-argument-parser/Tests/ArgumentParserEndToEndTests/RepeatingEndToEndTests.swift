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

final class RepeatingEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Option() var name: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingString() throws {
    AssertParse(Bar.self, []) { bar in
      XCTAssertTrue(bar.name.isEmpty)
    }

    AssertParse(Bar.self, ["--name", "Bar"]) { bar in
      XCTAssertEqual(bar.name.count, 1)
      XCTAssertEqual(bar.name.first, "Bar")
    }

    AssertParse(Bar.self, ["--name", "Bar", "--name", "Foo"]) { bar in
      XCTAssertEqual(bar.name.count, 2)
      XCTAssertEqual(bar.name.first, "Bar")
      XCTAssertEqual(bar.name.last, "Foo")
    }
  }
}

// MARK: -

fileprivate struct Foo: ParsableArguments {
  @Flag()
  var verbose: Int
}

extension RepeatingEndToEndTests {
  func testParsing_incrementInteger() throws {
    AssertParse(Foo.self, []) { options in
      XCTAssertEqual(options.verbose, 0)
    }
    AssertParse(Foo.self, ["--verbose"]) { options in
      XCTAssertEqual(options.verbose, 1)
    }
    AssertParse(Foo.self, ["--verbose", "--verbose"]) { options in
      XCTAssertEqual(options.verbose, 2)
    }
  }
}

// MARK: -

fileprivate struct Baz: ParsableArguments {
  @Flag var verbose: Bool = false
  @Option(parsing: .remaining) var names: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingStringRemaining_1() {
    AssertParse(Baz.self, []) { baz in
      XCTAssertFalse(baz.verbose)
      XCTAssertTrue(baz.names.isEmpty)
    }
  }

  func testParsing_repeatingStringRemaining_2() {
    AssertParse(Baz.self, ["--names"]) { baz in
      XCTAssertFalse(baz.verbose)
      XCTAssertTrue(baz.names.isEmpty)
    }
  }

  func testParsing_repeatingStringRemaining_3() {
    AssertParse(Baz.self, ["--names", "one"]) { baz in
      XCTAssertFalse(baz.verbose)
      XCTAssertEqual(baz.names, ["one"])
    }
  }

  func testParsing_repeatingStringRemaining_4() {
    AssertParse(Baz.self, ["--names", "one", "two"]) { baz in
      XCTAssertFalse(baz.verbose)
      XCTAssertEqual(baz.names, ["one", "two"])
    }
  }

  func testParsing_repeatingStringRemaining_5() {
    AssertParse(Baz.self, ["--verbose", "--names", "one", "two"]) { baz in
      XCTAssertTrue(baz.verbose)
      XCTAssertEqual(baz.names, ["one", "two"])
    }
  }

  func testParsing_repeatingStringRemaining_6() {
    AssertParse(Baz.self, ["--names", "one", "two", "--verbose"]) { baz in
      XCTAssertFalse(baz.verbose)
      XCTAssertEqual(baz.names, ["one", "two", "--verbose"])
    }
  }

  func testParsing_repeatingStringRemaining_7() {
    AssertParse(Baz.self, ["--verbose", "--names", "one", "two", "--verbose"]) { baz in
      XCTAssertTrue(baz.verbose)
      XCTAssertEqual(baz.names, ["one", "two", "--verbose"])
    }
  }

  func testParsing_repeatingStringRemaining_8() {
    AssertParse(Baz.self, ["--verbose", "--names", "one", "two", "--verbose", "--other", "three"]) { baz in
      XCTAssertTrue(baz.verbose)
      XCTAssertEqual(baz.names, ["one", "two", "--verbose", "--other", "three"])
    }
  }
}

// MARK: -

fileprivate struct Outer: ParsableCommand {
  static let configuration = CommandConfiguration(subcommands: [Inner.self])
}

fileprivate struct Inner: ParsableCommand {
  @Flag
  var verbose: Bool = false

  @Argument(parsing: .captureForPassthrough)
  var files: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_subcommandRemaining() {
    AssertParseCommand(
      Outer.self, Inner.self,
      ["inner", "--verbose", "one", "two", "--", "three", "--other"])
    { inner in
      XCTAssertTrue(inner.verbose)
      XCTAssertEqual(inner.files, ["one", "two", "--", "three", "--other"])
    }
  }
}

// MARK: -

fileprivate struct Qux: ParsableArguments {
  @Option(parsing: .upToNextOption) var names: [String] = []
  @Flag var verbose: Bool = false
  @Argument() var extra: String?
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingStringUpToNext() throws {
    AssertParse(Qux.self, []) { qux in
      XCTAssertFalse(qux.verbose)
      XCTAssertTrue(qux.names.isEmpty)
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--names", "one"]) { qux in
      XCTAssertFalse(qux.verbose)
      XCTAssertEqual(qux.names, ["one"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--names", "one", "two", "--verbose", "--names", "three", "--names", "four"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two", "three", "four"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["extra", "--names", "one", "--names", "two", "--verbose", "--names", "three", "four"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two", "three", "four"])
      XCTAssertEqual(qux.extra, "extra")
    }

    AssertParse(Qux.self, ["--names", "one", "two"]) { qux in
      XCTAssertFalse(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--names", "one", "two", "--verbose"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--names", "one", "two", "--verbose", "three"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two"])
      XCTAssertEqual(qux.extra, "three")
    }

    AssertParse(Qux.self, ["--verbose", "--names", "one", "two"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--verbose", "--names=one"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--verbose", "--names=one", "two"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one", "two"])
      XCTAssertNil(qux.extra)
    }

    AssertParse(Qux.self, ["--names=one", "--verbose", "two"]) { qux in
      XCTAssertTrue(qux.verbose)
      XCTAssertEqual(qux.names, ["one"])
      XCTAssertEqual(qux.extra, "two")
    }
  }

  func testParsing_repeatingStringUpToNext_Fails() throws {
    XCTAssertThrowsError(try Qux.parse(["--names", "one", "--other"]))
    XCTAssertThrowsError(try Qux.parse(["--names", "one", "two", "--other"]))
    XCTAssertThrowsError(try Qux.parse(["--names", "--other"]))
    XCTAssertThrowsError(try Qux.parse(["--names", "--verbose"]))
    XCTAssertThrowsError(try Qux.parse(["--names", "--verbose", "three"]))
  }
}

// MARK: -

fileprivate struct Wobble: ParsableArguments {
  struct WobbleError: Error {}
  struct Name: Equatable, Sendable {
    var value: String

    init(_ value: String) throws {
      if value == "bad" { throw WobbleError() }
      self.value = value
    }
  }
  @Option(transform: Name.init) var names: [Name] = []
  @Option(parsing: .upToNextOption, transform: Name.init) var moreNames: [Name] = []
  @Option(parsing: .remaining, transform: Name.init) var evenMoreNames: [Name] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingWithTransform() throws {
    let names = ["--names", "one", "--names", "two"]
    let moreNames = ["--more-names", "three", "four", "five"]
    let evenMoreNames = ["--even-more-names", "six", "--seven", "--eight"]

    AssertParse(Wobble.self, []) { wobble in
      XCTAssertTrue(wobble.names.isEmpty)
      XCTAssertTrue(wobble.moreNames.isEmpty)
      XCTAssertTrue(wobble.evenMoreNames.isEmpty)
    }

    AssertParse(Wobble.self, names) { wobble in
      XCTAssertEqual(wobble.names.map { $0.value }, ["one", "two"])
      XCTAssertTrue(wobble.moreNames.isEmpty)
      XCTAssertTrue(wobble.evenMoreNames.isEmpty)
    }

    AssertParse(Wobble.self, moreNames) { wobble in
      XCTAssertTrue(wobble.names.isEmpty)
      XCTAssertEqual(wobble.moreNames.map { $0.value }, ["three", "four", "five"])
      XCTAssertTrue(wobble.evenMoreNames.isEmpty)
    }

    AssertParse(Wobble.self, evenMoreNames) { wobble in
      XCTAssertTrue(wobble.names.isEmpty)
      XCTAssertTrue(wobble.moreNames.isEmpty)
      XCTAssertEqual(wobble.evenMoreNames.map { $0.value }, ["six", "--seven", "--eight"])
    }

    AssertParse(Wobble.self, Array([names, moreNames, evenMoreNames].joined())) { wobble in
      XCTAssertEqual(wobble.names.map { $0.value }, ["one", "two"])
      XCTAssertEqual(wobble.moreNames.map { $0.value }, ["three", "four", "five"])
      XCTAssertEqual(wobble.evenMoreNames.map { $0.value }, ["six", "--seven", "--eight"])
    }

    AssertParse(Wobble.self, Array([moreNames, names, evenMoreNames].joined())) { wobble in
      XCTAssertEqual(wobble.names.map { $0.value }, ["one", "two"])
      XCTAssertEqual(wobble.moreNames.map { $0.value }, ["three", "four", "five"])
      XCTAssertEqual(wobble.evenMoreNames.map { $0.value }, ["six", "--seven", "--eight"])
    }

    AssertParse(Wobble.self, Array([moreNames, evenMoreNames, names].joined())) { wobble in
      XCTAssertTrue(wobble.names.isEmpty)
      XCTAssertEqual(wobble.moreNames.map { $0.value }, ["three", "four", "five"])
      XCTAssertEqual(wobble.evenMoreNames.map { $0.value }, ["six", "--seven", "--eight", "--names", "one", "--names", "two"])
    }
  }

  func testParsing_repeatingWithTransform_Fails() throws {
    XCTAssertThrowsError(try Wobble.parse(["--names", "one", "--other"]))
    XCTAssertThrowsError(try Wobble.parse(["--more-names", "one", "--other"]))

    XCTAssertThrowsError(try Wobble.parse(["--names", "one", "--names", "bad"]))
    XCTAssertThrowsError(try Wobble.parse(["--more-names", "one", "two", "bad", "--names", "one"]))
    XCTAssertThrowsError(try Wobble.parse(["--even-more-names", "one", "two", "--names", "one", "bad"]))
  }
}

// MARK: -

fileprivate struct Weazle: ParsableArguments {
  @Flag var verbose: Bool = false
  @Argument() var names: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingArgument() throws {
    AssertParse(Weazle.self, ["one", "two", "three", "--verbose"]) { weazle in
      XCTAssertTrue(weazle.verbose)
      XCTAssertEqual(weazle.names, ["one", "two", "three"])
    }

    AssertParse(Weazle.self, ["--verbose", "one", "two", "three"]) { weazle in
      XCTAssertTrue(weazle.verbose)
      XCTAssertEqual(weazle.names, ["one", "two", "three"])
    }

    AssertParse(Weazle.self, ["one", "two", "three", "--", "--other", "--verbose"]) { weazle in
      XCTAssertFalse(weazle.verbose)
      XCTAssertEqual(weazle.names, ["one", "two", "three", "--other", "--verbose"])
    }
  }
}

// MARK: -

struct PerformanceTest: ParsableCommand {
  @Option(name: .short) var bundleIdentifiers: [String] = []

  mutating func run() throws { print(bundleIdentifiers) }
}

fileprivate func argumentGenerator(_ count: Int) -> [String] {
  Array((1...count).map { ["-b", "bundle-id\($0)"] }.joined())
}

fileprivate func time(_ body: () -> Void) -> TimeInterval {
  let start = Date()
  body()
  return Date().timeIntervalSince(start)
}

extension RepeatingEndToEndTests {
  // A regression test against array parsing performance going non-linear.
  func testParsing_repeatingPerformance() throws {
    let timeFor20 = time {
      AssertParse(PerformanceTest.self, argumentGenerator(100)) { test in
        XCTAssertEqual(100, test.bundleIdentifiers.count)
      }
    }
    let timeFor40 = time {
      AssertParse(PerformanceTest.self, argumentGenerator(200)) { test in
        XCTAssertEqual(200, test.bundleIdentifiers.count)
      }
    }

    XCTAssertLessThan(timeFor40, timeFor20 * 10)
  }
}
