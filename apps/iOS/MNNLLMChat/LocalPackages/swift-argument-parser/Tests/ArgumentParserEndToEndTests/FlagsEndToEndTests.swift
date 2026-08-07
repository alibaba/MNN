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

final class FlagsEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Flag
  var verbose: Bool = false

  @Flag(inversion: .prefixedNo)
  var extattr: Bool = false

  @Flag(inversion: .prefixedNo, exclusivity: .exclusive)
  var extattr2: Bool?

  @Flag(inversion: .prefixedEnableDisable, exclusivity: .chooseFirst)
  var logging: Bool = false
}

extension FlagsEndToEndTests {
  func testParsing_defaultValue() throws {
    AssertParse(Bar.self, []) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.extattr, false)
      XCTAssertEqual(options.extattr2, nil)
    }
  }

  func testParsing_settingValue() throws {
    AssertParse(Bar.self, ["--verbose"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.extattr, false)
      XCTAssertEqual(options.extattr2, nil)
    }

    AssertParse(Bar.self, ["--extattr"]) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.extattr, true)
      XCTAssertEqual(options.extattr2, nil)
    }

    AssertParse(Bar.self, ["--extattr2"]) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.extattr, false)
      XCTAssertEqual(options.extattr2, .some(true))
    }
  }

  func testParsing_invert() throws {
    AssertParse(Bar.self, ["--no-extattr"]) { options in
      XCTAssertEqual(options.extattr, false)
    }
    AssertParse(Bar.self, ["--extattr", "--no-extattr"]) { options in
      XCTAssertEqual(options.extattr, false)
    }
    AssertParse(Bar.self, ["--extattr", "--no-extattr", "--no-extattr"]) { options in
      XCTAssertEqual(options.extattr, false)
    }
    AssertParse(Bar.self, ["--no-extattr", "--no-extattr", "--extattr"]) { options in
      XCTAssertEqual(options.extattr, true)
    }
    AssertParse(Bar.self, ["--extattr", "--no-extattr", "--extattr"]) { options in
      XCTAssertEqual(options.extattr, true)
    }
    AssertParse(Bar.self, ["--enable-logging"]) { options in
      XCTAssertEqual(options.logging, true)
    }
    AssertParse(Bar.self, ["--no-extattr2", "--no-extattr2"]) { options in
      XCTAssertEqual(options.extattr2, false)
    }
    AssertParse(Bar.self, ["--disable-logging", "--enable-logging"]) { options in
      XCTAssertEqual(options.logging, false)
    }
  }
}

fileprivate struct Foo: ParsableArguments {
  @Flag(inversion: .prefixedEnableDisable)
  var index: Bool = false
  @Flag(inversion: .prefixedEnableDisable)
  var sandbox: Bool = true
  @Flag(inversion: .prefixedEnableDisable)
  var requiredElement: Bool
  @Flag(inversion: .prefixedEnableDisable)
  var optional: Bool? = nil
}

extension FlagsEndToEndTests {
  func testParsingEnableDisable_defaultValue() throws {
    AssertParse(Foo.self, ["--enable-required-element"]) { options in
      XCTAssertEqual(options.index, false)
      XCTAssertEqual(options.sandbox, true)
      XCTAssertEqual(options.requiredElement, true)
      XCTAssertNil(options.optional)
    }
  }

  func testParsingEnableDisable_disableAll() throws {
    AssertParse(Foo.self, ["--disable-index", "--disable-sandbox", "--disable-required-element", "--disable-optional"]) { options in
      XCTAssertEqual(options.index, false)
      XCTAssertEqual(options.sandbox, false)
      XCTAssertEqual(options.requiredElement, false)
      XCTAssertEqual(options.optional, false)
    }
  }

  func testParsingEnableDisable_enableAll() throws {
    AssertParse(Foo.self, ["--enable-index", "--enable-sandbox", "--enable-required-element", "--enable-optional"]) { options in
      XCTAssertEqual(options.index, true)
      XCTAssertEqual(options.sandbox, true)
      XCTAssertEqual(options.requiredElement, true)
      XCTAssertEqual(options.optional, true)
    }
  }

  func testParsingEnableDisable_Fails() throws {
    XCTAssertThrowsError(try Foo.parse([]))
    XCTAssertThrowsError(try Foo.parse(["--disable-index"]))
    XCTAssertThrowsError(try Foo.parse(["--disable-sandbox"]))
  }
}

enum Color: String, EnumerableFlag {
  case pink
  case purple
  case silver
}

enum Size: String, EnumerableFlag {
  case small
  case medium
  case large
  case extraLarge
  case humongous

  static func name(for value: Size) -> NameSpecification {
    switch value {
    case .small, .medium, .large:
      return .shortAndLong
    case .humongous:
      return [.long, .customLong("huge")]
    default:
      return .long
    }
  }

  static func help(for value: Size) -> ArgumentHelp? {
    switch value {
    case .small:
      return "A smallish size."
    case .medium:
      return "Not too big, not too small."
    case .humongous:
      return "Roughly the size of a barge."
    case .large, .extraLarge:
      return nil
    }
  }
}

enum Shape: String, EnumerableFlag {
  case round
  case square
  case oblong
}

fileprivate struct Baz: ParsableArguments {
  @Flag()
  var color: Color

  @Flag(help: "The size to use.")
  var size: Size = .small

  @Flag()
  var shape: Shape?
}

extension FlagsEndToEndTests {
  func testParsingCaseIterable_defaultValues() throws {
    AssertParse(Baz.self, ["--pink"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .small)
      XCTAssertEqual(options.shape, nil)
    }

    AssertParse(Baz.self, ["--pink", "--medium"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .medium)
      XCTAssertEqual(options.shape, nil)
    }

    AssertParse(Baz.self, ["--pink", "--round"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .small)
      XCTAssertEqual(options.shape, .round)
    }
  }

  func testParsingCaseIterable_AllValues() throws {
    AssertParse(Baz.self, ["--pink", "--small", "--round"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .small)
      XCTAssertEqual(options.shape, .round)
    }

    AssertParse(Baz.self, ["--purple", "--medium", "--square"]) { options in
      XCTAssertEqual(options.color, .purple)
      XCTAssertEqual(options.size, .medium)
      XCTAssertEqual(options.shape, .square)
    }

    AssertParse(Baz.self, ["--silver", "--large", "--oblong"]) { options in
      XCTAssertEqual(options.color, .silver)
      XCTAssertEqual(options.size, .large)
      XCTAssertEqual(options.shape, .oblong)
    }
  }

  func testParsingCaseIterable_CustomName() throws {
    AssertParse(Baz.self, ["--pink", "--extra-large"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .extraLarge)
      XCTAssertEqual(options.shape, nil)
    }

    AssertParse(Baz.self, ["--pink", "--huge"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .humongous)
      XCTAssertEqual(options.shape, nil)
    }

    AssertParse(Baz.self, ["--pink", "--humongous"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .humongous)
      XCTAssertEqual(options.shape, nil)
    }

    AssertParse(Baz.self, ["--pink", "--huge", "--humongous"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.size, .humongous)
      XCTAssertEqual(options.shape, nil)
    }
  }

  func testParsingCaseIterable_Help() throws {
    AssertHelp(.default, for: Baz.self, equals: """
            USAGE: baz --pink --purple --silver [--small] [--medium] [--large] [--extra-large] [--humongous] [--round] [--square] [--oblong]

            OPTIONS:
              --pink/--purple/--silver
              -s, --small             A smallish size. (default: --small)
              -m, --medium            Not too big, not too small.
              -l, --large             The size to use.
              --extra-large           The size to use.
              --humongous, --huge     Roughly the size of a barge.
              --round/--square/--oblong
              -h, --help              Show help information.
            
            """)
  }
  
  func testParsingCaseIterable_Fails() throws {
    // Missing color
    XCTAssertThrowsError(try Baz.parse([]))
    XCTAssertThrowsError(try Baz.parse(["--large", "--square"]))
    // Repeating flags
    XCTAssertThrowsError(try Baz.parse(["--pink", "--purple"]))
    XCTAssertThrowsError(try Baz.parse(["--pink", "--small", "--large"]))
    XCTAssertThrowsError(try Baz.parse(["--pink", "--round", "--square"]))
    // Case name instead of raw value
    XCTAssertThrowsError(try Baz.parse(["--pink", "--extraLarge"]))
  }
}

fileprivate struct Qux: ParsableArguments {
  @Flag()
  var color: [Color] = []

  @Flag()
  var size: [Size] = [.small, .medium]
}

extension FlagsEndToEndTests {
  func testParsingCaseIterableArray_Values() throws {
    AssertParse(Qux.self, []) { options in
      XCTAssertEqual(options.color, [])
      XCTAssertEqual(options.size, [.small, .medium])
    }
    AssertParse(Qux.self, ["--pink"]) { options in
      XCTAssertEqual(options.color, [.pink])
      XCTAssertEqual(options.size, [.small, .medium])
    }
    AssertParse(Qux.self, ["--pink", "--purple", "--small"]) { options in
      XCTAssertEqual(options.color, [.pink, .purple])
      XCTAssertEqual(options.size, [.small])
    }
    AssertParse(Qux.self, ["--pink", "--small", "--purple", "--medium"]) { options in
      XCTAssertEqual(options.color, [.pink, .purple])
      XCTAssertEqual(options.size, [.small, .medium])
    }
    AssertParse(Qux.self, ["--pink", "--pink", "--purple", "--pink"]) { options in
      XCTAssertEqual(options.color, [.pink, .pink, .purple, .pink])
      XCTAssertEqual(options.size, [.small, .medium])
    }
  }

  func testParsingCaseIterableArray_Fails() throws {
    XCTAssertThrowsError(try Qux.parse(["--pink", "--small", "--bloop"]))
  }
}

fileprivate struct RepeatOK: ParsableArguments {
  @Flag(exclusivity: .chooseFirst)
  var color: Color

  @Flag(exclusivity: .chooseLast)
  var shape: Shape

  @Flag(exclusivity: .exclusive)
  var size: Size = .small
}

extension FlagsEndToEndTests {
  func testParsingCaseIterable_RepeatableFlags() throws {
    AssertParse(RepeatOK.self, ["--pink", "--purple", "--square"]) { options in
      XCTAssertEqual(options.color, .pink)
      XCTAssertEqual(options.shape, .square)
    }

    AssertParse(RepeatOK.self, ["--round", "--oblong", "--silver"]) { options in
      XCTAssertEqual(options.color, .silver)
      XCTAssertEqual(options.shape, .oblong)
    }

    AssertParse(RepeatOK.self, ["--large", "--pink", "--round", "-l"]) { options in
      XCTAssertEqual(options.size, .large)
    }
  }
}
