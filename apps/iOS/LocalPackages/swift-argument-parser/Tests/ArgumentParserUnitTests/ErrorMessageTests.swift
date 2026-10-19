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
@testable import ArgumentParser

final class ErrorMessageTests: XCTestCase {}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  @Option() var name: String
  @Option(name: [.short, .long]) var format: String
}

extension ErrorMessageTests {
  func testMissing_1() {
    AssertErrorMessage(Bar.self, [], "Missing expected argument '--name <name>'")
  }

  func testMissing_2() {
    AssertErrorMessage(Bar.self, ["--name", "a"], "Missing expected argument '--format <format>'")
  }

  func testUnknownOption_1() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format", "b", "--verbose"], "Unknown option '--verbose'")
  }

  func testUnknownOption_2() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format", "b", "-q"], "Unknown option '-q'")
  }

  func testUnknownOption_3() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format", "b", "-bar"], "Unknown option '-bar'")
  }

  func testUnknownOption_4() {
    AssertErrorMessage(Bar.self, ["--name", "a", "-foz", "b"], "Unknown option '-o'")
  }

  func testMissingValue_1() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format"], "Missing value for '--format <format>'")
  }

  func testMissingValue_2() {
    AssertErrorMessage(Bar.self, ["--name", "a", "-f"], "Missing value for '-f <format>'")
  }

  func testUnusedValue_1() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format", "f", "b"], "Unexpected argument 'b'")
  }

  func testUnusedValue_2() {
    AssertErrorMessage(Bar.self, ["--name", "a", "--format", "f", "b", "baz"], "2 unexpected arguments: 'b', 'baz'")
  }
}

fileprivate enum Format: String, Equatable, Decodable, ExpressibleByArgument, CaseIterable {
  case text
  case json
  case csv
}

fileprivate enum Name: String, Equatable, Decodable, ExpressibleByArgument, CaseIterable {
  case bruce
  case clint
  case hulk
  case natasha
  case steve
  case thor
  case tony
}

fileprivate enum Counter: Int, ExpressibleByArgument, CaseIterable {
  case one = 1
  case two, three, four
}

fileprivate struct Foo: ParsableArguments {
  @Option(name: [.short, .long])
  var format: Format
  @Option(name: [.short, .long])
  var name: Name?
}

fileprivate struct EnumWithFewCasesArrayArgument: ParsableArguments {
  @Argument
  var formats: [Format]
}

fileprivate struct EnumWithManyCasesArrayArgument: ParsableArguments {
  @Argument
  var names: [Name]
}

fileprivate struct EnumWithIntRawValue: ParsableArguments {
  @Option var counter: Counter
}

extension ErrorMessageTests {
  func testWrongEnumValue() {
    AssertErrorMessage(Foo.self, ["--format", "png"], "The value 'png' is invalid for '--format <format>'. Please provide one of 'text', 'json' or 'csv'.")
    AssertErrorMessage(Foo.self, ["-f", "png"], "The value 'png' is invalid for '-f <format>'. Please provide one of 'text', 'json' or 'csv'.")
    AssertErrorMessage(Foo.self, ["-f", "text", "--name", "loki"],
      """
      The value 'loki' is invalid for '--name <name>'. Please provide one of the following:
        - bruce
        - clint
        - hulk
        - natasha
        - steve
        - thor
        - tony
      """)
    AssertErrorMessage(Foo.self, ["-f", "text", "-n", "loki"],
      """
      The value 'loki' is invalid for '-n <name>'. Please provide one of the following:
        - bruce
        - clint
        - hulk
        - natasha
        - steve
        - thor
        - tony
      """)
    AssertErrorMessage(EnumWithFewCasesArrayArgument.self, ["png"], "The value 'png' is invalid for '<formats>'. Please provide one of 'text', 'json' or 'csv'.")
    AssertErrorMessage(EnumWithManyCasesArrayArgument.self, ["loki"],
      """
      The value 'loki' is invalid for '<names>'. Please provide one of the following:
        - bruce
        - clint
        - hulk
        - natasha
        - steve
        - thor
        - tony
      """)
    
    AssertErrorMessage(EnumWithIntRawValue.self, ["--counter", "one"], """
      The value 'one' is invalid for '--counter <counter>'. \
      Please provide one of '1', '2', '3' or '4'.
      """)
  }
}

fileprivate struct Baz: ParsableArguments {
  @Flag
  var verbose: Bool = false
}

extension ErrorMessageTests {
  func testUnexpectedValue() {
    AssertErrorMessage(Baz.self, ["--verbose=foo"], "The option '--verbose' does not take any value, but 'foo' was specified.")
  }
}

fileprivate struct Qux: ParsableArguments {
  @Argument()
  var firstNumber: Int

  @Option(name: .customLong("number-two"))
  var secondNumber: Int
}

extension ErrorMessageTests {
  func testMissingArgument() {
    AssertErrorMessage(Qux.self, ["--number-two", "2"], "Missing expected argument '<first-number>'")
  }

  func testInvalidNumber() {
    AssertErrorMessage(Qux.self, ["--number-two", "2", "a"], "The value 'a' is invalid for '<first-number>'")
    AssertErrorMessage(Qux.self, ["--number-two", "a", "1"], "The value 'a' is invalid for '--number-two <number-two>'")
  }
}

fileprivate struct Qwz: ParsableArguments {
  @Option() var name: String?
  @Option(name: [.customLong("title", withSingleDash: true)]) var title: String?
}

extension ErrorMessageTests {
  func testMispelledArgument_1() {
    AssertErrorMessage(Qwz.self, ["--nme"], "Unknown option '--nme'. Did you mean '--name'?")
    AssertErrorMessage(Qwz.self, ["-name"], "Unknown option '-name'. Did you mean '--name'?")
  }

  func testMispelledArgument_2() {
    AssertErrorMessage(Qwz.self, ["-ttle"], "Unknown option '-ttle'. Did you mean '-title'?")
    AssertErrorMessage(Qwz.self, ["--title"], "Unknown option '--title'. Did you mean '-title'?")
  }

  func testMispelledArgument_3() {
    AssertErrorMessage(Qwz.self, ["--not-similar"], "Unknown option '--not-similar'")
  }

  func testMispelledArgument_4() {
    AssertErrorMessage(Qwz.self, ["-x"], "Unknown option '-x'")
  }
}

private struct Options: ParsableArguments {
  enum OutputBehaviour: String, EnumerableFlag {
    case stats, count, list

    static func name(for value: OutputBehaviour) -> NameSpecification {
      .shortAndLong
    }
  }

  @Flag(help: "Program output")
  var behaviour: OutputBehaviour = .list

  @Flag(inversion: .prefixedNo, exclusivity: .exclusive) var bool: Bool
}

private struct OptOptions: ParsableArguments {
  enum OutputBehaviour: String, EnumerableFlag {
    case stats, count, list

    static func name(for value: OutputBehaviour) -> NameSpecification {
      .short
    }
  }

  @Flag(help: "Program output")
  var behaviour: OutputBehaviour?
}

extension ErrorMessageTests {
  func testDuplicateFlags() {
    AssertErrorMessage(Options.self, ["--list", "--bool", "-s"], "Value to be set with flag \'-s\' had already been set with flag \'--list\'")
    AssertErrorMessage(Options.self, ["-cbl"], "Value to be set with flag \'l\' in \'-cbl\' had already been set with flag \'c\' in \'-cbl\'")
    AssertErrorMessage(Options.self, ["-bc", "--stats", "-l"], "Value to be set with flag \'--stats\' had already been set with flag \'c\' in \'-bc\'")

    AssertErrorMessage(Options.self, ["--no-bool", "--bool"], "Value to be set with flag \'--bool\' had already been set with flag \'--no-bool\'")

    AssertErrorMessage(OptOptions.self, ["-cbl"], "Value to be set with flag \'l\' in \'-cbl\' had already been set with flag \'c\' in \'-cbl\'")
  }
}

// (see issue #434).
private struct EmptyArray: ParsableArguments {
  @Option(parsing: .upToNextOption)
  var array: [String] = []
  
  @Flag(name: [.short, .long])
  var verbose = false
}

extension ErrorMessageTests {
  func testEmptyArrayOption() {
    AssertErrorMessage(EmptyArray.self, ["--array"], "Missing value for '--array <array>'")
    
    AssertErrorMessage(EmptyArray.self, ["--array", "--verbose"], "Missing value for '--array <array>'")
    AssertErrorMessage(EmptyArray.self, ["-verbose", "--array"], "Missing value for '--array <array>'")
    
    AssertErrorMessage(EmptyArray.self, ["--array", "-v"], "Missing value for '--array <array>'")
    AssertErrorMessage(EmptyArray.self, ["-v", "--array"], "Missing value for '--array <array>'")
  }
}

// MARK: -

fileprivate struct Repeat: ParsableArguments {
  @Option() var count: Int?
  @Argument() var phrase: String
}

extension ErrorMessageTests {
  func testBadOptionBeforeArgument() {
    AssertErrorMessage(
      Repeat.self,
      ["--cont", "5", "Hello"],
      "Unknown option '--cont'. Did you mean '--count'?")
  }
}
