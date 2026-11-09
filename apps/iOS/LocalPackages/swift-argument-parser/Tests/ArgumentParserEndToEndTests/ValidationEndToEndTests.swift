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

final class ValidationEndToEndTests: XCTestCase {
}

fileprivate enum UserValidationError: LocalizedError {
  case userValidationError

  var errorDescription: String? {
    switch self {
    case .userValidationError:
      return "UserValidationError"
    }
  }
}

fileprivate struct Foo: ParsableArguments {
  static var usageString: String = """
    Usage: foo [--count <count>] [<names> ...] [--version] [--throw]
      See 'foo --help' for more information.
    """

  static var helpString: String = """
    USAGE: foo [--count <count>] [<names> ...] [--version] [--throw]

    ARGUMENTS:
      <names>

    OPTIONS:
      --count <count>
      --version
      --throw
      -h, --help              Show help information.
    """

  @Option()
  var count: Int?

  @Argument()
  var names: [String] = []

  @Flag
  var version: Bool = false

  @Flag(name: [.customLong("throw")])
  var throwCustomError: Bool = false

  @Flag(help: .hidden)
  var showUsageOnly: Bool = false

  @Flag(help: .hidden)
  var failValidationSilently: Bool = false

  @Flag(help: .hidden)
  var failSilently: Bool = false

  mutating func validate() throws {
    if version {
      throw CleanExit.message("0.0.1")
    }

    if names.isEmpty {
      throw ValidationError("Must specify at least one name.")
    }

    if let count = count, names.count != count {
      throw ValidationError("Number of names (\(names.count)) doesn't match count (\(count)).")
    }

    if throwCustomError {
      throw UserValidationError.userValidationError
    }

    if showUsageOnly {
      throw ValidationError("")
    }

    if failValidationSilently {
      throw ExitCode.validationFailure
    }

    if failSilently {
      throw ExitCode.failure
    }
  }
}

extension ValidationEndToEndTests {
  func testValidation() throws {
    AssertParse(Foo.self, ["Joe"]) { foo in
      XCTAssertEqual(foo.names, ["Joe"])
      XCTAssertNil(foo.count)
    }

    AssertParse(Foo.self, ["Joe", "Moe", "--count", "2"]) { foo in
      XCTAssertEqual(foo.names, ["Joe", "Moe"])
      XCTAssertEqual(foo.count, 2)
    }
  }

  func testValidation_Version() throws {
    AssertErrorMessage(Foo.self, ["--version"], "0.0.1")
    AssertFullErrorMessage(Foo.self, ["--version"], "0.0.1")
  }

  func testValidation_Fails() throws {
    AssertErrorMessage(Foo.self, [], "Must specify at least one name.")
    AssertFullErrorMessage(Foo.self, [], """
            Error: Must specify at least one name.

            \(Foo.helpString)

            """)

    AssertErrorMessage(Foo.self, ["--count", "3", "Joe"], """
            Number of names (1) doesn't match count (3).
            """)
    AssertFullErrorMessage(Foo.self, ["--count", "3", "Joe"], """
            Error: Number of names (1) doesn't match count (3).
            \(Foo.usageString)
            """)
  }

  func testCustomErrorValidation() {
    // verify that error description is printed if available via LocalizedError
    AssertErrorMessage(Foo.self, ["--throw", "Joe"], UserValidationError.userValidationError.errorDescription!)
  }

  func testEmptyErrorValidation() {
    AssertErrorMessage(Foo.self, ["--show-usage-only", "Joe"], "")
    AssertFullErrorMessage(Foo.self, ["--show-usage-only", "Joe"], Foo.usageString)
    AssertFullErrorMessage(Foo.self, ["--fail-validation-silently", "Joe"], "")
    AssertFullErrorMessage(Foo.self, ["--fail-silently", "Joe"], "")
  }
}

fileprivate struct FooCommand: ParsableCommand {
  @Flag(help: .hidden)
  var foo = false
  @Flag(help: .hidden)
  var bar = false

  mutating func validate() throws {
    if foo {
      // --foo implies --bar
      bar = true
    }
  }

  func run() throws {
    XCTAssertEqual(foo, bar)
  }
}

extension ValidationEndToEndTests {
  func testMutationsPreserved() throws {
    var foo = try FooCommand.parseAsRoot(["--foo"])
    try foo.run()
  }
}
