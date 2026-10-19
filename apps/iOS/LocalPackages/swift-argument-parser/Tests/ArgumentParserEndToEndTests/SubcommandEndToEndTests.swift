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

final class SubcommandEndToEndTests: XCTestCase {
}

// MARK: Single value String

fileprivate struct Foo: ParsableCommand {
  static let configuration =
    CommandConfiguration(subcommands: [CommandA.self, CommandB.self])

  @Option() var name: String
}

fileprivate struct CommandA: ParsableCommand {
  static let configuration = CommandConfiguration(commandName: "a")

  @OptionGroup() var foo: Foo

  @Option() var bar: Int
}

fileprivate struct CommandB: ParsableCommand {
  static let configuration = CommandConfiguration(commandName: "b")

  @OptionGroup() var foo: Foo

  @Option() var baz: String
}

extension SubcommandEndToEndTests {
  func testParsing_SubCommand() throws {
    AssertParseCommand(Foo.self, CommandA.self, ["--name", "Foo", "a", "--bar", "42"]) { a in
      XCTAssertEqual(a.bar, 42)
      XCTAssertEqual(a.foo.name, "Foo")
    }

    AssertParseCommand(Foo.self, CommandB.self, ["--name", "A", "b", "--baz", "abc"]) { b in
      XCTAssertEqual(b.baz, "abc")
      XCTAssertEqual(b.foo.name, "A")
    }
  }

  func testParsing_SubCommand_manual() throws {
    AssertParseCommand(Foo.self, CommandA.self, ["--name", "Foo", "a", "--bar", "42"]) { a in
      XCTAssertEqual(a.bar, 42)
      XCTAssertEqual(a.foo.name, "Foo")
    }

    AssertParseCommand(Foo.self, Foo.self, ["--name", "Foo"]) { foo in
      XCTAssertEqual(foo.name, "Foo")
    }
  }

  func testParsing_SubCommand_help() throws {
    let helpFoo = Foo.message(for: CleanExit.helpRequest())
    let helpA = Foo.message(for: CleanExit.helpRequest(CommandA.self))
    let helpB = Foo.message(for: CleanExit.helpRequest(CommandB.self))

    AssertEqualStrings(actual: helpFoo, expected: """
            USAGE: foo --name <name> <subcommand>

            OPTIONS:
              --name <name>
              -h, --help              Show help information.

            SUBCOMMANDS:
              a
              b

              See 'foo help <subcommand>' for detailed help.
            """)
    AssertEqualStrings(actual: helpA, expected: """
            USAGE: foo a --name <name> --bar <bar>

            OPTIONS:
              --name <name>
              --bar <bar>
              -h, --help              Show help information.

            """)
    AssertEqualStrings(actual: helpB, expected: """
            USAGE: foo b --name <name> --baz <baz>

            OPTIONS:
              --name <name>
              --baz <baz>
              -h, --help              Show help information.

            """)
  }


  func testParsing_SubCommand_fails() throws {
    XCTAssertThrowsError(try Foo.parse(["--name", "Foo", "a", "--baz", "42"]), "'baz' is not an option for the 'a' subcommand.")
    XCTAssertThrowsError(try Foo.parse(["--name", "Foo", "b", "--bar", "42"]), "'bar' is not an option for the 'b' subcommand.")
  }
}

fileprivate var mathDidRun = false

fileprivate struct Math: ParsableCommand {
  enum Operation: String, ExpressibleByArgument {
    case add
    case multiply
  }

  @Option(help: "The operation to perform")
  var operation: Operation = .add

  @Flag(name: [.short, .long])
  var verbose: Bool = false

  @Argument(help: "The first operand")
  var operands: [Int] = []

  mutating func run() {
    XCTAssertEqual(operation, .multiply)
    XCTAssertTrue(verbose)
    XCTAssertEqual(operands, [5, 11])
    mathDidRun = true
  }
}

extension SubcommandEndToEndTests {
  func testParsing_SingleCommand() throws {
    var mathCommand = try Math.parseAsRoot(["--operation", "multiply", "-v", "5", "11"])
    XCTAssertFalse(mathDidRun)
    try mathCommand.run()
    XCTAssertTrue(mathDidRun)
  }
}

// MARK: Nested Command Arguments Validated

struct BaseCommand: ParsableCommand {
  enum BaseCommandError: Error {
    case baseCommandFailure
    case subCommandFailure
  }

  static let baseFlagValue = "base"

  static let configuration = CommandConfiguration(
    commandName: "base",
    subcommands: [SubCommand.self]
  )

  @Option()
  var baseFlag: String

  mutating func validate() throws {
    guard baseFlag == BaseCommand.baseFlagValue else {
      throw BaseCommandError.baseCommandFailure
    }
  }
}

extension BaseCommand {
  struct SubCommand : ParsableCommand {
    static let subFlagValue = "sub"

    static let configuration = CommandConfiguration(
      commandName: "sub",
      subcommands: [SubSubCommand.self]
    )

    @Option()
    var subFlag: String

    mutating func validate() throws {
      guard subFlag == SubCommand.subFlagValue else {
        throw BaseCommandError.subCommandFailure
      }
    }
  }
}

extension BaseCommand.SubCommand {
  struct SubSubCommand : ParsableCommand, TestableParsableArguments {
    let didValidateExpectation = XCTestExpectation(singleExpectation: "did validate subcommand")

    static let configuration = CommandConfiguration(
      commandName: "subsub"
    )

    @Flag
    var subSubFlag: Bool = false

    private enum CodingKeys: CodingKey {
      case subSubFlag
    }
  }
}

extension SubcommandEndToEndTests {
  func testValidate_subcommands() {
    // provide a value to base-flag that will throw
    AssertErrorMessage(
      BaseCommand.self,
      ["--base-flag", "foo", "sub", "--sub-flag", "foo", "subsub"],
      "baseCommandFailure"
    )

    // provide a value to sub-flag that will throw
    AssertErrorMessage(
      BaseCommand.self,
      ["--base-flag", BaseCommand.baseFlagValue, "sub", "--sub-flag", "foo", "subsub"],
      "subCommandFailure"
    )

    // provide a valid command and make sure both validates succeed
    AssertParseCommand(BaseCommand.self,
                       BaseCommand.SubCommand.SubSubCommand.self,
                       ["--base-flag", BaseCommand.baseFlagValue, "sub", "--sub-flag", BaseCommand.SubCommand.subFlagValue, "subsub", "--sub-sub-flag"]) { cmd in
                        XCTAssertTrue(cmd.subSubFlag)

                        // make sure that the instance of SubSubCommand provided
                        // had its validate method called, not just that any instance of SubSubCommand was validated
                        wait(for: [cmd.didValidateExpectation], timeout: 0.1)
    }
  }
}

// MARK: Version flags

private struct A: ParsableCommand {
  static let configuration = CommandConfiguration(
    version: "1.0.0",
    subcommands: [HasVersionFlag.self, NoVersionFlag.self])

  struct HasVersionFlag: ParsableCommand {
    @Flag var version: Bool = false
  }

  struct NoVersionFlag: ParsableCommand {
    @Flag var hello: Bool = false
  }
}

extension SubcommandEndToEndTests {
  func testParsingVersionFlags() throws {
    AssertErrorMessage(A.self, ["--version"], "1.0.0")
    AssertErrorMessage(A.self, ["no-version-flag", "--version"], "1.0.0")

    AssertParseCommand(A.self, A.HasVersionFlag.self, ["has-version-flag", "--version"]) { cmd in
      XCTAssertTrue(cmd.version)
    }
  }
}
