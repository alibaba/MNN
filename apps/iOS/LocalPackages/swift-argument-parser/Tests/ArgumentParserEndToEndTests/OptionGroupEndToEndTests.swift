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

final class OptionGroupEndToEndTests: XCTestCase {
}

fileprivate struct Inner: TestableParsableArguments {
  @Flag(name: [.short, .long])
  var extraVerbiage: Bool = false
  @Option
  var size: Int = 0
  @Argument()
  var name: String

  let didValidateExpectation = XCTestExpectation(singleExpectation: "inner validated")

  private enum CodingKeys: CodingKey {
    case extraVerbiage
    case size
    case name
  }
}

fileprivate struct Outer: TestableParsableArguments {
  @Flag
  var verbose: Bool = false
  @Argument()
  var before: String
  @OptionGroup()
  var inner: Inner
  @Argument()
  var after: String

  let didValidateExpectation = XCTestExpectation(singleExpectation: "outer validated")

  private enum CodingKeys: CodingKey {
    case verbose
    case before
    case inner
    case after
  }
}

fileprivate struct Command: TestableParsableCommand {
  static let configuration = CommandConfiguration(commandName: "testCommand")

  @OptionGroup()
  var outer: Outer

  let didValidateExpectation = XCTestExpectation(singleExpectation: "Command validated")
  let didRunExpectation = XCTestExpectation(singleExpectation: "Command ran")

  private enum CodingKeys: CodingKey {
    case outer
  }
}

extension OptionGroupEndToEndTests {
  func testOptionGroup_Defaults() throws {
    AssertParse(Outer.self, ["prefix", "name", "postfix"]) { options in
      XCTAssertEqual(options.verbose, false)
      XCTAssertEqual(options.before, "prefix")
      XCTAssertEqual(options.after, "postfix")

      XCTAssertEqual(options.inner.extraVerbiage, false)
      XCTAssertEqual(options.inner.size, 0)
      XCTAssertEqual(options.inner.name, "name")
    }

    AssertParse(Outer.self, ["prefix", "--extra-verbiage", "name", "postfix", "--verbose", "--size", "5"]) { options in
      XCTAssertEqual(options.verbose, true)
      XCTAssertEqual(options.before, "prefix")
      XCTAssertEqual(options.after, "postfix")

      XCTAssertEqual(options.inner.extraVerbiage, true)
      XCTAssertEqual(options.inner.size, 5)
      XCTAssertEqual(options.inner.name, "name")
    }
  }

  func testOptionGroup_isValidated() {
    // Parse the command, this should cause validation to be once each on
    // - command.outer.inner
    // - command.outer
    // - command
    AssertParseCommand(Command.self, Command.self, ["prefix", "name", "postfix"]) { command in
      wait(for: [command.didValidateExpectation, command.outer.didValidateExpectation, command.outer.inner.didValidateExpectation], timeout: 0.1)
    }
  }

  func testOptionGroup_Fails() throws {
    XCTAssertThrowsError(try Outer.parse([]))
    XCTAssertThrowsError(try Outer.parse(["prefix"]))
    XCTAssertThrowsError(try Outer.parse(["prefix", "name"]))
    XCTAssertThrowsError(try Outer.parse(["prefix", "name", "postfix", "extra"]))
    XCTAssertThrowsError(try Outer.parse(["prefix", "name", "postfix", "--size", "a"]))
  }
}

fileprivate struct DuplicatedFlagGroupCustom: ParsableArguments {
  @Flag(name: .customLong("duplicated-option"))
  var duplicated: Bool = false
}

fileprivate struct DuplicatedFlagGroupCustomCommand: ParsableCommand {
  @Flag var duplicated: Bool = false
  @OptionGroup var option: DuplicatedFlagGroupCustom
}

fileprivate struct DuplicatedFlagGroupLong: ParsableArguments {
  @Flag var duplicated: Bool = false
}

fileprivate struct DuplicatedFlagGroupLongCommand: ParsableCommand {
  @Flag(name: .customLong("duplicated-option"))
  var duplicated: Bool = false
  @OptionGroup var option: DuplicatedFlagGroupLong
}

extension OptionGroupEndToEndTests {
  func testUniqueNamesForDuplicatedFlag_NoFlags() throws {
    AssertParse(DuplicatedFlagGroupCustomCommand.self, []) { command in
      XCTAssertFalse(command.duplicated)
      XCTAssertFalse(command.option.duplicated)
    }
    AssertParse(DuplicatedFlagGroupLongCommand.self, []) { command in
      XCTAssertFalse(command.duplicated)
      XCTAssertFalse(command.option.duplicated)
    }
  }
  
  func testUniqueNamesForDuplicatedFlag_RootOnly() throws {
    AssertParse(DuplicatedFlagGroupCustomCommand.self, ["--duplicated"]) { command in
      XCTAssertTrue(command.duplicated)
      XCTAssertFalse(command.option.duplicated)
    }
    AssertParse(DuplicatedFlagGroupLongCommand.self, ["--duplicated"]) { command in
      XCTAssertFalse(command.duplicated)
      XCTAssertTrue(command.option.duplicated)
    }
  }
  
  func testUniqueNamesForDuplicatedFlag_OptionOnly() throws {
    AssertParse(DuplicatedFlagGroupCustomCommand.self, ["--duplicated-option"]) { command in
      XCTAssertFalse(command.duplicated)
      XCTAssertTrue(command.option.duplicated)
    }
    AssertParse(DuplicatedFlagGroupLongCommand.self, ["--duplicated-option"]) { command in
      XCTAssertTrue(command.duplicated)
      XCTAssertFalse(command.option.duplicated)
    }
  }
  
  func testUniqueNamesForDuplicatedFlag_RootAndOption() throws {
    AssertParse(DuplicatedFlagGroupCustomCommand.self, ["--duplicated", "--duplicated-option"]) { command in
      XCTAssertTrue(command.duplicated)
      XCTAssertTrue(command.option.duplicated)
    }
    AssertParse(DuplicatedFlagGroupLongCommand.self, ["--duplicated", "--duplicated-option"]) { command in
      XCTAssertTrue(command.duplicated)
      XCTAssertTrue(command.option.duplicated)
    }
  }
}

