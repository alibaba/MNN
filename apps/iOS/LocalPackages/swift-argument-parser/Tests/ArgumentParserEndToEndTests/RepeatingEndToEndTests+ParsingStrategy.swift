//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2022 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import XCTest
import ArgumentParserTestHelpers
@testable import ArgumentParser

// MARK: - allUnrecognized

fileprivate struct AllUnrecognizedArgs: ParsableCommand {
  static var configuration: CommandConfiguration {
    .init(version: "1.0")
  }
  
  @Flag var verbose: Bool = false
  @Flag(name: .customShort("f")) var useFiles: Bool = false
  @Flag(name: .customShort("i")) var useStandardInput: Bool = false
  @Flag(name: .customShort("h")) var hoopla: Bool = false
  @Option var config = "debug"
  @Argument(parsing: .allUnrecognized) var names: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingAllUnrecognized() throws {
    AssertParse(AllUnrecognizedArgs.self, []) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertFalse(cmd.hoopla)
      XCTAssertEqual(cmd.names, [])
    }
    AssertParse(AllUnrecognizedArgs.self, ["foo", "--verbose", "-fi", "bar", "-z", "--other"]) { cmd in
      XCTAssertTrue(cmd.verbose)
      XCTAssertTrue(cmd.useFiles)
      XCTAssertTrue(cmd.useStandardInput)
      XCTAssertFalse(cmd.hoopla)
      XCTAssertEqual(cmd.names, ["foo", "bar", "-z", "--other"])
    }
  }
  
  func testParsing_repeatingAllUnrecognized_Builtin() throws {
    AssertParse(AllUnrecognizedArgs.self, ["foo", "--verbose", "bar", "-z", "-h"]) { cmd in
      XCTAssertTrue(cmd.verbose)
      XCTAssertFalse(cmd.useFiles)
      XCTAssertFalse(cmd.useStandardInput)
      XCTAssertTrue(cmd.hoopla)
      XCTAssertEqual(cmd.names, ["foo", "bar", "-z"])
    }

    AssertParseCommand(AllUnrecognizedArgs.self, HelpCommand.self, ["foo", "--verbose", "bar", "-z", "--help"]) { cmd in
      // No need to test HelpCommand properties
    }
    XCTAssertThrowsError(try AllUnrecognizedArgs.parse(["foo", "--verbose", "--version"]))
  }
  
  func testParsing_repeatingAllUnrecognized_Fails() throws {
    // Only partially matches the `-fib` argument
    XCTAssertThrowsError(try PassthroughArgs.parse(["-fib"]))
  }
}

fileprivate struct AllUnrecognizedRoot: ParsableCommand {
  static var configuration: CommandConfiguration {
    .init(subcommands: [Child.self])
  }
  
  @Flag var verbose: Bool = false
  
  struct Child: ParsableCommand {
    @Flag var includeExtras: Bool = false
    @Option var config = "debug"
    @Argument(parsing: .allUnrecognized) var extras: [String] = []
    @OptionGroup var root: AllUnrecognizedRoot
  }
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingAllUnrecognized_Nested() throws {
    AssertParseCommand(
      AllUnrecognizedRoot.self, AllUnrecognizedRoot.Child.self,
      ["child"])
    { cmd in
      XCTAssertFalse(cmd.root.verbose)
      XCTAssertFalse(cmd.includeExtras)
      XCTAssertEqual(cmd.config, "debug")
      XCTAssertEqual(cmd.extras, [])
    }
    AssertParseCommand(
      AllUnrecognizedRoot.self, AllUnrecognizedRoot.Child.self,
      ["child", "--verbose", "--other", "one", "two", "--config", "prod"])
    { cmd in
      XCTAssertTrue(cmd.root.verbose)
      XCTAssertFalse(cmd.includeExtras)
      XCTAssertEqual(cmd.config, "prod")
      XCTAssertEqual(cmd.extras, ["--other", "one", "two"])
    }
  }
  
  func testParsing_repeatingAllUnrecognized_Nested_Fails() throws {
    // Extra arguments need to make it to the child
    XCTAssertThrowsError(try AllUnrecognizedRoot.parse(["--verbose", "--other"]))
  }
}

// MARK: - postTerminator

fileprivate struct PostTerminatorArgs: ParsableArguments {
  @Flag(name: .customShort("f")) var useFiles: Bool = false
  @Flag(name: .customShort("i")) var useStandardInput: Bool = false
  @Option var config = "debug"
  @Argument var title: String?
  @Argument(parsing: .postTerminator)
  var names: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingPostTerminator() throws {
    AssertParse(PostTerminatorArgs.self, []) { cmd in
      XCTAssertNil(cmd.title)
      XCTAssertEqual(cmd.names, [])
    }
    AssertParse(PostTerminatorArgs.self, ["--", "-fi"]) { cmd in
      XCTAssertNil(cmd.title)
      XCTAssertEqual(cmd.names, ["-fi"])
    }
    AssertParse(PostTerminatorArgs.self, ["-fi", "--", "-fi", "--"]) { cmd in
      XCTAssertTrue(cmd.useFiles)
      XCTAssertTrue(cmd.useStandardInput)
      XCTAssertNil(cmd.title)
      XCTAssertEqual(cmd.names, ["-fi", "--"])
    }
    AssertParse(PostTerminatorArgs.self, ["-fi", "title", "--", "title"]) { cmd in
      XCTAssertTrue(cmd.useFiles)
      XCTAssertTrue(cmd.useStandardInput)
      XCTAssertEqual(cmd.title, "title")
      XCTAssertEqual(cmd.names, ["title"])
    }
    AssertParse(PostTerminatorArgs.self, ["--config", "config", "--", "--config", "post"]) { cmd in
      XCTAssertEqual(cmd.config, "config")
      XCTAssertNil(cmd.title)
      XCTAssertEqual(cmd.names, ["--config", "post"])
    }
  }

  func testParsing_repeatingPostTerminator_Fails() throws {
    // Only partially matches the `-fib` argument
    XCTAssertThrowsError(try PostTerminatorArgs.parse(["-fib"]))
    // The post-terminator input can't provide the option's value
    XCTAssertThrowsError(try PostTerminatorArgs.parse(["--config", "--", "config"]))
  }
}

// MARK: - captureForPassthrough

fileprivate struct PassthroughArgs: ParsableCommand {
  @Flag var verbose: Bool = false
  @Flag(name: .customShort("f")) var useFiles: Bool = false
  @Flag(name: .customShort("i")) var useStandardInput: Bool = false
  @Option var config = "debug"
  @Argument(parsing: .captureForPassthrough) var names: [String] = []
}

extension RepeatingEndToEndTests {
  func testParsing_repeatingCaptureForPassthrough() throws {
    AssertParse(PassthroughArgs.self, []) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertEqual(cmd.names, [])
    }

    AssertParse(PassthroughArgs.self, ["--other"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertEqual(cmd.names, ["--other"])
    }

    AssertParse(PassthroughArgs.self, ["--verbose", "one", "two", "three"]) { cmd in
      XCTAssertTrue(cmd.verbose)
      XCTAssertEqual(cmd.names, ["one", "two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["one", "two", "three", "--other", "--verbose"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertEqual(cmd.names, ["one", "two", "three", "--other", "--verbose"])
    }

    AssertParse(PassthroughArgs.self, ["--verbose", "--other", "one", "two", "three"]) { cmd in
      XCTAssertTrue(cmd.verbose)
      XCTAssertEqual(cmd.names, ["--other", "one", "two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["--verbose", "--other", "one", "--", "two", "three"]) { cmd in
      XCTAssertTrue(cmd.verbose)
      XCTAssertEqual(cmd.names, ["--other", "one", "--", "two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["--other", "one", "--", "two", "three", "--verbose"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertEqual(cmd.names, ["--other", "one", "--", "two", "three", "--verbose"])
    }

    AssertParse(PassthroughArgs.self, ["--", "--verbose", "--other", "one", "two", "three"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertEqual(cmd.names, ["--", "--verbose", "--other", "one", "two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["-one", "-two", "three"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertFalse(cmd.useFiles)
      XCTAssertFalse(cmd.useStandardInput)
      XCTAssertEqual(cmd.names, ["-one", "-two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["--config", "release", "one", "two", "--config", "debug"]) { cmd in
      XCTAssertEqual(cmd.config, "release")
      XCTAssertEqual(cmd.names, ["one", "two", "--config", "debug"])
    }

    AssertParse(PassthroughArgs.self, ["--config", "release", "--config", "debug", "one", "two"]) { cmd in
      XCTAssertEqual(cmd.config, "debug")
      XCTAssertEqual(cmd.names, ["one", "two"])
    }

    AssertParse(PassthroughArgs.self, ["-if", "-one", "-two", "three"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertTrue(cmd.useFiles)
      XCTAssertTrue(cmd.useStandardInput)
      XCTAssertEqual(cmd.names, ["-one", "-two", "three"])
    }

    AssertParse(PassthroughArgs.self, ["-one", "-two", "-three", "-if"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertFalse(cmd.useFiles)
      XCTAssertFalse(cmd.useStandardInput)
      XCTAssertEqual(cmd.names, ["-one", "-two", "-three", "-if"])
    }

    AssertParse(PassthroughArgs.self, ["-one", "-two", "-three", "-if", "--help"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertFalse(cmd.useFiles)
      XCTAssertFalse(cmd.useStandardInput)
      XCTAssertEqual(cmd.names, ["-one", "-two", "-three", "-if", "--help"])
    }

    AssertParse(PassthroughArgs.self, ["-one", "-two", "-three", "-if", "-h"]) { cmd in
      XCTAssertFalse(cmd.verbose)
      XCTAssertFalse(cmd.useFiles)
      XCTAssertFalse(cmd.useStandardInput)
      XCTAssertEqual(cmd.names, ["-one", "-two", "-three", "-if", "-h"])
    }
  }

  func testParsing_repeatingCaptureForPassthrough_Fails() throws {
    // Only partially matches the `-fib` argument
    XCTAssertThrowsError(try PassthroughArgs.parse(["-fib"]))
  }
}

