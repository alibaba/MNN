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
@testable import ArgumentParser
import ArgumentParserTestHelpers

final class HelpTests: XCTestCase {
  override func setUp() {
    #if !os(Windows) && !os(WASI)
    unsetenv("COLUMNS")
    #endif
  }
}

func getErrorText<T: ParsableArguments>(_: T.Type, _ arguments: [String]) -> String {
  do {
    _ = try T.parse(arguments)
    XCTFail("Didn't generate a help error")
    return ""
  } catch {
    return T.message(for: error)
  }
}

func getErrorText<T: ParsableCommand>(_: T.Type, _ arguments: [String], screenWidth: Int) -> String {
  do {
    let command = try T.parseAsRoot(arguments)
    if let helpCommand = command as? HelpCommand {
      return helpCommand.generateHelp(screenWidth: screenWidth)
    } else {
      XCTFail("Didn't generate a help error")
      return ""
    }
  } catch {
    return T.message(for: error)
  }
}

extension HelpTests {
  func testGlobalHelp() throws {
    XCTAssertEqual(
      getErrorText(Package.self, ["help"]).trimmingLines(),
      """
                USAGE: package <subcommand>

                OPTIONS:
                  -h, --help              Show help information.

                SUBCOMMANDS:
                  clean
                  config
                  describe
                  generate-xcodeproj

                  See 'package help <subcommand>' for detailed help.
                """.trimmingLines())
  }

  func testGlobalHelp_messageForCleanExit_helpRequest() throws {
    XCTAssertEqual(
      Package.message(for: CleanExit.helpRequest()).trimmingLines(),
      """
                USAGE: package <subcommand>

                OPTIONS:
                  -h, --help              Show help information.

                SUBCOMMANDS:
                  clean
                  config
                  describe
                  generate-xcodeproj

                  See 'package help <subcommand>' for detailed help.
                """.trimmingLines()
    )
  }

  func testGlobalHelp_messageForCleanExit_message() throws {
    let expectedMessage = "Failure"
    XCTAssertEqual(
      Package.message(for: CleanExit.message(expectedMessage)).trimmingLines(),
      expectedMessage
    )
  }

  func testConfigHelp() throws {
    XCTAssertEqual(
      getErrorText(Package.self, ["help", "config"], screenWidth: 80).trimmingLines(),
      """
                USAGE: package config <subcommand>

                OPTIONS:
                  -h, --help              Show help information.

                SUBCOMMANDS:
                  get-mirror
                  set-mirror
                  unset-mirror

                  See 'package help config <subcommand>' for detailed help.
                """.trimmingLines())
  }

  func testGetMirrorHelp() throws {
    XCTAssertEqual(
      getErrorText(Package.self, ["help", "config",  "get-mirror"], screenWidth: 80).trimmingLines(),
      """
                USAGE: package config get-mirror [<options>] --package-url <package-url>

                OPTIONS:
                  --build-path <build-path>
                                          Specify build/cache directory (default: ./.build)
                  -c, --configuration <configuration>
                                          Build with configuration (default: debug)
                  --enable-automatic-resolution/--disable-automatic-resolution
                                          Use automatic resolution if Package.resolved file is
                                          out-of-date (default: --enable-automatic-resolution)
                  --enable-index-store/--disable-index-store
                                          Use indexing-while-building feature (default:
                                          --enable-index-store)
                  --enable-package-manifest-caching/--disable-package-manifest-caching
                                          Cache Package.swift manifests (default:
                                          --enable-package-manifest-caching)
                  --enable-prefetching/--disable-prefetching
                                          (default: --enable-prefetching)
                  --enable-sandbox/--disable-sandbox
                                          Use sandbox when executing subprocesses (default:
                                          --enable-sandbox)
                  --enable-pubgrub-resolver/--disable-pubgrub-resolver
                                          [Experimental] Enable the new Pubgrub dependency
                                          resolver (default: --disable-pubgrub-resolver)
                  --static-swift-stdlib/--no-static-swift-stdlib
                                          Link Swift stdlib statically (default:
                                          --no-static-swift-stdlib)
                  --package-path <package-path>
                                          Change working directory before any other operation
                                          (default: .)
                  --sanitize              Turn on runtime checks for erroneous behavior
                  --skip-update           Skip updating dependencies from their remote during a
                                          resolution
                  -v, --verbose           Increase verbosity of informational output
                  -Xcc <c-compiler-flag>  Pass flag through to all C compiler invocations
                  -Xcxx <cxx-compiler-flag>
                                          Pass flag through to all C++ compiler invocations
                  -Xlinker <linker-flag>  Pass flag through to all linker invocations
                  -Xswiftc <swift-compiler-flag>
                                          Pass flag through to all Swift compiler invocations
                  --package-url <package-url>
                                          The package dependency URL
                  -h, --help              Show help information.

                """.trimmingLines())
  }
}

struct Simple: ParsableArguments {
  @Flag var verbose: Bool = false
  @Option() var min: Int?
  @Argument() var max: Int

  static var helpText = """
        USAGE: simple [--verbose] [--min <min>] <max>

        ARGUMENTS:
          <max>

        OPTIONS:
          --verbose
          --min <min>
          -h, --help              Show help information.

        """.trimmingLines()
}

extension HelpTests {
  func testSimpleHelp() throws {
    XCTAssertEqual(
      getErrorText(Simple.self, ["--help"]).trimmingLines(),
      Simple.helpText)
    XCTAssertEqual(
      getErrorText(Simple.self, ["-h"]).trimmingLines(),
      Simple.helpText)
  }
}

struct CustomHelp: ParsableCommand {
  static let configuration = CommandConfiguration(
    helpNames: [.customShort("?"), .customLong("show-help")]
  )
}

extension HelpTests {
  func testCustomHelpNames() {
    let helpNames = [CustomHelp.self].getHelpNames(visibility: .default)
    XCTAssertEqual(helpNames, [.short("?"), .long("show-help")])
    let helpHiddenNames = [CustomHelp.self].getHelpNames(visibility: .hidden)
    XCTAssertEqual(helpHiddenNames, [.long("show-help-hidden")])

    AssertFullErrorMessage(CustomHelp.self, ["--error"], """
      Error: Unknown option '--error'
      Usage: custom-help
        See 'custom-help --show-help' for more information.
      """)
  }
}

struct NoHelp: ParsableCommand {
  static let configuration = CommandConfiguration(
    helpNames: []
  )

  @Option(help: "How many florps?") var count: Int
}

extension HelpTests {
  func testNoHelpNames() {
    let helpNames = [NoHelp.self].getHelpNames(visibility: .default)
    XCTAssertEqual(helpNames, [])
    let helpHiddenNames = [NoHelp.self].getHelpNames(visibility: .hidden)
    XCTAssertEqual(helpHiddenNames, [])

    AssertFullErrorMessage(NoHelp.self, ["--error"], """
      Error: Missing expected argument '--count <count>'
      Help:  --count <count>  How many florps?
      Usage: no-help --count <count>
      """)

    XCTAssertEqual(
      NoHelp.message(for: CleanExit.helpRequest()).trimmingLines(),
      """
            USAGE: no-help --count <count>

            OPTIONS:
              --count <count>         How many florps?

            """)
  }
}

struct SubCommandCustomHelp: ParsableCommand {
  static let configuration = CommandConfiguration (
    helpNames: [.customShort("p"), .customLong("parent-help")]
  )

  struct InheritHelp: ParsableCommand {

  }

  struct ModifiedHelp: ParsableCommand {
    static let configuration = CommandConfiguration (
      helpNames: [.customShort("s"), .customLong("subcommand-help")]
    )

    struct InheritImmediateParentdHelp: ParsableCommand {

    }
  }
}

extension HelpTests {
  func testSubCommandInheritHelpNames() {
    let names = [
      SubCommandCustomHelp.self,
      SubCommandCustomHelp.InheritHelp.self,
    ].getHelpNames(visibility: .default)
    XCTAssertEqual(names, [.short("p"), .long("parent-help")])
  }

  func testSubCommandCustomHelpNames() {
    let names = [
      SubCommandCustomHelp.self,
      SubCommandCustomHelp.ModifiedHelp.self
    ].getHelpNames(visibility: .default)
    XCTAssertEqual(names, [.short("s"), .long("subcommand-help")])
  }

  func testInheritImmediateParentHelpNames() {
    let names = [
      SubCommandCustomHelp.self,
      SubCommandCustomHelp.ModifiedHelp.self,
      SubCommandCustomHelp.ModifiedHelp.InheritImmediateParentdHelp.self
    ].getHelpNames(visibility: .default)
    XCTAssertEqual(names, [.short("s"), .long("subcommand-help")])
  }
}
