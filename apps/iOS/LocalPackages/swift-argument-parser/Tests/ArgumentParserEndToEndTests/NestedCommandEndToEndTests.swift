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

final class NestedCommandEndToEndTests: XCTestCase {
}

// MARK: Single value String

fileprivate struct Foo: ParsableCommand {
  static let configuration =
    CommandConfiguration(subcommands: [Build.self, Package.self])

  @Flag(name: .short)
  var verbose: Bool = false

  struct Build: ParsableCommand {
    @OptionGroup() var foo: Foo

    @Argument()
    var input: String
  }

  struct Package: ParsableCommand {
    static let configuration =
      CommandConfiguration(
        subcommands: [Clean.self, Config.self],
        aliases: ["pkg"])

    @Flag(name: .short)
    var force: Bool = false

    struct Clean: ParsableCommand {
      @OptionGroup() var foo: Foo
      @OptionGroup() var package: Package
    }

    struct Config: ParsableCommand {
      static let configuration = CommandConfiguration(aliases: ["cfg"])
      @OptionGroup() var foo: Foo
      @OptionGroup() var package: Package
    }
  }
}

fileprivate func AssertParseFooCommand<A>(_ type: A.Type, _ arguments: [String], file: StaticString = #file, line: UInt = #line, closure: (A) throws -> Void) where A: ParsableCommand {
  AssertParseCommand(Foo.self, type, arguments, file: file, line: line, closure: closure)
}


extension NestedCommandEndToEndTests {
  func testParsing_package() throws {
    AssertParseFooCommand(Foo.Package.self, ["package"]) { package in
      XCTAssertFalse(package.force)
    }

    AssertParseFooCommand(Foo.Package.self, ["pkg"]) { package in
      XCTAssertFalse(package.force)
    }

    AssertParseFooCommand(Foo.Package.Clean.self, ["package", "clean"]) { clean in
      XCTAssertEqual(clean.foo.verbose, false)
      XCTAssertEqual(clean.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Clean.self, ["pkg", "clean"]) { clean in
      XCTAssertEqual(clean.foo.verbose, false)
      XCTAssertEqual(clean.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Clean.self, ["package", "-f", "clean"]) { clean in
      XCTAssertEqual(clean.foo.verbose, false)
      XCTAssertEqual(clean.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Clean.self, ["pkg", "-f", "clean"]) { clean in
      XCTAssertEqual(clean.foo.verbose, false)
      XCTAssertEqual(clean.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-v", "config"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-v", "cfg"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "config", "-v"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "cfg", "-v"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["-v", "package", "config"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["-v", "pkg", "cfg"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, false)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-f", "config"]) { config in
      XCTAssertEqual(config.foo.verbose, false)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-f", "cfg"]) { config in
      XCTAssertEqual(config.foo.verbose, false)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "config", "-f"]) { config in
      XCTAssertEqual(config.foo.verbose, false)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "cfg", "-f"]) { config in
      XCTAssertEqual(config.foo.verbose, false)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-v", "config", "-f"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-v", "cfg", "-f"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-f", "config", "-v"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-f", "cfg", "-v"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-vf", "config"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-vf", "cfg"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["package", "-fv", "config"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }

    AssertParseFooCommand(Foo.Package.Config.self, ["pkg", "-fv", "cfg"]) { config in
      XCTAssertEqual(config.foo.verbose, true)
      XCTAssertEqual(config.package.force, true)
    }
  }

  func testParsing_build() throws {
    AssertParseFooCommand(Foo.Build.self, ["build", "file"]) { build in
      XCTAssertEqual(build.foo.verbose, false)
      XCTAssertEqual(build.input, "file")
    }
  }

  func testParsing_fails() throws {
    XCTAssertThrowsError(try Foo.parseAsRoot(["clean", "package"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["clean", "pkg"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["config", "package"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["cfg", "pkg"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "c"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "c"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "build", "clean"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "build", "clean"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "clean", "foo"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "clean", "foo"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "config", "bar"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "cfg", "bar"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["package", "clean", "build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["pkg", "clean", "build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["build", "-f"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["build", "--build"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["build", "--build", "12"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["-f", "package", "clean"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["-f", "pkg", "clean"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["-f", "package", "config"]))
    XCTAssertThrowsError(try Foo.parseAsRoot(["-f", "pkg", "config"]))
  }
}

private struct Options: ParsableArguments {
  @Option() var firstName: String?
}

private struct UniqueOptions: ParsableArguments {
  @Option() var lastName: String?
}

private struct Super: ParsableCommand {
  static var configuration: CommandConfiguration {
    .init(subcommands: [Sub1.self, Sub2.self])
  }

  @OptionGroup() var options: Options

  struct Sub1: ParsableCommand {
    @OptionGroup() var options: Options
  }

  struct Sub2: ParsableCommand {
    @OptionGroup() var options: UniqueOptions
  }
}

extension NestedCommandEndToEndTests {
  func testParsing_SharedOptions() throws {
    AssertParseCommand(Super.self, Super.self, []) { sup in
      XCTAssertNil(sup.options.firstName)
    }

    AssertParseCommand(Super.self, Super.self, ["--first-name", "Foo"]) { sup in
      XCTAssertEqual("Foo", sup.options.firstName)
    }

    AssertParseCommand(Super.self, Super.Sub1.self, ["sub1"]) { sub1 in
      XCTAssertNil(sub1.options.firstName)
    }

    AssertParseCommand(Super.self, Super.Sub1.self, ["sub1", "--first-name", "Foo"]) { sub1 in
      XCTAssertEqual("Foo", sub1.options.firstName)
    }

    AssertParseCommand(Super.self, Super.Sub2.self, ["sub2", "--last-name", "Foo"]) { sub2 in
      XCTAssertEqual("Foo", sub2.options.lastName)
    }
  }
}
