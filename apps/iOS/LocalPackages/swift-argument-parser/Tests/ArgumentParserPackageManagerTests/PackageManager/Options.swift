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

import ArgumentParser

struct Options: ParsableArguments {
  @Option(help: "Specify build/cache directory")
  var buildPath: String = "./.build"

  enum Configuration: String, ExpressibleByArgument, Decodable {
    case debug
    case release
  }

  @Option(name: .shortAndLong,
          help: "Build with configuration")
  var configuration: Configuration = .debug

  @Flag(inversion: .prefixedEnableDisable,
        help: "Use automatic resolution if Package.resolved file is out-of-date")
  var automaticResolution: Bool = true

  @Flag(inversion: .prefixedEnableDisable,
        help: "Use indexing-while-building feature")
  var indexStore: Bool = true

  @Flag(inversion: .prefixedEnableDisable,
        help: "Cache Package.swift manifests")
  var packageManifestCaching: Bool = true

  @Flag(inversion: .prefixedEnableDisable)
  var prefetching: Bool = true

  @Flag(inversion: .prefixedEnableDisable,
        help: "Use sandbox when executing subprocesses")
  var sandbox: Bool = true

  @Flag(inversion: .prefixedEnableDisable,
        help: "[Experimental] Enable the new Pubgrub dependency resolver")
  var pubgrubResolver: Bool = false
  @Flag(inversion: .prefixedNo,
        help: "Link Swift stdlib statically")
  var staticSwiftStdlib: Bool = false
  @Option(help: "Change working directory before any other operation")
  var packagePath: String = "."

  @Flag(help: "Turn on runtime checks for erroneous behavior")
  var sanitize: Bool = false

  @Flag(help: "Skip updating dependencies from their remote during a resolution")
  var skipUpdate: Bool = false

  @Flag(name: .shortAndLong,
        help: "Increase verbosity of informational output")
  var verbose: Bool = false

  @Option(name: .customLong("Xcc", withSingleDash: true),
          parsing: .unconditionalSingleValue,
          help: ArgumentHelp("Pass flag through to all C compiler invocations",
                             valueName: "c-compiler-flag"))
  var cCompilerFlags: [String] = []

  @Option(name: .customLong("Xcxx", withSingleDash: true),
          parsing: .unconditionalSingleValue,
          help: ArgumentHelp("Pass flag through to all C++ compiler invocations",
                             valueName: "cxx-compiler-flag"))
  var cxxCompilerFlags: [String] = []

  @Option(name: .customLong("Xlinker", withSingleDash: true),
          parsing: .unconditionalSingleValue,
          help: ArgumentHelp("Pass flag through to all linker invocations",
                             valueName: "linker-flag"))
  var linkerFlags: [String] = []

  @Option(name: .customLong("Xswiftc", withSingleDash: true),
          parsing: .unconditionalSingleValue,
          help: ArgumentHelp("Pass flag through to all Swift compiler invocations",
                             valueName: "swift-compiler-flag"))
  var swiftCompilerFlags: [String] = []
}

struct Package: ParsableCommand {
  static let configuration = CommandConfiguration(
    subcommands: [Clean.self, Config.self, Describe.self, GenerateXcodeProject.self, Hidden.self])
}

extension Package {
  struct Hidden: ParsableCommand {
    static let configuration = CommandConfiguration(shouldDisplay: false)
  }
}
