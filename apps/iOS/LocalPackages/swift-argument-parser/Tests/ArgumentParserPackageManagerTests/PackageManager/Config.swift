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

extension Package {
  /// Manipulate configuration of the package
  struct Config: ParsableCommand {}
}

extension Package.Config {
  public static var configuration = CommandConfiguration(
    subcommands: [GetMirror.self, SetMirror.self, UnsetMirror.self])
  
  /// Print mirror configuration for the given package dependency
  struct GetMirror: ParsableCommand {
    @OptionGroup()
    var options: Options
    
    @Option(name: .customLong("package-url"), help: "The package dependency URL")
    var packageURL: String
  }
  
  /// Set a mirror for a dependency
  struct SetMirror: ParsableCommand {
    @OptionGroup()
    var options: Options
    
    @Option(name: .customLong("mirror-url"), help: "The mirror URL")
    var mirrorURL: String
    
    @Option(name: .customLong("package-url"), help: "The package dependency URL")
    var packageURL: String
  }
  
  /// Remove an existing mirror
  struct UnsetMirror: ParsableCommand {
    @OptionGroup()
    var options: Options
    
    @Option(name: .customLong("mirror-url"), help: "The mirror URL")
    var mirrorURL: String
    
    @Option(name: .customLong("package-url"), help: "The package dependency URL")
    var packageURL: String
  }
}
