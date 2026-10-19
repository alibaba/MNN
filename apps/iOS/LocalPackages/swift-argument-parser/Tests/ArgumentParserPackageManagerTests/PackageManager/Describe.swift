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
  /// Describe the current package
  struct Describe: ParsableCommand {
    @OptionGroup()
    var options: Options
    
    @Option(help: "Output format")
    var type: OutputType
    
    enum OutputType: String, ExpressibleByArgument, Decodable {
      case json
      case text
    }
  }
}
