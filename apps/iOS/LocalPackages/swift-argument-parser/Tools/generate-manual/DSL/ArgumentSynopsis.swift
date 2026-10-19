//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import ArgumentParser
import ArgumentParserToolInfo

struct ArgumentSynopsis: MDocComponent {
  var argument: ArgumentInfoV0

  var body: MDocComponent {
    if argument.isOptional {
      MDocMacro.OptionalCommandLineComponent(arguments: [synopsis])
    } else {
      synopsis
    }
  }

  // ArgumentInfoV0 formatted as MDoc without optional bracket wrapper.
  var synopsis: MDocASTNode {
    switch argument.kind {
    case .positional:
      return argument.manualPageDescription
    case .option:
      // preferredName cannot be nil
      let name = argument.preferredName!
      return MDocMacro.CommandOption(options: [name.manualPage])
        .withUnsafeChildren(nodes: [argument.manualPageValueName])
    case .flag:
      // preferredName cannot be nil
      let name = argument.preferredName!
      return MDocMacro.CommandOption(options: [name.manualPage])
    }
  }
}
