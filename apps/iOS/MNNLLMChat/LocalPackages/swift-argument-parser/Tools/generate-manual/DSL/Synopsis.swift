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

struct Synopsis: MDocComponent {
  var command: CommandInfoV0

  var body: MDocComponent {
    Section(title: "synopsis") {
      MDocMacro.DocumentName()

      if command.subcommands != nil {
        if command.defaultSubcommand != nil {
          MDocMacro.BeginOptionalCommandLineComponent()
        }
        MDocMacro.CommandArgument(arguments: ["subcommand"])
        if command.defaultSubcommand != nil {
          MDocMacro.EndOptionalCommandLineComponent()
        }
      }
      for argument in command.arguments ?? [] {
        ArgumentSynopsis(argument: argument)
      }
    }
  }
}
