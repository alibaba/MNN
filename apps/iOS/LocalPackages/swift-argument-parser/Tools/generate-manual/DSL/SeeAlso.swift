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

struct SeeAlso: MDocComponent {
  var section: Int
  var command: CommandInfoV0
  private var references: [String] {
    (command.subcommands ?? [])
      .map(\.manualPageTitle)
      .sorted()
  }

  var body: MDocComponent {
    Section(title: "see also") {
      ForEach(references) { reference, index in
        MDocMacro.CrossManualReference(title: reference, section: section)
          .withUnsafeChildren(
            nodes: index == references.count - 1 ? [] : [","])
      }
    }
  }
}
