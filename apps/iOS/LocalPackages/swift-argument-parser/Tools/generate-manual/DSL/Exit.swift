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

struct Exit: MDocComponent {
  var section: Int

  var body: MDocComponent {
    Section(title: "exit status") {
      if [1, 6, 8].contains(section) {
        MDocMacro.ExitStandard()
      }
    }
  }
}
