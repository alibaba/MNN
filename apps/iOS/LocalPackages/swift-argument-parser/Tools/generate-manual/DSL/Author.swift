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

struct Author: MDocComponent {
  var author: AuthorArgument
  var trailing: String

  var body: MDocComponent {
    switch author {
    case let .name(name):
      MDocMacro.Author(split: false)
      MDocMacro.Author(name: name)
        .withUnsafeChildren(nodes: [trailing])
    case let .email(email):
      MDocMacro.MailTo(email: email)
        .withUnsafeChildren(nodes: [trailing])
    case let .both(name, email):
      MDocMacro.Author(split: false)
      MDocMacro.Author(name: name)
      MDocMacro.BeginAngleBrackets()
      MDocMacro.MailTo(email: email)
      MDocMacro.EndAngleBrackets()
        .withUnsafeChildren(nodes: [trailing])
    }
  }
}
