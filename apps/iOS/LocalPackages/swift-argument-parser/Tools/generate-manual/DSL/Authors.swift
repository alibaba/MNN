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

struct Authors: MDocComponent {
  var authors: [AuthorArgument]

  var body: MDocComponent {
    Section(title: "authors") {
      if !authors.isEmpty {
        "The"
        MDocMacro.DocumentName()
        "reference was written by"
        ForEach(authors) { author, index in
          switch index {
          case authors.count - 2 where authors.count > 2:
            Author(
              author: author,
              trailing: ",")
            "and"
          case authors.count - 2:
            Author(
              author: author,
              trailing: "and")
          case authors.count - 1:
            Author(
              author: author,
              trailing: ".")
          default:
            Author(
              author: author,
              trailing: ",")
          }
        }
      }
    }
  }
}
