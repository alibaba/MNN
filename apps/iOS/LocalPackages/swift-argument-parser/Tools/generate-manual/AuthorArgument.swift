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

fileprivate extension Character {
  static let emailStart: Character = "<"
  static let emailEnd: Character = ">"
}

fileprivate extension Substring {
  mutating func collecting(until terminator: (Element) throws -> Bool) rethrows -> String {
    let terminatorIndex = try firstIndex(where: terminator) ?? endIndex
    let collected = String(self[..<terminatorIndex])
    self = self[terminatorIndex...]
    return collected
  }

  mutating func next() {
    if !isEmpty { removeFirst() }
  }
}

enum AuthorArgument {
  case name(name: String)
  case email(email: String)
  case both(name: String, email: String)
}

extension AuthorArgument: ExpressibleByArgument {
  // parsed as:
  // - name: `name`
  // - email: `<email>`
  // - both: `name<email>`
  public init?(argument: String) {
    var argument = argument[...]
    // collect until the email start character is seen.
    let name = argument.collecting(until: { $0 == .emailStart })
    // drop the email start character.
    argument.next()
    // collect until the email end character is seen.
    let email = argument.collecting(until: { $0 == .emailEnd })
    // drop the email end character.
    argument.next()
    // ensure no collected characters remain.
    guard argument.isEmpty else { return nil }

    switch (name.isEmpty, email.isEmpty) {
    case (true, true):
      return nil
    case (false, true):
      self = .name(name: name)
    case (true, false):
      self = .email(email: email)
    case (false, false):
      self = .both(name: name, email: email)
    }
  }
}
