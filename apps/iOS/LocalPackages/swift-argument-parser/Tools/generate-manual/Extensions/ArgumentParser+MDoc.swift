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

extension CommandInfoV0 {
  func manualPageFileName(section: Int) -> String {
    manualPageTitle + ".\(section)"
  }

  var manualPageDocumentTitle: String {
    let parts = (superCommands ?? []) + [commandName]
    return parts.joined(separator: ".").uppercased()
  }

  var manualPageTitle: String {
    let parts = (superCommands ?? []) + [commandName]
    return parts.joined(separator: ".")
  }

  var manualPageName: String {
    let parts = (superCommands ?? []) + [commandName]
    return parts.joined(separator: " ")
  }
}

extension ArgumentInfoV0 {
  // ArgumentInfoV0 value name as MDoc with "..." appended if the argument is
  // repeating.
  var manualPageValueName: MDocASTNode {
    var valueName = valueName ?? ""
    if isRepeating {
      valueName += "..."
    }
    // FIXME: MDocMacro.Emphasis?
    return MDocMacro.CommandArgument(arguments: [valueName])
  }

  // ArgumentDefinition formatted as MDoc for use in a description section.
  var manualPageDescription: MDocASTNode {
    // names.partitioned.map(\.manualPage).interspersed(with: ",")
    var synopses = (names ?? []).partitioned
      .flatMap { [$0.manualPage, ","] }
    synopses = synopses.dropLast()

    switch kind {
    case .positional:
      return manualPageValueName
    case .option:
      return MDocMacro.CommandOption(options: synopses)
        .withUnsafeChildren(nodes: [manualPageValueName])
    case .flag:
      return MDocMacro.CommandOption(options: synopses)
    }
  }
}

extension ArgumentInfoV0.NameInfoV0 {
  // Name formatted as MDoc.
  var manualPage: MDocASTNode {
    switch kind {
    case .long:
      return "-\(name)"
    case .short:
      return name
    case .longWithSingleDash:
      return name
    }
  }
}

extension Array where Element == ParsableCommand.Type {
  var commandNames: [String] {
    var commandNames = [String]()
    if let superName = first?.configuration._superCommandName {
      commandNames.append(superName)
    }
    commandNames.append(contentsOf: map { $0._commandName })
    return commandNames
  }
}

extension BidirectionalCollection where Element == ArgumentInfoV0.NameInfoV0 {
  var preferredName: Element? {
    first { $0.kind != .short } ?? first
  }

  var partitioned: [Element] {
    filter { $0.kind == .short } + filter { $0.kind != .short }
  }
}
