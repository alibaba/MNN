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

struct HelpCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "help",
    abstract: "Show subcommand help information.",
    helpNames: [])
  
  /// Any subcommand names provided after the `help` subcommand.
  @Argument var subcommands: [String] = []
  
  /// Capture and ignore any extra help flags given by the user.
  @Flag(name: [.short, .long, .customLong("help", withSingleDash: true)], help: .private)
  var help = false
  
  private(set) var commandStack: [ParsableCommand.Type] = []
  private(set) var visibility: ArgumentVisibility = .default

  init() {}
  
  mutating func run() throws {
    throw CommandError(
      commandStack: commandStack,
      parserError: .helpRequested(visibility: visibility))
  }
  
  mutating func buildCommandStack(with parser: CommandParser) throws {
    commandStack = parser.commandStack(for: subcommands)
  }

  /// Used for testing.
  func generateHelp(screenWidth: Int) -> String {
    HelpGenerator(
      commandStack: commandStack,
      visibility: visibility)
      .rendered(screenWidth: screenWidth)
  }
  
  enum CodingKeys: CodingKey {
    case subcommands
    case help
  }
  
  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.subcommands = try container.decode([String].self, forKey: .subcommands)
    self.help = try container.decode(Bool.self, forKey: .help)
  }
  
  init(commandStack: [ParsableCommand.Type], visibility: ArgumentVisibility) {
    self.commandStack = commandStack
    self.visibility = visibility
    self.subcommands = commandStack.map { $0._commandName }
    self.help = false
  }
}
