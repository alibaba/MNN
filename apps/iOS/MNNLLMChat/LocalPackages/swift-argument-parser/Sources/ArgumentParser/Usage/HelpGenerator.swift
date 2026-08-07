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

internal struct HelpGenerator {
  static let helpIndent = 2
  static let labelColumnWidth = 26
  static var systemScreenWidth: Int { Platform.terminalWidth }

  struct Section {
    struct Element: Hashable {
      var label: String
      var abstract: String = ""
      var discussion: String = ""
      
      var paddedLabel: String {
        String(repeating: " ", count: HelpGenerator.helpIndent) + label
      }
      
      func rendered(screenWidth: Int) -> String {
        let paddedLabel = self.paddedLabel
        let wrappedAbstract = self.abstract
          .wrapped(to: screenWidth, wrappingIndent: HelpGenerator.labelColumnWidth)
        let wrappedDiscussion = self.discussion.isEmpty
          ? ""
          : self.discussion.wrapped(to: screenWidth, wrappingIndent: HelpGenerator.helpIndent * 4) + "\n"
        let renderedAbstract: String = {
          guard !abstract.isEmpty else { return "" }
          if paddedLabel.count < HelpGenerator.labelColumnWidth {
            // Render after padded label.
            return String(wrappedAbstract.dropFirst(paddedLabel.count))
          } else {
            // Render in a new line.
            return "\n" + wrappedAbstract
          }
        }()
        return paddedLabel
          + renderedAbstract + "\n"
          + wrappedDiscussion
      }
    }
    
    enum Header: CustomStringConvertible, Equatable {
      case positionalArguments
      case subcommands
      case options
      case title(String)
      
      var description: String {
        switch self {
        case .positionalArguments:
          return "Arguments"
        case .subcommands:
          return "Subcommands"
        case .options:
          return "Options"
        case .title(let name):
          return name
        }
      }
    }
    
    var header: Header
    var elements: [Element]
    var discussion: String = ""
    var isSubcommands: Bool = false
    
    func rendered(screenWidth: Int) -> String {
      guard !elements.isEmpty else { return "" }
      
      let renderedElements = elements.map { $0.rendered(screenWidth: screenWidth) }.joined()
      return "\(String(describing: header).uppercased()):\n"
        + renderedElements
    }
  }
  
  struct DiscussionSection {
    var title: String = ""
    var content: String
  }
  
  var commandStack: [ParsableCommand.Type]
  var abstract: String
  var usage: String
  var sections: [Section]
  var discussionSections: [DiscussionSection]
  
  init(commandStack: [ParsableCommand.Type], visibility: ArgumentVisibility) {
    guard let currentCommand = commandStack.last else {
      fatalError()
    }
    
    let currentArgSet = ArgumentSet(currentCommand, visibility: visibility, parent: nil)
    self.commandStack = commandStack

    // Build the tool name and subcommand name from the command configuration
    var toolName = commandStack.map { $0._commandName }.joined(separator: " ")
    if let superName = commandStack.first!.configuration._superCommandName {
      toolName = "\(superName) \(toolName)"
    }

    if let usage = currentCommand.configuration.usage {
      self.usage = usage
    } else {
      var usage = UsageGenerator(toolName: toolName, definition: [currentArgSet])
        .synopsis
      if !currentCommand.configuration.subcommands.isEmpty {
        if usage.last != " " { usage += " " }
        usage += "<subcommand>"
      }
      self.usage = usage
    }
    
    self.abstract = currentCommand.configuration.abstract
    if !currentCommand.configuration.discussion.isEmpty {
      if !self.abstract.isEmpty {
        self.abstract += "\n"
      }
      self.abstract += "\n\(currentCommand.configuration.discussion)"
    }

    self.sections = HelpGenerator.generateSections(commandStack: commandStack, visibility: visibility)
    self.discussionSections = []
  }
  
  init(_ type: ParsableArguments.Type, visibility: ArgumentVisibility) {
    self.init(commandStack: [type.asCommand], visibility: visibility)
  }

  private static func generateSections(commandStack: [ParsableCommand.Type], visibility: ArgumentVisibility) -> [Section] {
    guard !commandStack.isEmpty else { return [] }
    
    var positionalElements: [Section.Element] = []
    var optionElements: [Section.Element] = []
    
    // Simulate an ordered dictionary using a dictionary and array for ordering.
    var titledSections: [String: [Section.Element]] = [:]
    var sectionTitles: [String] = []

    /// Start with a full slice of the ArgumentSet so we can peel off one or
    /// more elements at a time.
    var args = commandStack.argumentsForHelp(visibility: visibility)[...]
    while let arg = args.popFirst() {
      assert(arg.help.visibility.isAtLeastAsVisible(as: visibility))
      
      let synopsis: String
      let abstract: String

      let allValueStrings = arg.help.allValueStrings.filter { !$0.isEmpty }
      let defaultValue = arg.help.defaultValue ?? ""

      let allAndDefaultValues: String
      switch (!allValueStrings.isEmpty, !defaultValue.isEmpty) {
      case (false, false):
        allAndDefaultValues = ""
      case (true, false):
        allAndDefaultValues = "(values: \(allValueStrings.joined(separator: ", ")))"
      case (false, true):
        allAndDefaultValues = "(default: \(defaultValue))"
      case (true, true):
        allAndDefaultValues = "(values: \(allValueStrings.joined(separator: ", ")); default: \(defaultValue))"
      }

      if arg.help.isComposite {
        // If this argument is composite, we have a group of arguments to
        // output together.
        let groupEnd = args.firstIndex(where: { $0.help.keys != arg.help.keys }) ?? args.endIndex
        let groupedArgs = [arg] + args[..<groupEnd]
        args = args[groupEnd...]

        synopsis = groupedArgs
          .lazy
          .map { $0.synopsisForHelp }
          .joined(separator: "/")
        abstract = groupedArgs
          .lazy
          .map { $0.help.abstract }
          .first { !$0.isEmpty } ?? ""
      } else {
        synopsis = arg.synopsisForHelp
        abstract = arg.help.abstract
      }
      
      let description = [abstract, allAndDefaultValues]
        .lazy
        .filter { !$0.isEmpty }
        .joined(separator: " ")

      let element = Section.Element(
        label: synopsis,
        abstract: description,
        discussion: arg.help.discussion)
      switch (arg.kind, arg.help.parentTitle) {
      case (_, let sectionTitle) where !sectionTitle.isEmpty:
        if !titledSections.keys.contains(sectionTitle) {
          sectionTitles.append(sectionTitle)
        }
        titledSections[sectionTitle, default: []].append(element)
      case (.positional, _):
        positionalElements.append(element)
      default:
        optionElements.append(element)
      }
    }
    
    let configuration = commandStack.last!.configuration
    let subcommandElements: [Section.Element] =
      configuration.subcommands.compactMap { command in
        guard command.configuration.shouldDisplay else { return nil }
        var label = command._commandName
        for alias in command.configuration.aliases {
            label += ", \(alias)"
        }
        if command == configuration.defaultSubcommand {
            label += " (default)"
        }
        return Section.Element(
          label: label,
          abstract: command.configuration.abstract)
    }
    
    // Combine the compiled groups in this order:
    // - arguments
    // - named sections
    // - options/flags
    // - subcommands
    return [
      Section(header: .positionalArguments, elements: positionalElements),
    ] + sectionTitles.map { name in
      Section(header: .title(name), elements: titledSections[name, default: []])
    } + [
      Section(header: .options, elements: optionElements),
      Section(header: .subcommands, elements: subcommandElements),
    ]
  }
  
  func usageMessage() -> String {
    guard !usage.isEmpty else { return "" }
    return "Usage: \(usage.hangingIndentingEachLine(by: 7))"
  }
  
  var includesSubcommands: Bool {
    guard let subcommandSection = sections.first(where: { $0.header == .subcommands })
      else { return false }
    return !subcommandSection.elements.isEmpty
  }
  
  func rendered(screenWidth: Int? = nil) -> String {
    let screenWidth = screenWidth ?? HelpGenerator.systemScreenWidth
    let renderedSections = sections
      .map { $0.rendered(screenWidth: screenWidth) }
      .filter { !$0.isEmpty }
      .joined(separator: "\n")
    let renderedAbstract = abstract.isEmpty
      ? ""
      : "OVERVIEW: \(abstract)".wrapped(to: screenWidth) + "\n\n"
    
    var helpSubcommandMessage = ""
    if includesSubcommands {
      var names = commandStack.map { $0._commandName }
      if let superName = commandStack.first!.configuration._superCommandName {
        names.insert(superName, at: 0)
      }
      names.insert("help", at: 1)

      helpSubcommandMessage = """

          See '\(names.joined(separator: " ")) <subcommand>' for detailed help.
        """
    }
    
    let renderedUsage = usage.isEmpty
      ? ""
      : "USAGE: \(usage.hangingIndentingEachLine(by: 7))\n\n"
    
    return """
    \(renderedAbstract)\
    \(renderedUsage)\
    \(renderedSections)\(helpSubcommandMessage)
    """
  }
}

fileprivate extension CommandConfiguration {
  static var defaultHelpNames: NameSpecification { [.short, .long] }
}

fileprivate extension NameSpecification {
  /// Generates a list of `Name`s for the help command at any visibility level.
  ///
  /// If the `default` visibility is used, the help names are returned
  /// unmodified. If a non-default visibility is used the short names are
  /// removed and the long names (both single and double dash) are appended with
  /// the name of the visibility level. After the optional name modification
  /// step, the name are returned in descending order.
  func generateHelpNames(visibility: ArgumentVisibility) -> [Name] {
    self
      .makeNames(InputKey(name: "help", parent: nil))
      .compactMap { name in
        guard visibility.base != .default else { return name }
        switch name {
        case .long(let helpName):
          return .long("\(helpName)-\(visibility.base)")
        case .longWithSingleDash(let helpName):
          return .longWithSingleDash("\(helpName)-\(visibility)")
        case .short:
          // Cannot create a non-default help flag from a short name.
          return nil
        }
      }
      .sorted(by: >)
  }
}

internal extension BidirectionalCollection where Element == ParsableCommand.Type {
  /// Returns a list of help names at the request visibility level for the top
  /// most ParsableCommand in the command stack with custom helpNames. If the
  /// command stack contains no custom help names the default help names.
  func getHelpNames(visibility: ArgumentVisibility) -> [Name] {
    self.last(where: { $0.configuration.helpNames != nil })
      .map { $0.configuration.helpNames!.generateHelpNames(visibility: visibility) }
      ?? CommandConfiguration.defaultHelpNames.generateHelpNames(visibility: visibility)
  }

  func getPrimaryHelpName() -> Name? {
    getHelpNames(visibility: .default).preferredName
  }
  
  func versionArgumentDefinition() -> ArgumentDefinition? {
    guard contains(where: { !$0.configuration.version.isEmpty })
      else { return nil }
    return ArgumentDefinition(
      kind: .named([.long("version")]),
      help: .init(
        allValueStrings: [],
        options: [.isOptional],
        help: "Show the version.",
        defaultValue: nil,
        key: InputKey(name: "", parent: nil),
        isComposite: false),
      completion: .default,
      update: .nullary({ _, _, _ in })
    )
  }
  
  func helpArgumentDefinition() -> ArgumentDefinition? {
    let names = getHelpNames(visibility: .default)
    guard !names.isEmpty else { return nil }
    return ArgumentDefinition(
      kind: .named(names),
      help: .init(
        allValueStrings: [],
        options: [.isOptional],
        help: "Show help information.",
        defaultValue: nil,
        key: InputKey(name: "", parent: nil),
        isComposite: false),
      completion: .default,
      update: .nullary({ _, _, _ in })
    )
  }
  
  func dumpHelpArgumentDefinition() -> ArgumentDefinition {
    return ArgumentDefinition(
      kind: .named([.long("experimental-dump-help")]),
      help: .init(
        allValueStrings: [],
        options: [.isOptional],
        help: ArgumentHelp("Dump help information as JSON."),
        defaultValue: nil,
        key: InputKey(name: "", parent: nil),
        isComposite: false),
      completion: .default,
      update: .nullary({ _, _, _ in })
    )
  }
  
  /// Returns the ArgumentSet for the last command in this stack, including
  /// help and version flags, when appropriate.
  func argumentsForHelp(visibility: ArgumentVisibility) -> ArgumentSet {
    guard var arguments = self.last.map({ ArgumentSet($0, visibility: visibility, parent: nil) })
      else { return ArgumentSet() }
    self.versionArgumentDefinition().map { arguments.append($0) }
    self.helpArgumentDefinition().map { arguments.append($0) }
    
    // To add when 'dump-help' is public API:
    // arguments.append(self.dumpHelpArgumentDefinition())
    
    return arguments
  }
}
