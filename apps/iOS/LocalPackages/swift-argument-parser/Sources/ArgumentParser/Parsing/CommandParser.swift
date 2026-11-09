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

struct CommandError: Error {
  var commandStack: [ParsableCommand.Type]
  var parserError: ParserError
}

struct HelpRequested: Error {
  var visibility: ArgumentVisibility
}

struct CommandParser {
  let commandTree: Tree<ParsableCommand.Type>
  var currentNode: Tree<ParsableCommand.Type>
  var decodedArguments: [DecodedArguments] = []
  
  var rootCommand: ParsableCommand.Type {
    commandTree.element
  }
  
  var commandStack: [ParsableCommand.Type] {
    let result = decodedArguments.compactMap { $0.commandType }
    if currentNode.element == result.last {
      return result
    } else {
      return result + [currentNode.element]
    }
  }
  
  init(_ rootCommand: ParsableCommand.Type) {
    do {
      self.commandTree = try Tree(root: rootCommand)
    } catch Tree<ParsableCommand.Type>.InitializationError.recursiveSubcommand(let command) {
      fatalError("The ParsableCommand \"\(command)\" can't have itself as its own subcommand.")
    } catch Tree<ParsableCommand.Type>.InitializationError.aliasMatchingCommand(let command) {
      fatalError("The ParsableCommand \"\(command)\" can't have an alias with the same name as the command itself.")
    } catch {
      fatalError("Unexpected error: \(error).")
    }
    self.currentNode = commandTree

    // A command tree that has a depth greater than zero gets a `help`
    // subcommand.
    if !commandTree.isLeaf {
      commandTree.addChild(Tree(HelpCommand.self))
    }
  }
}

extension CommandParser {
  /// Consumes the next argument in `split` if it matches a subcommand at the
  /// current node of the command tree.
  ///
  /// If a matching subcommand is found, the subcommand argument is consumed
  /// in `split`.
  ///
  /// - Returns: A node for the matched subcommand if one was found;
  ///   otherwise, `nil`.
  fileprivate func consumeNextCommand(split: inout SplitArguments) -> Tree<ParsableCommand.Type>? {
    guard let (origin, element) = split.peekNext(),
      element.isValue,
      let value = split.originalInput(at: origin),
      let subcommandNode = currentNode.firstChild(withName: value)
    else { return nil }
    _ = split.popNextValue()
    return subcommandNode
  }
  
  /// Throws a `HelpRequested` error if the user has specified any of the
  /// built-in flags.
  ///
  /// - Parameters:
  ///   - split: The remaining arguments to examine.
  ///   - requireSoloArgument: `true` if the built-in flag must be the only
  ///     one remaining for this to catch it.
  func checkForBuiltInFlags(
    _ split: SplitArguments,
    requireSoloArgument: Bool = false
  ) throws {
    guard !requireSoloArgument || split.originalInput.count == 1 else { return }
    
    // Look for help flags
    guard !split.contains(anyOf: self.commandStack.getHelpNames(visibility: .default)) else {
      throw HelpRequested(visibility: .default)
    }

    // Look for help-hidden flags
    guard !split.contains(anyOf: self.commandStack.getHelpNames(visibility: .hidden)) else {
      throw HelpRequested(visibility: .hidden)
    }

    // Look for dump-help flag
    guard !split.contains(Name.long("experimental-dump-help")) else {
      throw CommandError(commandStack: commandStack, parserError: .dumpHelpRequested)
    }

    // Look for a version flag if any commands in the stack define a version
    if commandStack.contains(where: { !$0.configuration.version.isEmpty }) {
      guard !split.contains(Name.long("version")) else {
        throw CommandError(commandStack: commandStack, parserError: .versionRequested)
      }
    }
  }
  
  /// Returns the last parsed value if there are no remaining unused arguments.
  ///
  /// If there are remaining arguments or if no commands have been parsed,
  /// this throws an error.
  fileprivate func extractLastParsedValue(_ split: SplitArguments) throws -> ParsableCommand {
    try checkForBuiltInFlags(split)
    
    // We should have used up all arguments at this point:
    guard !split.containsNonTerminatorArguments else {
      // Check if one of the arguments is an unknown option
      for element in split.elements {
        if case .option(let argument) = element.value {
          throw ParserError.unknownOption(InputOrigin.Element.argumentIndex(element.index), argument.name)
        }
      }
       
      let extra = split.coalescedExtraElements()
      throw ParserError.unexpectedExtraValues(extra)
    }
    
    guard let lastCommand = decodedArguments.lazy.compactMap({ $0.command }).last else {
      throw ParserError.invalidState
    }
    
    return lastCommand
  }
  
  /// Extracts the current command from `split`, throwing if decoding isn't
  /// possible.
  fileprivate mutating func parseCurrent(_ split: inout SplitArguments) throws -> ParsableCommand {
    // Parse the arguments, ignoring anything unexpected
    var parser = LenientParser(currentNode.element, split)
    let values = try parser.parse()
    
    if currentNode.element.includesAllUnrecognizedArgument {
      // If this command includes an all-unrecognized argument, any built-in
      // flags will have been parsed into that argument. Check for flags
      // before decoding.
      try checkForBuiltInFlags(values.capturedUnrecognizedArguments)
    }
    
    // Decode the values from ParsedValues into the ParsableCommand:
    let decoder = ArgumentDecoder(values: values, previouslyDecoded: decodedArguments)
    var decodedResult: ParsableCommand
    do {
      decodedResult = try currentNode.element.init(from: decoder)
    } catch let error {
      // If decoding this command failed, see if they were asking for
      // help before propagating that parsing failure.
      try checkForBuiltInFlags(split)
      throw error
    }
    
    // Decoding was successful, so remove the arguments that were used
    // by the decoder.
    split.removeAll(in: decoder.usedOrigins)
    
    // Save the decoded results to add to the next command.
    let newDecodedValues = decoder.previouslyDecoded
      .filter { prev in !decodedArguments.contains(where: { $0.type == prev.type })}
    decodedArguments.append(contentsOf: newDecodedValues)
    decodedArguments.append(DecodedArguments(type: currentNode.element, value: decodedResult))

    return decodedResult
  }
  
  /// Starting with the current node, extracts commands out of `split` and
  /// descends into subcommands as far as possible.
  internal mutating func descendingParse(_ split: inout SplitArguments) throws {
    while true {
      var parsedCommand = try parseCurrent(&split)

      // after decoding a command, make sure to validate it
      do {
        try parsedCommand.validate()
        var lastArgument = decodedArguments.removeLast()
        lastArgument.value = parsedCommand
        decodedArguments.append(lastArgument)
      } catch {
        try checkForBuiltInFlags(split)
        throw CommandError(commandStack: commandStack, parserError: ParserError.userValidationError(error))
      }

      // Look for next command in the argument list.
      if let nextCommand = consumeNextCommand(split: &split) {
        currentNode = nextCommand
        continue
      }
      
      // Look for the help flag before falling back to a default command.
      try checkForBuiltInFlags(split, requireSoloArgument: true)
      
      // No command was found, so fall back to the default subcommand.
      if let defaultSubcommand = currentNode.element.configuration.defaultSubcommand {
        guard let subcommandNode = currentNode.firstChild(equalTo: defaultSubcommand) else {
          throw ParserError.invalidState
        }
        currentNode = subcommandNode
        continue
      }
      
      // No more subcommands to parse.
      return
    }
  }
  
  /// Returns the fully-parsed matching command for `arguments`, or an
  /// appropriate error.
  ///
  /// - Parameter arguments: The array of arguments to parse. This should not
  ///   include the command name as the first argument.
  mutating func parse(arguments: [String]) -> Result<ParsableCommand, CommandError> {
    do {
      try handleCustomCompletion(arguments)
    } catch {
      return .failure(CommandError(commandStack: [commandTree.element], parserError: error as! ParserError))
    }
    
    var split: SplitArguments
    do {
      split = try SplitArguments(arguments: arguments)
    } catch let error as ParserError {
      return .failure(CommandError(commandStack: [commandTree.element], parserError: error))
    } catch {
      return .failure(CommandError(commandStack: [commandTree.element], parserError: .invalidState))
    }
    
    do {
      try checkForCompletionScriptRequest(&split)
      try descendingParse(&split)
      let result = try extractLastParsedValue(split)
      
      // HelpCommand is a valid result, but needs extra information about
      // the tree from the parser to build its stack of commands.
      if var helpResult = result as? HelpCommand {
        try helpResult.buildCommandStack(with: self)
        return .success(helpResult)
      }
      return .success(result)
    } catch let error as CommandError {
      return .failure(error)
    } catch let error as ParserError {
      let error = arguments.isEmpty ? ParserError.noArguments(error) : error
      return .failure(CommandError(commandStack: commandStack, parserError: error))
    } catch let helpRequest as HelpRequested {
      return .success(HelpCommand(
        commandStack: commandStack,
        visibility: helpRequest.visibility))
    } catch {
      return .failure(CommandError(commandStack: commandStack, parserError: .invalidState))
    }
  }
}

// MARK: Completion Script Support

struct GenerateCompletions: ParsableCommand {
  @Option() var generateCompletionScript: String
}

struct AutodetectedGenerateCompletions: ParsableCommand {
  @Flag() var generateCompletionScript = false
}

extension CommandParser {
  func checkForCompletionScriptRequest(_ split: inout SplitArguments) throws {
    // Pseudo-commands don't support `--generate-completion-script` flag
    guard rootCommand.configuration._superCommandName == nil else {
      return
    }
    
    // We don't have the ability to check for `--name [value]`-style args yet,
    // so we need to try parsing two different commands.
    
    // First look for `--generate-completion-script <shell>`
    var completionsParser = CommandParser(GenerateCompletions.self)
    if let result = try? completionsParser.parseCurrent(&split) as? GenerateCompletions {
      throw CommandError(commandStack: commandStack, parserError: .completionScriptRequested(shell: result.generateCompletionScript))
    }
    
    // Check for for `--generate-completion-script` without a value
    var autodetectedParser = CommandParser(AutodetectedGenerateCompletions.self)
    if let result = try? autodetectedParser.parseCurrent(&split) as? AutodetectedGenerateCompletions,
       result.generateCompletionScript
    {
      throw CommandError(commandStack: commandStack, parserError: .completionScriptRequested(shell: nil))
    }
  }
    
  func handleCustomCompletion(_ arguments: [String]) throws {
    // Completion functions use a custom format:
    //
    // <command> ---completion [<subcommand> ...] -- <argument-name> [<completion-text>]
    //
    // The triple-dash prefix makes '---completion' invalid syntax for regular
    // arguments, so it's safe to use for this internal purpose.
    guard arguments.first == "---completion"
      else { return }
    
    var args = arguments.dropFirst()
    var current = commandTree
    while let subcommandName = args.popFirst() {
      // A double dash separates the subcommands from the argument information
      if subcommandName == "--" { break }
      
      guard let nextCommandNode = current.firstChild(withName: subcommandName)
        else { throw ParserError.invalidState }
      current = nextCommandNode
    }
    
    // Some kind of argument name is the next required element
    guard let argToMatch = args.popFirst() else {
      throw ParserError.invalidState
    }
    // Completion text is optional here
    let completionValues = Array(args)

    // Generate the argument set and parse the argument to find in the set
    let argset = ArgumentSet(current.element, visibility: .private, parent: nil)
    let parsedArgument = try! parseIndividualArg(argToMatch, at: 0).first!
    
    // Look up the specified argument and retrieve its custom completion function
    let completionFunction: ([String]) -> [String]
    
    switch parsedArgument.value {
    case .option(let parsed):
      guard let matchedArgument = argset.first(matching: parsed),
        case .custom(let f) = matchedArgument.completion.kind
        else { throw ParserError.invalidState }
      completionFunction = f

    case .value(let str):
      guard let matchedArgument = argset.firstPositional(named: str),
        case .custom(let f) = matchedArgument.completion.kind
        else { throw ParserError.invalidState }
      completionFunction = f
      
    case .terminator:
      throw ParserError.invalidState
    }
    
    // Parsing and retrieval successful! We don't want to continue with any
    // other parsing here, so after printing the result of the completion
    // function, exit with a success code.
    let output = completionFunction(completionValues).joined(separator: "\n")
    throw ParserError.completionScriptCustomResponse(output)
  }
}

// MARK: Building Command Stacks

extension CommandParser {
  /// Builds an array of commands that matches the given command names.
  ///
  /// This stops building the stack if it encounters any command names that
  /// aren't in the command tree, so it's okay to pass a list of arbitrary
  /// commands. Will always return at least the root of the command tree.
  func commandStack(for commandNames: [String]) -> [ParsableCommand.Type] {
    var node = commandTree
    var result = [node.element]
    
    for name in commandNames {
      guard let nextNode = node.firstChild(withName: name) else {
        // Reached a non-command argument.
        // Ignore anything after this point
        return result
      }
      result.append(nextNode.element)
      node = nextNode
    }
    
    return result
  }
  
  func commandStack(for subcommand: ParsableCommand.Type) -> [ParsableCommand.Type] {
    let path = commandTree.path(to: subcommand)
    return path.isEmpty
      ? [commandTree.element]
      : path
  }
}

extension SplitArguments {
  func contains(_ needle: Name) -> Bool {
    self.elements.contains {
      switch $0.value {
      case .option(.name(let name)),
           .option(.nameWithValue(let name, _)):
        return name == needle
      default:
        return false
      }
    }
  }

  func contains(anyOf names: [Name]) -> Bool {
    self.elements.contains {
      switch $0.value {
      case .option(.name(let name)),
           .option(.nameWithValue(let name, _)):
        return names.contains(name)
      default:
        return false
      }
    }
  }
}
