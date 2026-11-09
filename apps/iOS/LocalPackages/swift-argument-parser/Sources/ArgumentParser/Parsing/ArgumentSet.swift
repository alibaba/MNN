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

/// A nested tree of argument definitions.
///
/// The main reason for having a nested representation is to build help output.
/// For output like:
///
///     Usage: mytool [-v | -f] <input> <output>
///
/// The `-v | -f` part is one *set* that’s optional, `<input> <output>` is
/// another. Both of these can then be combined into a third set.
struct ArgumentSet {
  var content: [ArgumentDefinition] = []
  var namePositions: [Name: Int] = [:]
  
  init<S: Sequence>(_ arguments: S) where S.Element == ArgumentDefinition {
    self.content = Array(arguments)
    self.namePositions = Dictionary(
      content.enumerated().flatMap { i, arg in arg.names.map { ($0.nameToMatch, i) } },
      uniquingKeysWith: { first, _ in first })
  }
  
  init() {}
  
  init(_ arg: ArgumentDefinition) {
    self.init([arg])
  }

  init(sets: [ArgumentSet]) {
    self.init(sets.joined())
  }
  
  mutating func append(_ arg: ArgumentDefinition) {
    let newPosition = content.count
    content.append(arg)
    for name in arg.names where namePositions[name.nameToMatch] == nil {
      namePositions[name.nameToMatch] = newPosition
    }
  }
}

extension ArgumentSet: CustomDebugStringConvertible {
  var debugDescription: String {
    content
      .map { $0.debugDescription }
      .joined(separator: " / ")
  }
}

extension ArgumentSet: RandomAccessCollection {
  var startIndex: Int { content.startIndex }
  var endIndex: Int { content.endIndex }
  subscript(position: Int) -> ArgumentDefinition {
    content[position]
  }
}

// MARK: Flag

extension ArgumentSet {
  /// Creates an argument set for a single Boolean flag.
  static func flag(key: InputKey, name: NameSpecification, default initialValue: Bool?, help: ArgumentHelp?) -> ArgumentSet {
    // The flag is required if initialValue is `nil`, otherwise it's optional
    let helpOptions: ArgumentDefinition.Help.Options = initialValue != nil ? .isOptional : []
    let defaultValueString = initialValue == true ? "true" : nil
    
    let help = ArgumentDefinition.Help(
      allValueStrings: [],
      options: helpOptions,
      help: help,
      defaultValue: defaultValueString,
      key: key,
      isComposite: false)
    let arg = ArgumentDefinition(kind: .name(key: key, specification: name), help: help, completion: .default, update: .nullary({ (origin, name, values) in
      values.set(true, forKey: key, inputOrigin: origin)
    }), initial: { origin, values in
      if let initialValue = initialValue {
        values.set(initialValue, forKey: key, inputOrigin: origin)
      }
    })
    return ArgumentSet(arg)
  }

  static func updateFlag<Value: Equatable>(key: InputKey, value: Value, origin: InputOrigin, values: inout ParsedValues, exclusivity: FlagExclusivity) throws {
    let hasUpdated: Bool
    if let previous = values.element(forKey: key) {
      hasUpdated = !previous.inputOrigin.elements.isEmpty
    } else {
      hasUpdated = false
    }
    
    switch (hasUpdated, exclusivity.base) {
    case (true, .exclusive):
      // This value has already been set.
      if let previous = values.element(forKey: key) {
        if (previous.value as? Value) == value {
          // setting the value again will consume the argument
          values.set(value, forKey: key, inputOrigin: origin)
        }
        else {
          throw ParserError.duplicateExclusiveValues(previous: previous.inputOrigin, duplicate: origin, originalInput: values.originalInput)
        }
      }
    case (true, .chooseFirst):
      values.update(forKey: key, inputOrigin: origin, initial: value, closure: { _ in })
    case (false, _), (_, .chooseLast):
      values.set(value, forKey: key, inputOrigin: origin)
    }
  }
  
  /// Creates an argument set for a pair of inverted Boolean flags.
  static func flag(
    key: InputKey,
    name: NameSpecification,
    default initialValue: Bool?,
    required: Bool,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity,
    help: ArgumentHelp?) -> ArgumentSet
  {
    let helpOptions: ArgumentDefinition.Help.Options = required ? [] : .isOptional
        
    let (enableNames, disableNames) = inversion.enableDisableNamePair(for: key, name: name)
    let initialValueNames = initialValue.map {
      $0 ? enableNames : disableNames
    }
    
    let enableHelp = ArgumentDefinition.Help(allValueStrings: [], options: helpOptions, help: help, defaultValue: initialValueNames?.first?.synopsisString, key: key, isComposite: true)
    let disableHelp = ArgumentDefinition.Help(allValueStrings: [], options: [.isOptional], help: help, defaultValue: nil, key: key, isComposite: false)

    let enableArg = ArgumentDefinition(kind: .named(enableNames), help: enableHelp, completion: .default, update: .nullary({ (origin, name, values) in
        try ArgumentSet.updateFlag(key: key, value: true, origin: origin, values: &values, exclusivity: exclusivity)
    }), initial: { origin, values in
      if let initialValue = initialValue {
        values.set(initialValue, forKey: key, inputOrigin: origin)
      }
    })
    let disableArg = ArgumentDefinition(kind: .named(disableNames), help: disableHelp, completion: .default, update: .nullary({ (origin, name, values) in
        try ArgumentSet.updateFlag(key: key, value: false, origin: origin, values: &values, exclusivity: exclusivity)
    }), initial: { _, _ in })
    return ArgumentSet([enableArg, disableArg])
  }
  
  /// Creates an argument set for an incrementing integer flag.
  static func counter(key: InputKey, name: NameSpecification, help: ArgumentHelp?) -> ArgumentSet {
    let help = ArgumentDefinition.Help(allValueStrings: [], options: [.isOptional, .isRepeating], help: help, defaultValue: nil, key: key, isComposite: false)
    let arg = ArgumentDefinition(kind: .name(key: key, specification: name), help: help, completion: .default, update: .nullary({ (origin, name, values) in
      guard let a = values.element(forKey: key)?.value, let b = a as? Int else {
        throw ParserError.invalidState
      }
      values.set(b + 1, forKey: key, inputOrigin: origin)
    }), initial: { origin, values in
      values.set(0, forKey: key, inputOrigin: origin)
    })
    return ArgumentSet(arg)
  }
}

extension ArgumentSet {
  /// Fills the given `ParsedValues` instance with initial values from this
  /// argument set.
  func setInitialValues(into parsed: inout ParsedValues) throws {
    for arg in self {
      try arg.initial(InputOrigin(), &parsed)
    }
  }
}

extension ArgumentSet {
  /// Find an `ArgumentDefinition` that matches the given `ParsedArgument`.
  ///
  /// As we iterate over the values from the command line, we try to find a
  /// definition that matches the particular element.
  /// - Parameters:
  ///   - parsed: The argument from the command line
  ///   - origin: Where `parsed` came from.
  /// - Returns: The matching definition.
  func first(
    matching parsed: ParsedArgument
  ) -> ArgumentDefinition? {
    namePositions[parsed.name].map { content[$0] }
  }
  
  func firstPositional(
    named name: String
  ) -> ArgumentDefinition? {
    let key = InputKey(name: name, parent: nil)
    return first(where: { $0.help.keys.contains(key) })
  }
}

/// A parser for a given input and set of arguments defined by the given
/// command.
///
/// This parser will consume only the arguments that it understands. If any
/// arguments are declared to capture all remaining input, or a subcommand
/// is configured as such, parsing stops on the first positional argument or
/// unrecognized dash-prefixed argument.
struct LenientParser {
  var command: ParsableCommand.Type
  var argumentSet: ArgumentSet
  var inputArguments: SplitArguments
  
  init(_ command: ParsableCommand.Type, _ split: SplitArguments) {
    self.command = command
    self.argumentSet = ArgumentSet(command, visibility: .private, parent: nil)
    self.inputArguments = split
  }
  
  var defaultCapturesForPassthrough: Bool {
    command.defaultIncludesPassthroughArguments
  }
  
  var subcommands: [ParsableCommand.Type] {
    command.configuration.subcommands
  }
  
  mutating func parseValue(
    _ argument: ArgumentDefinition,
    _ parsed: ParsedArgument,
    _ originElement: InputOrigin.Element,
    _ update: ArgumentDefinition.Update.Unary,
    _ result: inout ParsedValues,
    _ usedOrigins: inout InputOrigin
  ) throws {
    let origin = InputOrigin(elements: [originElement])
    switch argument.parsingStrategy {
    case .default:
      // We need a value for this option.
      if let value = parsed.value {
        // This was `--foo=bar` style:
        try update(origin, parsed.name, value, &result)
        usedOrigins.formUnion(origin)
      } else if argument.allowsJoinedValue,
         let (origin2, value) = inputArguments.extractJoinedElement(at: originElement)
      {
        // Found a joined argument
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, String(value), &result)
        usedOrigins.formUnion(origins)
      } else if let (origin2, value) = inputArguments.popNextElementIfValue(after: originElement) {
        // Use `popNextElementIfValue(after:)` to handle cases where short option
        // labels are combined
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, value, &result)
        usedOrigins.formUnion(origins)
      } else {
        throw ParserError.missingValueForOption(origin, parsed.name)
      }
      
    case .scanningForValue:
      // We need a value for this option.
      if let value = parsed.value {
        // This was `--foo=bar` style:
        try update(origin, parsed.name, value, &result)
        usedOrigins.formUnion(origin)
      } else if argument.allowsJoinedValue,
          let (origin2, value) = inputArguments.extractJoinedElement(at: originElement) {
        // Found a joined argument
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, String(value), &result)
        usedOrigins.formUnion(origins)
      } else if let (origin2, value) = inputArguments.popNextValue(after: originElement) {
        // Use `popNext(after:)` to handle cases where short option
        // labels are combined
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, value, &result)
        usedOrigins.formUnion(origins)
      } else {
        throw ParserError.missingValueForOption(origin, parsed.name)
      }
      
    case .unconditional:
      // Use an attached value if it exists...
      if let value = parsed.value {
        // This was `--foo=bar` style:
        try update(origin, parsed.name, value, &result)
        usedOrigins.formUnion(origin)
      } else if argument.allowsJoinedValue,
          let (origin2, value) = inputArguments.extractJoinedElement(at: originElement) {
        // Found a joined argument
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, String(value), &result)
        usedOrigins.formUnion(origins)
      } else {
        guard let (origin2, value) = inputArguments.popNextElementAsValue(after: originElement) else {
          throw ParserError.missingValueForOption(origin, parsed.name)
        }
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, value, &result)
        usedOrigins.formUnion(origins)
      }
      
    case .allRemainingInput:
      // Reset initial value with the found input origins:
      try argument.initial(origin, &result)
      
      // Use an attached value if it exists...
      if let value = parsed.value {
        // This was `--foo=bar` style:
        try update(origin, parsed.name, value, &result)
        usedOrigins.formUnion(origin)
      } else if argument.allowsJoinedValue,
          let (origin2, value) = inputArguments.extractJoinedElement(at: originElement) {
        // Found a joined argument
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, String(value), &result)
        usedOrigins.formUnion(origins)
        inputArguments.removeAll(in: usedOrigins)
      }
      
      // ...and then consume the rest of the arguments
      while let (origin2, value) = inputArguments.popNextElementAsValue(after: originElement) {
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, value, &result)
        usedOrigins.formUnion(origins)
      }
      
    case .upToNextOption:
      // Use an attached value if it exists...
      var foundAttachedValue = false
      if let value = parsed.value {
        // This was `--foo=bar` style:
        try update(origin, parsed.name, value, &result)
        usedOrigins.formUnion(origin)
        foundAttachedValue = true
      } else if argument.allowsJoinedValue,
          let (origin2, value) = inputArguments.extractJoinedElement(at: originElement) {
        // Found a joined argument
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, String(value), &result)
        usedOrigins.formUnion(origins)
        inputArguments.removeAll(in: usedOrigins)
        foundAttachedValue = true
      }
      
      // Clear out the initial origin first, since it can include
      // the exploded elements of an options group (see issue #327).
      usedOrigins.formUnion(origin)
      inputArguments.removeAll(in: origin)
      
      // Fix incorrect error message
      // for @Option array without values (see issue #434).
      guard let first = inputArguments.elements.first,
            first.isValue
      else {
        // No independent values to be found, which is an error if there was
        // no `--foo=bar`-style value already found.
        if foundAttachedValue {
          break
        } else {
          throw ParserError.missingValueForOption(origin, parsed.name)
        }
      }
      
      // ...and then consume the arguments until hitting an option
      while let (origin2, value) = inputArguments.popNextElementIfValue() {
        let origins = origin.inserting(origin2)
        try update(origins, parsed.name, value, &result)
        usedOrigins.formUnion(origins)
      }
      
    case .postTerminator, .allUnrecognized:
      // These parsing kinds are for arguments only.
      throw ParserError.invalidState
    }
  }
  
  mutating func parsePositionalValues(
    from unusedInput: SplitArguments,
    into result: inout ParsedValues
  ) throws {
    var endOfInput = unusedInput.elements.endIndex
    
    // If this argument set includes a definition that should collect all the
    // post-terminator inputs, capture them before trying to fill other
    // `@Argument` definitions.
    if let postTerminatorArg = argumentSet.first(where: { def in
      def.isRepeatingPositional && def.parsingStrategy == .postTerminator
    }),
       case let .unary(update) = postTerminatorArg.update,
       let terminatorIndex = unusedInput.elements.firstIndex(where: \.isTerminator)
    {
      for input in unusedInput.elements[(terminatorIndex + 1)...] {
        // Everything post-terminator is a value, force-unwrapping here is safe:
        let value = input.value.valueString!
        try update([.argumentIndex(input.index)], nil, value, &result)
      }
      
      endOfInput = terminatorIndex
    }

    // Create a stack out of the remaining unused inputs that aren't "partial"
    // arguments (i.e. the individual components of a `-vix` grouped short
    // option input).
    var argumentStack = unusedInput.elements[..<endOfInput].filter {
      $0.index.subIndex == .complete
    }[...]
    guard !argumentStack.isEmpty else { return }
        
    /// Pops arguments off the stack until the next valid value. Skips over
    /// dash-prefixed inputs unless `unconditional` is `true`.
    func next(unconditional: Bool) -> SplitArguments.Element? {
      while let arg = argumentStack.popFirst() {
        if arg.isValue || unconditional {
          return arg
        }
      }

      return nil
    }
    
    // For all positional arguments, consume one or more inputs.
    var usedOrigins = InputOrigin()
    ArgumentLoop:
    for argumentDefinition in argumentSet {
      guard case .positional = argumentDefinition.kind else { continue }
      switch argumentDefinition.parsingStrategy {
      case .default, .allRemainingInput:
        break
      default:
        continue ArgumentLoop
      }
      guard case let .unary(update) = argumentDefinition.update else {
        preconditionFailure("Shouldn't see a nullary positional argument.")
      }
      let allowOptionsAsInput = argumentDefinition.parsingStrategy == .allRemainingInput
      
      repeat {
        guard let arg = next(unconditional: allowOptionsAsInput) else {
          break ArgumentLoop
        }
        let origin: InputOrigin.Element = .argumentIndex(arg.index)
        let value = unusedInput.originalInput(at: origin)!
        try update([origin], nil, value, &result)
        usedOrigins.insert(origin)
      } while argumentDefinition.isRepeatingPositional
    }
        
    // If there's an `.allUnrecognized` argument array, collect leftover args.
    if let allUnrecognizedArg = argumentSet.first(where: { def in
      def.isRepeatingPositional && def.parsingStrategy == .allUnrecognized
    }),
       case let .unary(update) = allUnrecognizedArg.update
    {
      result.capturedUnrecognizedArguments = SplitArguments(
        _elements: Array(argumentStack),
        originalInput: [])
      while let arg = argumentStack.popFirst() {
        let origin: InputOrigin.Element = .argumentIndex(arg.index)
        let value = unusedInput.originalInput(at: origin)!
        try update([origin], nil, value, &result)
      }
    }
  }

  mutating func parse() throws -> ParsedValues {
    let originalInput = inputArguments
    defer { inputArguments = originalInput }
    
    // If this argument set includes a positional argument that unconditionally
    // captures all remaining input, we use a different behavior, where we
    // shortcut out at the first sign of a positional argument or unrecognized
    // option/flag label.
    let capturesForPassthrough = defaultCapturesForPassthrough || argumentSet.contains(where: { arg in
      arg.isRepeatingPositional && arg.parsingStrategy == .allRemainingInput
    })
    
    var result = ParsedValues(elements: [:], originalInput: inputArguments.originalInput)
    var allUsedOrigins = InputOrigin()
    
    try argumentSet.setInitialValues(into: &result)
    
    // Loop over all arguments:
    ArgumentLoop:
    while let (origin, next) = inputArguments.popNext() {
      var usedOrigins = InputOrigin()
      defer {
        inputArguments.removeAll(in: usedOrigins)
        allUsedOrigins.formUnion(usedOrigins)
      }
      
      switch next.value {
      case .value(let argument):
        // Special handling for matching subcommand names. We generally want
        // parsing to skip over unrecognized input, but if the current
        // command or the matched subcommand captures all remaining input,
        // then we want to break out of parsing at this point.
        let matchedSubcommand = subcommands.first(where: { 
          $0._commandName == argument || $0.configuration.aliases.contains(argument) 
        })
        if let matchedSubcommand {
          if !matchedSubcommand.includesPassthroughArguments && defaultCapturesForPassthrough {
            continue ArgumentLoop
          } else if matchedSubcommand.includesPassthroughArguments {
            break ArgumentLoop
          }
        }
        
        // If we're capturing all, the first positional value represents the
        // start of positional input.
        if capturesForPassthrough { break ArgumentLoop }
        // We'll parse positional values later.
        break
      case let .option(parsed):
        // Look for an argument that matches this `--option` or `-o`-style
        // input. If we can't find one, just move on to the next input. We
        // defer catching leftover arguments until we've fully extracted all
        // the information for the selected command.
        guard let argument = argumentSet.first(matching: parsed) else
        {
          // If we're capturing all, an unrecognized option/flag is the start
          // of positional input. However, the first time we see an option
          // pack (like `-fi`) it looks like a long name with a single-dash
          // prefix, which may not match an argument even if its subcomponents
          // will match.
          if capturesForPassthrough && parsed.subarguments.isEmpty { break ArgumentLoop }
          
          // Otherwise, continue parsing. This option/flag may get picked up
          // by a child command.
          continue
        }
        
        switch argument.update {
        case let .nullary(update):
          // We don’t expect a value for this option.
          guard parsed.value == nil else {
            throw ParserError.unexpectedValueForOption(origin, parsed.name, parsed.value!)
          }
          _ = try update([origin], parsed.name, &result)
          usedOrigins.insert(origin)
        case let .unary(update):
          try parseValue(argument, parsed, origin, update, &result, &usedOrigins)
        }
      case .terminator:
        // Ignore the terminator, it might get picked up as a positional value later.
        break
      }
    }
    
    // We have parsed all non-positional values at this point.
    // Next: parse / consume the positional values.
    var unusedArguments = originalInput
    unusedArguments.removeAll(in: allUsedOrigins)
    try parsePositionalValues(from: unusedArguments, into: &result)

    return result
  }
}
