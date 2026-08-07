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

#if swift(>=5.11)
internal import protocol Foundation.LocalizedError
#elseif swift(>=5.10)
import protocol Foundation.LocalizedError
#else
@_implementationOnly import protocol Foundation.LocalizedError
#endif

struct UsageGenerator {
  var toolName: String
  var definition: ArgumentSet
}

extension UsageGenerator {
  init(definition: ArgumentSet) {
    let toolName = CommandLine._staticArguments[0]
      .split(separator: "/").last.map(String.init) ?? "<command>"
    self.init(toolName: toolName, definition: definition)
  }
  
  init(toolName: String, parsable: ParsableArguments, visibility: ArgumentVisibility, parent: InputKey?) {
    self.init(
      toolName: toolName,
      definition: ArgumentSet(type(of: parsable), visibility: visibility, parent: parent))
  }
  
  init(toolName: String, definition: [ArgumentSet]) {
    self.init(toolName: toolName, definition: ArgumentSet(sets: definition))
  }
}

extension UsageGenerator {
  /// The tool synopsis.
  ///
  /// In `roff`.
  var synopsis: String {
    var options = Array(definition)
    switch options.count {
    case 0:
      return toolName
    case let x where x > 12:
      // When we have too many options, keep required and positional arguments,
      // but discard the rest.
      options = options.filter {
        $0.isPositional || !$0.help.options.contains(.isOptional)
      }
      // If there are between 1 and 12 options left, print them, otherwise print
      // a simplified usage string.
      if !options.isEmpty, options.count <= 12 {
        let synopsis = options
          .map { $0.synopsis }
          .joined(separator: " ")
        return "\(toolName) [<options>] \(synopsis)"
      }
      return "\(toolName) <options>"
    default:
      let synopsis = options
        .map { $0.synopsis }
        .joined(separator: " ")
      return "\(toolName) \(synopsis)"
    }
  }
}

extension ArgumentDefinition {
  var synopsisForHelp: String {
    switch kind {
    case .named:
      let joinedSynopsisString = names
        .partitioned
        .map { $0.synopsisString }
        .joined(separator: ", ")

      switch update {
      case .unary:
        return "\(joinedSynopsisString) <\(valueName)>"
      case .nullary:
        return joinedSynopsisString
      }
    case .positional:
      return "<\(valueName)>"
    case .default:
      return ""
    }
  }

  var unadornedSynopsis: String {
    switch kind {
    case .named:
      guard let name = names.preferredName else {
        fatalError("preferredName cannot be nil for named arguments")
      }

      switch update {
      case .unary:
        return "\(name.synopsisString) <\(valueName)>"
      case .nullary:
        return name.synopsisString
      }
    case .positional:
      return "<\(valueName)>"
    case .default:
      return ""
    }
  }

  var synopsis: String {
    var synopsis = unadornedSynopsis
    if help.options.contains(.isRepeating) {
      synopsis += " ..."
    }
    if help.options.contains(.isOptional) {
      synopsis = "[\(synopsis)]"
    }
    if parsingStrategy == .postTerminator {
      synopsis = "-- \(synopsis)"
    }
    return synopsis
  }
}

extension ArgumentSet {
  /// Will generate a descriptive help message if possible.
  ///
  /// If no descriptive help message can be generated, `nil` will be returned.
  ///
  /// - Parameter error: the parse error that occurred.
  func errorDescription(error: Swift.Error) -> String? {
    switch error {
    case let parserError as ParserError:
      return ErrorMessageGenerator(arguments: self, error: parserError)
        .makeErrorMessage()
    case let commandError as CommandError:
      return ErrorMessageGenerator(arguments: self, error: commandError.parserError)
        .makeErrorMessage()
    default:
      return nil
    }
  }
  
  func helpDescription(error: Swift.Error) -> String? {
    switch error {
    case let parserError as ParserError:
      return ErrorMessageGenerator(arguments: self, error: parserError)
        .makeHelpMessage()
    case let commandError as CommandError:
      return ErrorMessageGenerator(arguments: self, error: commandError.parserError)
        .makeHelpMessage()
    default:
      return nil
    }
  }
}

struct ErrorMessageGenerator {
  var arguments: ArgumentSet
  var error: ParserError
}

extension ErrorMessageGenerator {
  func makeErrorMessage() -> String? {
    switch error {
    case .helpRequested, .versionRequested, .completionScriptRequested, .completionScriptCustomResponse, .dumpHelpRequested:
      return nil

    case .unsupportedShell(let shell?):
      return unsupportedShell(shell)
    case .unsupportedShell:
      return unsupportedAutodetectedShell
      
    case .notImplemented:
      return notImplementedMessage
    case .invalidState:
      return invalidState
    case .unknownOption(let o, let n):
      return unknownOptionMessage(origin: o, name: n)
    case .missingValueForOption(let o, let n):
      return missingValueForOptionMessage(origin: o, name: n)
    case .unexpectedValueForOption(let o, let n, let v):
      return unexpectedValueForOptionMessage(origin: o, name: n, value: v)
    case .unexpectedExtraValues(let v):
      return unexpectedExtraValuesMessage(values: v)
    case .duplicateExclusiveValues(previous: let previous, duplicate: let duplicate, originalInput: let arguments):
      return duplicateExclusiveValues(previous: previous, duplicate: duplicate, arguments: arguments)
    case .noValue(forKey: let k):
      return noValueMessage(key: k)
    case .unableToParseValue(let o, let n, let v, forKey: let k, originalError: let e):
      return unableToParseValueMessage(origin: o, name: n, value: v, key: k, error: e)
    case .invalidOption(let str):
      return "Invalid option: \(str)"
    case .nonAlphanumericShortOption(let c):
      return "Invalid option: -\(c)"
    case .missingSubcommand:
      return "Missing required subcommand."
    case .userValidationError(let error):
      switch error {
      case let error as LocalizedError:
        return error.errorDescription
      default:
        return String(describing: error)
      }
    case .noArguments(let error):
      switch error {
      case let error as ParserError:
        return ErrorMessageGenerator(arguments: self.arguments, error: error).makeErrorMessage()
      case let error as LocalizedError:
        return error.errorDescription
      default:
        return String(describing: error)
      }
    }
  }
  
  func makeHelpMessage() -> String? {
    switch error {
    case .unableToParseValue(let o, let n, let v, forKey: let k, originalError: let e):
      return unableToParseHelpMessage(origin: o, name: n, value: v, key: k, error: e)
    case .missingValueForOption(_, let n):
      return missingValueForOptionHelpMessage(name: n)
    case .noValue(let k):
      return noValueHelpMessage(key: k)
    default:
      return nil
    }
  }
}

extension ErrorMessageGenerator {
  func arguments(for key: InputKey) -> [ArgumentDefinition] {
    arguments
      .filter { $0.help.keys.contains(key) }
  }
  
  func help(for key: InputKey) -> ArgumentDefinition.Help? {
    arguments
      .first { $0.help.keys.contains(key) }
      .map { $0.help }
  }
  
  func valueName(for name: Name) -> String? {
    arguments
      .first { $0.names.contains(name) }
      .map { $0.valueName }
  }
}

extension ErrorMessageGenerator {
  var notImplementedMessage: String {
    return "Internal error. Parsing command-line arguments hit unimplemented code path."
  }
  var invalidState: String {
    return "Internal error. Invalid state while parsing command-line arguments."
  }

  var unsupportedAutodetectedShell: String {
    """
    Can't autodetect a supported shell.
    Please use --generate-completion-script=<shell> with one of:
        \(CompletionShell.allCases.map { $0.rawValue }.joined(separator: " "))
    """
  }

  func unsupportedShell(_ shell: String) -> String {
    """
    Can't generate completion scripts for '\(shell)'.
    Please use --generate-completion-script=<shell> with one of:
        \(CompletionShell.allCases.map { $0.rawValue }.joined(separator: " "))
    """
  }

  func unknownOptionMessage(origin: InputOrigin.Element, name: Name) -> String {
    if case .short = name {
      return "Unknown option '\(name.synopsisString)'"
    }
    
    // An empirically derived magic number
    let SIMILARITY_FLOOR = 4

    let notShort: (Name) -> Bool = { (name: Name) in
      switch name {
      case .short: return false
      case .long: return true
      case .longWithSingleDash: return true
      }
    }
    let suggestion = arguments
      .flatMap({ $0.names })
      .filter({ $0.synopsisString.editDistance(to: name.synopsisString) < SIMILARITY_FLOOR }) // only include close enough suggestion
      .filter(notShort) // exclude short option suggestions
      .min(by: { lhs, rhs in // find the suggestion closest to the argument
        lhs.synopsisString.editDistance(to: name.synopsisString) < rhs.synopsisString.editDistance(to: name.synopsisString)
      })
    
    if let suggestion = suggestion {
      return "Unknown option '\(name.synopsisString)'. Did you mean '\(suggestion.synopsisString)'?"
    }
    return "Unknown option '\(name.synopsisString)'"
  }
  
  func missingValueForOptionMessage(origin: InputOrigin, name: Name) -> String {
    if let valueName = valueName(for: name) {
      return "Missing value for '\(name.synopsisString) <\(valueName)>'"
    } else {
      return "Missing value for '\(name.synopsisString)'"
    }
  }
  
  func unexpectedValueForOptionMessage(origin: InputOrigin.Element, name: Name, value: String) -> String? {
    return "The option '\(name.synopsisString)' does not take any value, but '\(value)' was specified."
  }
  
  func unexpectedExtraValuesMessage(values: [(InputOrigin, String)]) -> String? {
    switch values.count {
    case 0:
      return nil
    case 1:
      return "Unexpected argument '\(values.first!.1)'"
    default:
      let v = values.map { $0.1 }.joined(separator: "', '")
      return "\(values.count) unexpected arguments: '\(v)'"
    }
  }
  
  func duplicateExclusiveValues(previous: InputOrigin, duplicate: InputOrigin, arguments: [String]) -> String? {
    func elementString(_ origin: InputOrigin, _ arguments: [String]) -> String? {
      guard case .argumentIndex(let split) = origin.elements.first else { return nil }
      var argument = "\'\(arguments[split.inputIndex.rawValue])\'"
      if case let .sub(offsetIndex) = split.subIndex {
        let stringIndex = argument.index(argument.startIndex, offsetBy: offsetIndex+2)
        argument = "\'\(argument[stringIndex])\' in \(argument)"
      }
      return "flag \(argument)"
    }

    // Note that the RHS of these coalescing operators cannot be reached at this time.
    let dupeString = elementString(duplicate, arguments) ?? "position \(duplicate)"
    let origString = elementString(previous, arguments) ?? "position \(previous)"

    //TODO: review this message once environment values are supported.
    return "Value to be set with \(dupeString) had already been set with \(origString)"
  }
  
  func noValueMessage(key: InputKey) -> String? {
    let args = arguments(for: key)
    let possibilities: [String] = args.compactMap {
      $0.help.visibility.base == .default
        ? $0.nonOptional.synopsis
        : nil
    }
    switch possibilities.count {
    case 0:
      return "No value set for non-argument var \(key). Replace with a static variable, or let constant."
    case 1:
      return "Missing expected argument '\(possibilities.first!)'"
    default:
      let p = possibilities.joined(separator: "', '")
      return "Missing one of: '\(p)'"
    }
  }
  
  func unableToParseHelpMessage(origin: InputOrigin, name: Name?, value: String, key: InputKey, error: Error?) -> String {
    guard let abstract = help(for: key)?.abstract else { return "" }
    
    let valueName = arguments(for: key).first?.valueName
    
    switch (name, valueName) {
    case let (n?, v?):
      return "\(n.synopsisString) <\(v)>  \(abstract)"
    case let (_, v?):
      return "<\(v)>  \(abstract)"
    case (_, _):
      return ""
    }
  }

  func missingValueForOptionHelpMessage(name: Name) -> String {
    guard let arg = arguments.first(where: { $0.names.contains(name) }) else {
      return ""
    }

    let help = arg.help.abstract
    return "\(name.synopsisString) <\(arg.valueName)>  \(help)"
  }

  func noValueHelpMessage(key: InputKey) -> String {
    guard let abstract = help(for: key)?.abstract else { return "" }
    guard let arg = arguments(for: key).first else { return "" }

    if let synopsisString = arg.names.first?.synopsisString {
      return "\(synopsisString) <\(arg.valueName)>  \(abstract)"
    }
    return "<\(arg.valueName)>  \(abstract)"
  }
  
  func unableToParseValueMessage(origin: InputOrigin, name: Name?, value: String, key: InputKey, error: Error?) -> String {
    let argumentValue = arguments(for: key).first
    let valueName = argumentValue?.valueName

    // We want to make the "best effort" in producing a custom error message.
    // We favor `LocalizedError.errorDescription` and fall back to
    // `CustomStringConvertible`. To opt in, return your custom error message
    // as the `description` property of `CustomStringConvertible`.
    let customErrorMessage: String = {
      switch error {
      case let err as LocalizedError where err.errorDescription != nil:
        return ": " + err.errorDescription! // !!! Checked above that this will not be nil
      case let err?:
        return ": " + String(describing: err)
      default:
        return argumentValue?.formattedValueList ?? ""
      }
    }()

    switch (name, valueName) {
    case let (n?, v?):
      return "The value '\(value)' is invalid for '\(n.synopsisString) <\(v)>'\(customErrorMessage)"
    case let (_, v?):
      return "The value '\(value)' is invalid for '<\(v)>'\(customErrorMessage)"
    case let (n?, _):
      return "The value '\(value)' is invalid for '\(n.synopsisString)'\(customErrorMessage)"
    case (nil, nil):
      return "The value '\(value)' is invalid.\(customErrorMessage)"
    }
  }
}

private extension ArgumentDefinition {
  var formattedValueList: String {
    if help.allValueStrings.isEmpty {
      return ""
    }

    if help.allValueStrings.count < 6 {
      let quotedValues = help.allValueStrings.map { "'\($0)'" }
      let validList: String
      if quotedValues.count <= 2 {
        validList = quotedValues.joined(separator: " and ")
      } else {
        validList = quotedValues.dropLast().joined(separator: ", ") + " or \(quotedValues.last!)"
      }
      return ". Please provide one of \(validList)."
    } else {
      let bulletValueList = help.allValueStrings.map { "  - \($0)" }.joined(separator: "\n")
      return ". Please provide one of the following:\n\(bulletValueList)"
    }
  }
}
