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

struct ArgumentDefinition {
  /// A closure that modifies a `ParsedValues` instance to include this
  /// argument's value.
  enum Update {
    typealias Nullary = (InputOrigin, Name?, inout ParsedValues) throws -> Void
    typealias Unary = (InputOrigin, Name?, String, inout ParsedValues) throws -> Void
    
    /// An argument that gets its value solely from its presence.
    case nullary(Nullary)
    
    /// An argument that takes a string as its value.
    case unary(Unary)
  }
  
  typealias Initial = (InputOrigin, inout ParsedValues) throws -> Void
  
  enum Kind {
    /// An option or flag, with a name and an optional value.
    case named([Name])
    
    /// A positional argument.
    case positional
    
    /// A pseudo-argument that takes its value from a property's default value
    /// instead of from command-line arguments.
    case `default`
  }
  
  struct Help {
    struct Options: OptionSet {
      var rawValue: UInt

      static let isOptional = Options(rawValue: 1 << 0)
      static let isRepeating = Options(rawValue: 1 << 1)
    }

    var options: Options
    var defaultValue: String?
    var keys: [InputKey]
    var allValueStrings: [String]
    var isComposite: Bool
    var abstract: String
    var discussion: String
    var valueName: String
    var visibility: ArgumentVisibility
    var parentTitle: String

    init(
      allValueStrings: [String],
      options: Options,
      help: ArgumentHelp?,
      defaultValue: String?,
      key: InputKey,
      isComposite: Bool
    ) {
      self.options = options
      self.defaultValue = defaultValue
      self.keys = [key]
      self.allValueStrings = allValueStrings
      self.isComposite = isComposite
      self.abstract = help?.abstract ?? ""
      self.discussion = help?.discussion ?? ""
      self.valueName = help?.valueName ?? ""
      self.visibility = help?.visibility ?? .default
      self.parentTitle = ""
    }
  }
  
  /// This folds the public `ArrayParsingStrategy` and `SingleValueParsingStrategy`
  /// into a single enum.
  enum ParsingStrategy {
    /// Expect the next `SplitArguments.Element` to be a value and parse it.
    /// Will fail if the next input is an option.
    case `default`
    /// Parse the next `SplitArguments.Element.value`
    case scanningForValue
    /// Parse the next `SplitArguments.Element` as a value, regardless of its type.
    case unconditional
    /// Parse multiple `SplitArguments.Element.value` up to the next non-`.value`
    case upToNextOption
    /// Parse all remaining `SplitArguments.Element` as values, regardless of its type.
    case allRemainingInput
    /// Collect all the elements after the terminator, preventing them from
    /// appearing in any other position.
    case postTerminator
    /// Collect all unused inputs once recognized arguments/options/flags have
    /// been parsed.
    case allUnrecognized
  }
  
  var kind: Kind
  var help: Help
  var completion: CompletionKind
  var parsingStrategy: ParsingStrategy
  var update: Update
  var initial: Initial
  
  var names: [Name] {
    switch kind {
    case .named(let n): return n
    case .positional, .default: return []
    }
  }
  
  var valueName: String {
    help.valueName.mapEmpty {
      names.preferredName?.valueString
        ?? help.keys.first?.name.convertedToSnakeCase(separator: "-")
        ?? "value"
    }
  }

  init(
    kind: Kind,
    help: Help,
    completion: CompletionKind,
    parsingStrategy: ParsingStrategy = .default,
    update: Update,
    initial: @escaping Initial = { _, _ in }
  ) {
    if case (.positional, .nullary) = (kind, update) {
      preconditionFailure("Can't create a nullary positional argument.")
    }
    
    self.kind = kind
    self.help = help
    self.completion = completion
    self.parsingStrategy = parsingStrategy
    self.update = update
    self.initial = initial
  }
}

extension ArgumentDefinition: CustomDebugStringConvertible {
  var debugDescription: String {
    switch (kind, update) {
    case (.named(let names), .nullary):
      return names
        .map { $0.synopsisString }
        .joined(separator: ",")
    case (.named(let names), .unary):
      return names
        .map { $0.synopsisString }
        .joined(separator: ",")
        + " <\(valueName)>"
    case (.positional, _):
      return "<\(valueName)>"
    case (.default, _):
      return ""
    }
  }
}

extension ArgumentDefinition {
  var optional: ArgumentDefinition {
    var result = self
    result.help.options.insert(.isOptional)
    return result
  }
  
  var nonOptional: ArgumentDefinition {
    var result = self
    result.help.options.remove(.isOptional)
    return result
  }
}

extension ArgumentDefinition {
  var isPositional: Bool {
    if case .positional = kind {
      return true
    }
    return false
  }
  
  var isRepeatingPositional: Bool {
    isPositional && help.options.contains(.isRepeating)
  }

  var isNullary: Bool {
    if case .nullary = update {
      return true
    } else {
      return false
    }
  }
  
  var allowsJoinedValue: Bool {
    names.contains(where: { $0.allowsJoined })
  }
}

extension ArgumentDefinition.Kind {
  static func name(key: InputKey, specification: NameSpecification) -> ArgumentDefinition.Kind {
    let names = specification.makeNames(key)
    return ArgumentDefinition.Kind.named(names)
  }
}


// MARK: - Common @Argument, @Option, Unparsed Initializer Path
extension ArgumentDefinition {
  // MARK: Unparsed Keys
  /// Creates an argument definition for a property that isn't parsed from the
  /// command line.
  ///
  /// This initializer is used for any property defined on a `ParsableArguments`
  /// type that isn't decorated with one of ArgumentParser's property wrappers.
  init(unparsedKey: String, default defaultValue: Any?, parent: InputKey?) {
    self.init(
      container: Bare<Any>.self,
      key: InputKey(name: unparsedKey, parent: parent),
      kind: .default,
      allValueStrings: [],
      help: .private,
      defaultValueDescription: nil,
      parsingStrategy: .default,
      parser: { (key, origin, name, valueString) in
        throw ParserError.unableToParseValue(
          origin, name, valueString, forKey: key, originalError: nil)
      },
      initial: defaultValue,
      completion: nil)
  }

  init<Container>(
    container: Container.Type,
    key: InputKey,
    kind: ArgumentDefinition.Kind,
    help: ArgumentHelp?,
    parsingStrategy: ParsingStrategy,
    initial: Container.Initial?,
    completion: CompletionKind?
  ) where Container: ArgumentDefinitionContainerExpressibleByArgument {
    self.init(
      container: Container.self,
      key: key,
      kind: kind,
      allValueStrings: Container.Contained.allValueStrings,
      help: help,
      defaultValueDescription: Container.defaultValueDescription(initial),
      parsingStrategy: parsingStrategy,
      parser: { (key, origin, name, valueString) -> Container.Contained in
        guard let value = Container.Contained(argument: valueString) else {
          throw ParserError.unableToParseValue(
            origin, name, valueString, forKey: key, originalError: nil)
        }
        return value
      },
      initial: initial,
      completion: completion ?? Container.Contained.defaultCompletionKind)
  }

  init<Container>(
    container: Container.Type,
    key: InputKey,
    kind: ArgumentDefinition.Kind,
    help: ArgumentHelp?,
    parsingStrategy: ParsingStrategy,
    transform: @escaping (String) throws -> Container.Contained,
    initial: Container.Initial?,
    completion: CompletionKind?
  ) where Container: ArgumentDefinitionContainer {
    self.init(
      container: Container.self,
      key: key,
      kind: kind,
      allValueStrings: [],
      help: help,
      defaultValueDescription: nil,
      parsingStrategy: parsingStrategy,
      parser: { (key, origin, name, valueString) -> Container.Contained in
        do {
          return try transform(valueString)
        } catch {
          throw ParserError.unableToParseValue(
            origin, name, valueString, forKey: key, originalError: error)
        }
      },
      initial: initial,
      completion: completion)
  }

  private init<Container>(
    container: Container.Type,
    key: InputKey,
    kind: ArgumentDefinition.Kind,
    allValueStrings: [String],
    help: ArgumentHelp?,
    defaultValueDescription: String?,
    parsingStrategy: ParsingStrategy,
    parser: @escaping (InputKey, InputOrigin, Name?, String) throws -> Container.Contained,
    initial: Container.Initial?,
    completion: CompletionKind?
  ) where Container: ArgumentDefinitionContainer {
    self.init(
      kind: kind,
      help: .init(
        allValueStrings: allValueStrings,
        options: Container.helpOptions.union(initial != nil ? [.isOptional] : []),
        help: help,
        defaultValue: defaultValueDescription,
        key: key,
        isComposite: false),
      completion: completion ?? .default,
      parsingStrategy: parsingStrategy,
      update: .unary({ (origin, name, valueString, parsedValues) in
        let value = try parser(key, origin, name, valueString)
        Container.update(
          parsedValues: &parsedValues,
          value: value,
          key: key,
          origin: origin)
      }),
      initial: { origin, values in
        let inputOrigin: InputOrigin
        switch kind {
        case .default:
          inputOrigin = InputOrigin(element: .defaultValue)
        case .named, .positional:
          inputOrigin = origin
        }
        values.set(initial, forKey: key, inputOrigin: inputOrigin)
      })
  }
}

// MARK: - Abstraction over T, Option<T>, Array<T>
protocol ArgumentDefinitionContainer {
  associatedtype Contained
  associatedtype Initial

  static var helpOptions: ArgumentDefinition.Help.Options { get }
  static func update(
    parsedValues: inout ParsedValues,
    value: Contained,
    key: InputKey,
    origin: InputOrigin)
}

protocol ArgumentDefinitionContainerExpressibleByArgument:
  ArgumentDefinitionContainer where Contained: ExpressibleByArgument {
  static func defaultValueDescription(_ initial: Initial?) -> String?
}

enum Bare<T> { }

extension Bare: ArgumentDefinitionContainer {
  typealias Contained = T
  typealias Initial = T

  static var helpOptions: ArgumentDefinition.Help.Options { [] }

  static func update(
    parsedValues: inout ParsedValues,
    value: Contained,
    key: InputKey,
    origin: InputOrigin
  ) {
    parsedValues.set(value, forKey: key, inputOrigin: origin)
  }
}

extension Bare: ArgumentDefinitionContainerExpressibleByArgument
where Contained: ExpressibleByArgument {
  static func defaultValueDescription(_ initial: T?) -> String? {
    guard let initial = initial else { return nil }
    return initial.defaultValueDescription
  }
}

extension Optional: ArgumentDefinitionContainer {
  typealias Contained = Wrapped
  typealias Initial = Wrapped

  static var helpOptions: ArgumentDefinition.Help.Options { [.isOptional] }

  static func update(
    parsedValues: inout ParsedValues,
    value: Contained,
    key: InputKey,
    origin: InputOrigin
  ) {
    parsedValues.set(value, forKey: key, inputOrigin: origin)
  }
}

extension Optional: ArgumentDefinitionContainerExpressibleByArgument
where Contained: ExpressibleByArgument {
  static func defaultValueDescription(_ initial: Initial?) -> String? {
    guard let initial = initial else { return nil }
    return initial.defaultValueDescription
  }
}

extension Array: ArgumentDefinitionContainer {
  typealias Contained = Element
  typealias Initial = Array<Element>

  static var helpOptions: ArgumentDefinition.Help.Options { [.isRepeating] }

  static func update(
    parsedValues: inout ParsedValues,
    value: Element,
    key: InputKey,
    origin: InputOrigin
  ) {
    parsedValues.update(
      forKey: key,
      inputOrigin: origin,
      initial: .init(),
      closure: { $0.append(value) })
  }
}

extension Array: ArgumentDefinitionContainerExpressibleByArgument
where Element: ExpressibleByArgument {
  static func defaultValueDescription(_ initial: Array<Element>?) -> String? {
    guard let initial = initial else { return nil }
    guard !initial.isEmpty else { return nil }
    return initial
      .lazy
      .map { $0.defaultValueDescription }
      .joined(separator: ", ")
  }
}
