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

/// A property wrapper that represents a command-line option.
///
/// Use the `@Option` wrapper to define a property of your custom command as a
/// command-line option. An *option* is a named value passed to a command-line
/// tool, like `--configuration debug`. Options can be specified in any order.
///
/// An option can have a default value specified as part of its
/// declaration; options with optional `Value` types implicitly have `nil` as
/// their default value. Options that are neither declared as `Optional` nor
/// given a default value are required for users of your command-line tool.
///
/// For example, the following program defines three options:
///
/// ```swift
/// @main
/// struct Greet: ParsableCommand {
///     @Option var greeting = "Hello"
///     @Option var age: Int? = nil
///     @Option var name: String
///
///     mutating func run() {
///         print("\(greeting) \(name)!")
///         if let age {
///             print("Congrats on making it to the ripe old age of \(age)!")
///         }
///     }
/// }
/// ```
///
/// `greeting` has a default value of `"Hello"`, which can be overridden by
/// providing a different string as an argument, while `age` defaults to `nil`.
/// `name` is a required option because it is non-`nil` and has no default
/// value.
///
///     $ greet --name Alicia
///     Hello Alicia!
///     $ greet --age 28 --name Seungchin --greeting Hi
///     Hi Seungchin!
///     Congrats on making it to the ripe old age of 28!
@propertyWrapper
public struct Option<Value>: Decodable, ParsedWrapper {
  internal var _parsedValue: Parsed<Value>
  
  internal init(_parsedValue: Parsed<Value>) {
    self._parsedValue = _parsedValue
  }
  
  public init(from _decoder: Decoder) throws {
    try self.init(_decoder: _decoder)
  }

  /// This initializer works around a quirk of property wrappers, where the
  /// compiler will not see no-argument initializers in extensions. Explicitly
  /// marking this initializer unavailable means that when `Value` conforms to
  /// `ExpressibleByArgument`, that overload will be selected instead.
  ///
  /// ```swift
  /// @Option() var foo: String // Syntax without this initializer
  /// @Option var foo: String   // Syntax with this initializer
  /// ```
  @available(*, unavailable, message: "A default value must be provided unless the value type conforms to ExpressibleByArgument.")
  public init() {
    fatalError("unavailable")
  }

  /// The value presented by this property wrapper.
  public var wrappedValue: Value {
    get {
      switch _parsedValue {
      case .value(let v):
        return v
      case .definition:
        fatalError(directlyInitializedError)
      }
    }
    set {
      _parsedValue = .value(newValue)
    }
  }
}

extension Option: CustomStringConvertible {
  public var description: String {
    switch _parsedValue {
    case .value(let v):
      return String(describing: v)
    case .definition:
      return "Option(*definition*)"
    }
  }
}

extension Option: Sendable where Value: Sendable {}
extension Option: DecodableParsedWrapper where Value: Decodable {}

/// The strategy to use when parsing a single value from `@Option` arguments.
///
/// - SeeAlso: ``ArrayParsingStrategy``
public struct SingleValueParsingStrategy: Hashable {
  internal var base: ArgumentDefinition.ParsingStrategy
  
  /// Parse the input after the option. Expect it to be a value.
  ///
  /// For inputs such as `--foo foo`, this would parse `foo` as the
  /// value. However, the input `--foo --bar foo bar` would
  /// result in an error. Even though two values are provided, they don’t
  /// succeed each option. Parsing would result in an error such as the following:
  ///
  ///     Error: Missing value for '--foo <foo>'
  ///     Usage: command [--foo <foo>]
  ///
  /// This is the **default behavior** for `@Option`-wrapped properties.
  public static var next: SingleValueParsingStrategy {
    self.init(base: .default)
  }
  
  /// Parse the next input, even if it could be interpreted as an option or
  /// flag.
  ///
  /// For inputs such as `--foo --bar baz`, if `.unconditional` is used for `foo`,
  /// this would read `--bar` as the value for `foo` and would use `baz` as
  /// the next positional argument.
  ///
  /// This allows reading negative numeric values or capturing flags to be
  /// passed through to another program since the leading hyphen is normally
  /// interpreted as the start of another option.
  ///
  /// - Note: This is usually *not* what users would expect. Use with caution.
  public static var unconditional: SingleValueParsingStrategy {
    self.init(base: .unconditional)
  }
  
  /// Parse the next input, as long as that input can't be interpreted as
  /// an option or flag.
  ///
  /// - Note: This will skip other options and _read ahead_ in the input
  /// to find the next available value. This may be *unexpected* for users.
  /// Use with caution.
  ///
  /// For example, if `--foo` takes a value, then the input `--foo --bar bar`
  /// would be parsed such that the value `bar` is used for `--foo`.
  public static var scanningForValue: SingleValueParsingStrategy {
    self.init(base: .scanningForValue)
  }
}

extension SingleValueParsingStrategy: Sendable { }

/// The strategy to use when parsing multiple values from `@Option` arguments into an
/// array.
public struct ArrayParsingStrategy: Hashable {
  internal var base: ArgumentDefinition.ParsingStrategy
  
  /// Parse one value per option, joining multiple into an array.
  ///
  /// For example, for a parsable type with a property defined as
  /// `@Option(parsing: .singleValue) var read: [String]`,
  /// the input `--read foo --read bar` would result in the array
  /// `["foo", "bar"]`. The same would be true for the input
  /// `--read=foo --read=bar`.
  ///
  /// - Note: This follows the default behavior of differentiating between values and options. As
  ///     such, the value for this option will be the next value (non-option) in the input. For the
  ///     above example, the input `--read --name Foo Bar` would parse `Foo` into
  ///     `read` (and `Bar` into `name`).
  public static var singleValue: ArrayParsingStrategy {
    self.init(base: .default)
  }
  
  /// Parse the value immediately after the option while allowing repeating options, joining multiple into an array.
  ///
  /// This is identical to `.singleValue` except that the value will be read
  /// from the input immediately after the option, even if it could be interpreted as an option.
  ///
  /// For example, for a parsable type with a property defined as
  /// `@Option(parsing: .unconditionalSingleValue) var read: [String]`,
  /// the input `--read foo --read bar` would result in the array
  /// `["foo", "bar"]` -- just as it would have been the case for `.singleValue`.
  ///
  /// - Note: However, the input `--read --name Foo Bar --read baz` would result in
  /// `read` being set to the array `["--name", "baz"]`. This is usually *not* what users
  /// would expect. Use with caution.
  public static var unconditionalSingleValue: ArrayParsingStrategy {
    self.init(base: .unconditional)
  }
  
  /// Parse all values up to the next option.
  ///
  /// For example, for a parsable type with a property defined as
  /// `@Option(parsing: .upToNextOption) var files: [String]`,
  /// the input `--files foo bar` would result in the array
  /// `["foo", "bar"]`.
  ///
  /// Parsing stops as soon as there’s another option in the input such that
  /// `--files foo bar --verbose` would also set `files` to the array
  /// `["foo", "bar"]`.
  public static var upToNextOption: ArrayParsingStrategy {
    self.init(base: .upToNextOption)
  }

  /// Parse all remaining arguments into an array.
  ///
  /// `.remaining` can be used for capturing pass-through flags. For example, for
  /// a parsable type defined as
  /// `@Option(parsing: .remaining) var passthrough: [String]`:
  ///
  ///     $ cmd --passthrough --foo 1 --bar 2 -xvf
  ///     ------------
  ///     options.passthrough == ["--foo", "1", "--bar", "2", "-xvf"]
  ///
  /// - Note: This will read all inputs following the option without attempting to do any parsing. This is
  /// usually *not* what users would expect. Use with caution.
  ///
  /// Consider using a trailing `@Argument` instead and letting users explicitly turn off parsing
  /// through the terminator `--`. That is the more common approach. For example:
  /// ```swift
  /// struct Options: ParsableArguments {
  ///     @Option var title: String
  ///     @Argument var remainder: [String]
  /// }
  /// ```
  /// would parse the input `--title Foo -- Bar --baz` such that the `remainder`
  /// would hold the value `["Bar", "--baz"]`.
  public static var remaining: ArrayParsingStrategy {
    self.init(base: .allRemainingInput)
  }
}

extension ArrayParsingStrategy: Sendable { }

// MARK: - @Option T: ExpressibleByArgument Initializers
extension Option where Value: ExpressibleByArgument {
  /// Creates a property with a default value that reads its value from a
  /// labeled option.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// that has an `ExpressibleByArgument` type, providing a default value:
  ///
  /// ```swift
  /// @Option var title: String = "<Title>"
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  public init(
    wrappedValue: Value,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  @available(*, deprecated, message: """
    Swap the order of the 'help' and 'completion' arguments.
    """)
  public init(
    wrappedValue _wrappedValue: Value,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    completion: CompletionKind?,
    help: ArgumentHelp?
  ) {
    self.init(
      wrappedValue: _wrappedValue,
      name: name,
      parsing: parsingStrategy,
      help: help,
      completion: completion)
  }

  /// Creates a required property that reads its value from a labeled option.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// that has an `ExpressibleByArgument` type, but without a default value:
  ///
  /// ```swift
  /// @Option var title: String
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  public init(
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Option T Initializers
extension Option {
  /// Creates a property with a default value that reads its value from a
  /// labeled option, parsing with the given closure.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// with a transform closure and a default value:
  ///
  /// ```swift
  /// @Option(transform: { $0.first ?? " " })
  /// var char: Character = "_"
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: The default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     type, or else throws an error.
  @preconcurrency
  public init(
    wrappedValue: Value,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> Value
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a required property that reads its value from a labeled option,
  /// parsing with the given closure.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// with a transform closure and without a default value:
  ///
  /// ```swift
  /// @Option(transform: { $0.first ?? " " })
  /// var char: Character
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     type, or else throws an error.
  @preconcurrency
  @_disfavoredOverload
  public init(
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> Value
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Option Optional<T: ExpressibleByArgument> Initializers
extension Option {
  /// Creates an optional property that reads its value from a labeled option,
  /// with an explicit `nil` default.
  ///
  /// This initializer allows a user to provide a `nil` default value for an
  /// optional `@Option`-marked property:
  ///
  /// ```swift
  /// @Option var count: Int? = nil
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  public init<T>(
    wrappedValue _value: _OptionalNilComparisonType,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  @available(*, deprecated, message: """
    Optional @Options with default values should be declared as non-Optional.
    """)
  @_disfavoredOverload
  public init<T>(
    wrappedValue _wrappedValue: Optional<T>,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: _wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates an optional property that reads its value from a labeled option.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// with an optional type and no default value:
  ///
  /// ```swift
  /// @Option var count: Int?
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  public init<T>(
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Option Optional<T> Initializers
extension Option {
  /// Creates an optional property that reads its value from a labeled option,
  /// parsing with the given closure, with an explicit `nil` default.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// with a transform closure and with a default value of `nil`:
  ///
  /// ```swift
  /// @Option(transform: { $0.first ?? " " })
  /// var char: Character? = nil
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     type, or else throws an error.
  @preconcurrency
  public init<T>(
    wrappedValue _value: _OptionalNilComparisonType,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  @available(*, deprecated, message: """
    Optional @Options with default values should be declared as non-Optional.
    """)
  @_disfavoredOverload
  @preconcurrency
  public init<T>(
    wrappedValue _wrappedValue: Optional<T>,
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: _wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates an optional property that reads its value from a labeled option,
  /// parsing with the given closure.
  ///
  /// This initializer is used when you declare an `@Option`-attributed property
  /// with a transform closure and without a default value:
  ///
  /// ```swift
  /// @Option(transform: { $0.first ?? " " })
  /// var char: Character?
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when looking for this option's
  ///     value.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     type, or else throws an error.
  @preconcurrency
  public init<T>(
    name: NameSpecification = .long,
    parsing parsingStrategy: SingleValueParsingStrategy = .next,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Option Array<T: ExpressibleByArgument> Initializers
extension Option {
  /// Creates an array property that reads its values from zero or
  /// more labeled options.
  ///
  /// This initializer is used when you declare an `@Option`-attributed array
  /// property with a default value:
  ///
  /// ```swift
  /// @Option(name: .customLong("char"))
  /// var chars: [Character] = []
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///     If this initial value is non-empty, elements passed from the command
  ///     line are appended to the original contents.
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when parsing the elements for
  ///     this option.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type, or else throws an error.
  public init<T>(
    wrappedValue: Array<T>,
    name: NameSpecification = .long,
    parsing parsingStrategy: ArrayParsingStrategy = .singleValue,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a required array property that reads its values from zero or
  /// more labeled options.
  ///
  /// This initializer is used when you declare an `@Option`-attributed array
  /// property without a default value:
  ///
  /// ```swift
  /// @Option(name: .customLong("char"))
  /// var chars: [Character]
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when parsing the elements for
  ///     this option.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  public init<T>(
    name: NameSpecification = .long,
    parsing parsingStrategy: ArrayParsingStrategy = .singleValue,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Option Array<T> Initializers
extension Option {
  /// Creates an array property that reads its values from zero or
  /// more labeled options, parsing each element with the given closure.
  ///
  /// This initializer is used when you declare an `@Option`-attributed array
  /// property with a transform closure and a default value:
  ///
  /// ```swift
  /// @Option(name: .customLong("char"), transform: { $0.first ?? " " })
  /// var chars: [Character] = []
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///     If this initial value is non-empty, elements passed from the command
  ///     line are appended to the original contents.
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when parsing the elements for
  ///     this option.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type, or else throws an error.
  @preconcurrency
  public init<T>(
    wrappedValue: Array<T>,
    name: NameSpecification = .long,
    parsing parsingStrategy: ArrayParsingStrategy = .singleValue,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a required array property that reads its values from zero or
  /// more labeled options, parsing each element with the given closure.
  ///
  /// This initializer is used when you declare an `@Option`-attributed array
  /// property with a transform closure and without a default value:
  ///
  /// ```swift
  /// @Option(name: .customLong("char"), transform: { $0.first ?? " " })
  /// var chars: [Character]
  /// ```
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this option.
  ///   - parsingStrategy: The behavior to use when parsing the elements for
  ///     this option.
  ///   - help: Information about how to use this option.
  ///   - completion: The type of command-line completion provided for this
  ///     option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type, or else throws an error.
  @preconcurrency
  public init<T>(
    name: NameSpecification = .long,
    parsing parsingStrategy: ArrayParsingStrategy = .singleValue,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .name(key: key, specification: name),
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}
