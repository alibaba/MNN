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

/// A property wrapper that represents a positional command-line argument.
///
/// Use the `@Argument` wrapper to define a property of your custom command as
/// a positional argument. A *positional argument* for a command-line tool is
/// specified without a label and must appear in declaration order. `@Argument`
/// properties with `Optional` type or a default value are optional for the user
/// of your command-line tool.
///
/// For example, the following program has two positional arguments. The `name`
/// argument is required, while `greeting` is optional because it has a default
/// value.
///
/// ```swift
/// @main
/// struct Greet: ParsableCommand {
///     @Argument var name: String
///     @Argument var greeting: String = "Hello"
///
///     mutating func run() {
///         print("\(greeting) \(name)!")
///     }
/// }
/// ```
///
/// You can call this program with just a name or with a name and a
/// greeting. When you supply both arguments, the first argument is always
/// treated as the name, due to the order of the property declarations.
///
///     $ greet Nadia
///     Hello Nadia!
///     $ greet Tamara Hi
///     Hi Tamara!
@propertyWrapper
public struct Argument<Value>:
  Decodable, ParsedWrapper
{
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
  /// @Argument() var foo: String // Syntax without this initializer
  /// @Argument var foo: String   // Syntax with this initializer
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

extension Argument: CustomStringConvertible {
  public var description: String {
    switch _parsedValue {
    case .value(let v):
      return String(describing: v)
    case .definition:
      return "Argument(*definition*)"
    }
  }
}

extension Argument: Sendable where Value: Sendable { }
extension Argument: DecodableParsedWrapper where Value: Decodable { }

/// The strategy to use when parsing multiple values from positional arguments
/// into an array.
public struct ArgumentArrayParsingStrategy: Hashable {
  internal var base: ArgumentDefinition.ParsingStrategy
  
  /// Parse only unprefixed values from the command-line input, ignoring
  /// any inputs that have a dash prefix. This is the default strategy.
  ///
  /// `remaining` is the default parsing strategy for argument arrays.
  ///
  /// For example, the `Example` command defined below has a `words` array that
  /// uses the `remaining` parsing strategy:
  ///
  ///     @main
  ///     struct Example: ParsableCommand {
  ///         @Flag var verbose = false
  ///
  ///         @Argument(parsing: .remaining)
  ///         var words: [String]
  ///
  ///         func run() {
  ///             print(words.joined(separator: "\n"))
  ///         }
  ///     }
  ///
  /// Any non-dash-prefixed inputs will be captured in the `words` array.
  ///
  /// ```
  /// $ example --verbose one two
  /// one
  /// two
  /// $ example one two --verbose
  /// one
  /// two
  /// $ example one two --other
  /// Error: Unknown option '--other'
  /// ```
  ///
  /// If a user uses the `--` terminator in their input, all following inputs
  /// will be captured in `words`.
  ///
  /// ```
  /// $ example one two -- --verbose --other
  /// one
  /// two
  /// --verbose
  /// --other
  /// ```
  public static var remaining: ArgumentArrayParsingStrategy {
    self.init(base: .default)
  }
  
  /// After parsing, capture all unrecognized inputs in this argument array.
  ///
  /// You can use the `allUnrecognized` parsing strategy to suppress
  /// "unexpected argument" errors or to capture unrecognized inputs for further
  /// processing.
  ///
  /// For example, the `Example` command defined below has an `other` array that
  /// uses the `allUnrecognized` parsing strategy:
  ///
  ///     @main
  ///     struct Example: ParsableCommand {
  ///         @Flag var verbose = false
  ///         @Argument var name: String
  ///
  ///         @Argument(parsing: .allUnrecognized)
  ///         var other: [String]
  ///
  ///         func run() {
  ///             print(other.joined(separator: "\n"))
  ///         }
  ///     }
  ///
  /// After parsing the `--verbose` flag and `<name>` argument, any remaining
  /// input is captured in the `other` array.
  ///
  /// ```
  /// $ example --verbose Negin one two
  /// one
  /// two
  /// $ example Asa --verbose --other -zzz
  /// --other
  /// -zzz
  /// ```
  public static var allUnrecognized: ArgumentArrayParsingStrategy {
    self.init(base: .allUnrecognized)
  }
  
  /// Before parsing arguments, capture all inputs that follow the `--`
  /// terminator in this argument array.
  ///
  /// For example, the `Example` command defined below has a `words` array that
  /// uses the `postTerminator` parsing strategy:
  ///
  ///     @main
  ///     struct Example: ParsableCommand {
  ///         @Flag var verbose = false
  ///         @Argument var name = ""
  ///
  ///         @Argument(parsing: .postTerminator)
  ///         var words: [String]
  ///
  ///         func run() {
  ///             print(words.joined(separator: "\n"))
  ///         }
  ///     }
  ///
  /// Before looking for the `--verbose` flag and `<name>` argument, any inputs
  /// after the `--` terminator are captured into the `words` array.
  ///
  /// ```
  /// $ example --verbose Asa -- one two --other
  /// one
  /// two
  /// --other
  /// $ example Asa Extra -- one two --other
  /// Error: Unexpected argument 'Extra'
  /// ```
  ///
  /// Because options are parsed before arguments, an option that consumes or
  /// suppresses the `--` terminator can prevent a `postTerminator` argument
  /// array from capturing any input. In particular, the
  /// ``SingleValueParsingStrategy/unconditional``,
  /// ``ArrayParsingStrategy/unconditionalSingleValue``, and
  /// ``ArrayParsingStrategy/remaining`` parsing strategies can all consume
  /// the terminator as part of their values.
  ///
  /// - Note: This parsing strategy can be surprising for users, since it
  ///   changes the behavior of the `--` terminator. Prefer ``remaining``
  ///   whenever possible.
  public static var postTerminator: ArgumentArrayParsingStrategy {
    self.init(base: .postTerminator)
  }
  
  /// Parse all remaining inputs after parsing any known options or flags,
  /// including dash-prefixed inputs and the `--` terminator.
  ///
  /// You can use the `captureForPassthrough` parsing strategy if you need to
  /// capture a user's input to manually pass it unchanged to another command.
  ///
  /// When you use this parsing strategy, the parser stops parsing flags and
  /// options as soon as it encounters a positional argument or an unrecognized
  /// flag, and captures all remaining inputs in the array argument.
  ///
  /// For example, the `Example` command defined below has an `words` array that
  /// uses the `captureForPassthrough` parsing strategy:
  ///
  ///     @main
  ///     struct Example: ParsableCommand {
  ///         @Flag var verbose = false
  ///
  ///         @Argument(parsing: .captureForPassthrough)
  ///         var words: [String] = []
  ///
  ///         func run() {
  ///             print(words.joined(separator: "\n"))
  ///         }
  ///     }
  ///
  /// Any values after the first unrecognized input are captured in the `words`
  /// array.
  ///
  /// ```
  /// $ example --verbose one two --other
  /// one
  /// two
  /// --other
  /// $ example one two --verbose
  /// one
  /// two
  /// --verbose
  /// ```
  ///
  /// With the `captureForPassthrough` parsing strategy, the `--` terminator
  /// is included in the captured values.
  ///
  /// ```
  /// $ example --verbose one two -- --other
  /// one
  /// two
  /// --
  /// --other
  /// ```
  ///
  /// - Note: This parsing strategy can be surprising for users, particularly
  ///   when combined with options and flags. Prefer ``remaining`` or
  ///   ``allUnrecognized`` whenever possible, since users can always terminate
  ///   options and flags with the `--` terminator. With the `remaining`
  ///   parsing strategy, the input `--verbose -- one two --other` would have
  ///   the same result as the first example above.
  public static var captureForPassthrough: ArgumentArrayParsingStrategy {
    self.init(base: .allRemainingInput)
  }
  
  @available(*, deprecated, renamed: "captureForPassthrough")
  public static var unconditionalRemaining: ArgumentArrayParsingStrategy {
    .captureForPassthrough
  }
}

extension ArgumentArrayParsingStrategy: Sendable { }

// MARK: - @Argument T: ExpressibleByArgument Initializers
extension Argument where Value: ExpressibleByArgument {
  /// Creates a property with a default value provided by standard Swift default
  /// value syntax.
  ///
  /// This method is called to initialize an `Argument` with a default value
  /// such as:
  /// ```swift
  /// @Argument var foo: String = "bar"
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///   implicitly by the compiler during property wrapper initialization.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init(
    wrappedValue: Value,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a property with no default value.
  ///
  /// This method is called to initialize an `Argument` without a default value
  /// such as:
  /// ```swift
  /// @Argument var foo: String
  /// ```
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init(
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Argument T Initializers
extension Argument {
  /// Creates a property with a default value provided by standard Swift default
  /// value syntax, parsing with the given closure.
  ///
  /// This method is called to initialize an `Argument` with a default value
  /// such as:
  /// ```swift
  /// @Argument(transform: baz)
  /// var foo: String = "bar"
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided
  ///     implicitly by the compiler during property wrapper initialization.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's type
  ///     or throws an error.
  @preconcurrency
  public init(
    wrappedValue: Value,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> Value
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        transform: transform,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a property with no default value, parsing with the given closure.
  ///
  /// This method is called to initialize an `Argument` with no default value such as:
  /// ```swift
  /// @Argument(transform: baz)
  /// var foo: String
  /// ```
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type or throws an error.
  @preconcurrency
  @_disfavoredOverload
  public init(
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> Value
  ) {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Bare<Value>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Argument Optional<T: ExpressibleByArgument> Initializers
extension Argument {
  /// This initializer allows a user to provide a `nil` default value for an
  /// optional `@Argument`-marked property without allowing a non-`nil` default
  /// value.
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init<T>(
    wrappedValue _value: _OptionalNilComparisonType,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  @available(*, deprecated, message: """
    Optional @Arguments with default values should be declared as non-Optional.
    """)
  @_disfavoredOverload
  public init<T>(
    wrappedValue _wrappedValue: Optional<T>,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        initial: _wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates an optional property that reads its value from an argument.
  ///
  /// The argument is optional for the caller of the command and defaults to
  /// `nil`.
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init<T>(
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Argument Optional<T> Initializers
extension Argument {
  /// This initializer allows a user to provide a `nil` default value for an
  /// optional `@Argument`-marked property without allowing a non-`nil` default
  /// value.
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type or throws an error.
  @preconcurrency
  public init<T>(
    wrappedValue _value: _OptionalNilComparisonType,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  @available(*, deprecated, message: """
    Optional @Arguments with default values should be declared as non-Optional.
    """)
  @_disfavoredOverload
  @preconcurrency
  public init<T>(
    wrappedValue _wrappedValue: Optional<T>,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        transform: transform,
        initial: _wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates an optional property that reads its value from an argument.
  ///
  /// The argument is optional for the caller of the command and defaults to
  /// `nil`.
  ///
  /// - Parameters:
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type or throws an error.
  @preconcurrency
  public init<T>(
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Optional<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Optional<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: .default,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Argument Array<T: ExpressibleByArgument> Initializers
extension Argument {
  /// Creates a property that reads an array from zero or more arguments.
  ///
  /// - Parameters:
  ///   - initial: A default value to use for this property.
  ///   - parsingStrategy: The behavior to use when parsing multiple values from
  ///     the command-line arguments.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init<T>(
    wrappedValue: Array<T>,
    parsing parsingStrategy: ArgumentArrayParsingStrategy = .remaining,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a property with no default value that reads an array from zero or
  /// more arguments.
  ///
  /// This method is called to initialize an array `Argument` with no default
  /// value such as:
  /// ```swift
  /// @Argument()
  /// var foo: [String]
  /// ```
  ///
  /// - Parameters:
  ///   - parsingStrategy: The behavior to use when parsing multiple values from
  ///   the command-line arguments.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  public init<T>(
    parsing parsingStrategy: ArgumentArrayParsingStrategy = .remaining,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil
  ) where T: ExpressibleByArgument, Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: parsingStrategy.base,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}

// MARK: - @Argument Array<T> Initializers
extension Argument {
  /// Creates a property that reads an array from zero or more arguments,
  /// parsing each element with the given closure.
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property.
  ///   - parsingStrategy: The behavior to use when parsing multiple values from
  ///     the command-line arguments.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type or throws an error.
  @preconcurrency
  public init<T>(
    wrappedValue: Array<T>,
    parsing parsingStrategy: ArgumentArrayParsingStrategy = .remaining,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: wrappedValue,
        completion: completion)

      return ArgumentSet(arg)
    })
  }

  /// Creates a property with no default value that reads an array from zero or
  /// more arguments, parsing each element with the given closure.
  ///
  /// This method is called to initialize an array `Argument` with no default
  /// value such as:
  /// ```swift
  /// @Argument(transform: baz)
  /// var foo: [String]
  /// ```
  ///
  /// - Parameters:
  ///   - parsingStrategy: The behavior to use when parsing multiple values from
  ///     the command-line arguments.
  ///   - help: Information about how to use this argument.
  ///   - completion: Kind of completion provided to the user for this option.
  ///   - transform: A closure that converts a string into this property's
  ///     element type or throws an error.
  @preconcurrency
  public init<T>(
    parsing parsingStrategy: ArgumentArrayParsingStrategy = .remaining,
    help: ArgumentHelp? = nil,
    completion: CompletionKind? = nil,
    transform: @Sendable @escaping (String) throws -> T
  ) where Value == Array<T> {
    self.init(_parsedValue: .init { key in
      let arg = ArgumentDefinition(
        container: Array<T>.self,
        key: key,
        kind: .positional,
        help: help,
        parsingStrategy: parsingStrategy.base,
        transform: transform,
        initial: nil,
        completion: completion)

      return ArgumentSet(arg)
    })
  }
}
