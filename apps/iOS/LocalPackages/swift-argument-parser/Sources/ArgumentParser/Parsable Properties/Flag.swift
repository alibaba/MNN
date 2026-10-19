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

/// A property wrapper that represents a command-line flag.
///
/// Use the `@Flag` wrapper to define a property of your custom type as a
/// command-line flag. A *flag* is a dash-prefixed label that can be provided on
/// the command line, such as `-d` and `--debug`.
///
/// For example, the following program declares a flag that lets a user indicate
/// that seconds should be included when printing the time.
///
/// ```swift
/// @main
/// struct Time: ParsableCommand {
///     @Flag var includeSeconds = false
///
///     mutating func run() {
///         if includeSeconds {
///             print(Date.now.formatted(.dateTime.hour().minute().second()))
///         } else {
///             print(Date.now.formatted(.dateTime.hour().minute()))
///         }
///     }
/// }
/// ```
///
/// `includeSeconds` has a default value of `false`, but becomes `true` if
/// `--include-seconds` is provided on the command line.
///
///     $ time
///     11:09 AM
///     $ time --include-seconds
///     11:09:15 AM
///
/// A flag can have a value that is a `Bool`, an `Int`, or any `EnumerableFlag`
/// type. When using an `EnumerableFlag` type as a flag, the individual cases
/// form the flags that are used on the command line.
///
///     @main
///     struct Math: ParsableCommand {
///         enum Operation: EnumerableFlag {
///             case add
///             case multiply
///         }
///
///         @Flag var operation: Operation
///
///         mutating func run() {
///             print("Time to \(operation)!")
///         }
///     }
///
/// Instead of using the name of the `operation` property as the flag in this
/// case, the two cases of the `Operation` enumeration become valid flags.
/// The `operation` property is neither optional nor given a default value, so
/// one of the two flags is required.
///
///     $ math --add
///     Time to add!
///     $ math
///     Error: Missing one of: '--add', '--multiply'
@propertyWrapper
public struct Flag<Value>: Decodable, ParsedWrapper {
  internal var _parsedValue: Parsed<Value>

  internal init(_parsedValue: Parsed<Value>) {
    self._parsedValue = _parsedValue
  }
  
  public init(from _decoder: Decoder) throws {
    try self.init(_decoder: _decoder)
  }

  /// This initializer works around a quirk of property wrappers, where the
  /// compiler will not see no-argument initializers in extensions. Explicitly
  /// marking this initializer unavailable means that when `Value` is a type
  /// supported by `Flag` like `Bool` or `EnumerableFlag`, the appropriate
  /// overload will be selected instead.
  ///
  /// ```swift
  /// @Flag() var flag: Bool  // Syntax without this initializer
  /// @Flag var flag: Bool    // Syntax with this initializer
  /// ```
  @available(*, unavailable, message: "A default value must be provided unless the value type is supported by Flag.")
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

extension Flag: Sendable where Value: Sendable {}

extension Flag: CustomStringConvertible {
  public var description: String {
    switch _parsedValue {
    case .value(let v):
      return String(describing: v)
    case .definition:
      return "Flag(*definition*)"
    }
  }
}

extension Flag: DecodableParsedWrapper where Value: Decodable {}

/// The options for converting a Boolean flag into a `true`/`false` pair.
public struct FlagInversion: Hashable {
  internal enum Representation {
    case prefixedNo
    case prefixedEnableDisable
  }
  
  internal var base: Representation
  
  /// Adds a matching flag with a `no-` prefix to represent `false`.
  ///
  /// For example, the `shouldRender` property in this declaration is set to
  /// `true` when a user provides `--render` and to `false` when the user
  /// provides `--no-render`:
  ///
  ///     @Flag(name: .customLong("render"), inversion: .prefixedNo)
  ///     var shouldRender: Bool
  public static var prefixedNo: FlagInversion {
    self.init(base: .prefixedNo)
  }
  
  /// Uses matching flags with `enable-` and `disable-` prefixes.
  ///
  /// For example, the `extraOutput` property in this declaration is set to
  /// `true` when a user provides `--enable-extra-output` and to `false` when
  /// the user provides `--disable-extra-output`:
  ///
  ///     @Flag(inversion: .prefixedEnableDisable)
  ///     var extraOutput: Bool
  public static var prefixedEnableDisable: FlagInversion {
    self.init(base: .prefixedEnableDisable)
  }
}

extension FlagInversion: Sendable { }

/// The options for treating enumeration-based flags as exclusive.
public struct FlagExclusivity: Hashable {
  internal enum Representation {
    case exclusive
    case chooseFirst
    case chooseLast
  }
  
  internal var base: Representation
  
  /// Only one of the enumeration cases may be provided.
  public static var exclusive: FlagExclusivity {
    self.init(base: .exclusive)
  }
  
  /// The first enumeration case that is provided is used.
  public static var chooseFirst: FlagExclusivity {
    self.init(base: .chooseFirst)
  }
  
  /// The last enumeration case that is provided is used.
  public static var chooseLast: FlagExclusivity {
    self.init(base: .chooseLast)
  }
}

extension FlagExclusivity: Sendable { }

extension Flag where Value == Optional<Bool> {
  /// Creates a Boolean property that reads its value from the presence of
  /// one or more inverted flags.
  ///
  /// Use this initializer to create an optional Boolean flag with an on/off
  /// pair. With the following declaration, for example, the user can specify
  /// either `--use-https` or `--no-use-https` to set the `useHTTPS` flag to
  /// `true` or `false`, respectively. If neither is specified, the resulting
  /// flag value would be `nil`.
  ///
  ///     @Flag(inversion: .prefixedNo)
  ///     var useHTTPS: Bool?
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this flag.
  ///   - inversion: The method for converting this flags name into an on/off
  ///     pair.
  ///   - exclusivity: The behavior to use when an on/off pair of flags is
  ///     specified.
  ///   - help: Information about how to use this flag.
  public init(
    name: NameSpecification = .long,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity = .chooseLast,
    help: ArgumentHelp? = nil
  ) {
    self.init(_parsedValue: .init { key in
      .flag(
        key: key,
        name: name,
        default: nil,
        required: false,
        inversion: inversion,
        exclusivity: exclusivity,
        help: help)
    })
  }
  
  /// This initializer allows a user to provide a `nil` default value for
  /// `@Flag`-marked `Optional<Bool>` property without allowing a non-`nil`
  /// default value.
  public init(
    wrappedValue _value: _OptionalNilComparisonType,
    name: NameSpecification = .long,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity = .chooseLast,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      name: name,
      inversion: inversion,
      exclusivity: exclusivity,
      help: help)
  }

}

extension Flag where Value == Bool {
  /// Creates a Boolean property with an optional default value, intended to be called by other constructors to centralize logic.
  ///
  /// This private `init` allows us to expose multiple other similar constructors to allow for standard default property initialization while reducing code duplication.
  private init(
    name: NameSpecification,
    initial: Bool?,
    help: ArgumentHelp? = nil
  ) {
    self.init(_parsedValue: .init { key in
      .flag(key: key, name: name, default: initial, help: help)
    })
  }

  /// Creates a Boolean property with default value provided by standard Swift default value syntax that reads its value from the presence of a flag.
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided implicitly by the compiler during property wrapper initialization.
  ///   - name: A specification for what names are allowed for this flag.
  ///   - help: Information about how to use this flag.
  public init(
    wrappedValue: Bool,
    name: NameSpecification = .long,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      name: name,
      initial: wrappedValue,
      help: help
    )
  }

  /// Creates a property with an optional default value, intended to be called by other constructors to centralize logic.
  ///
  /// This private `init` allows us to expose multiple other similar constructors to allow for standard default property initialization while reducing code duplication.
  private init(
    name: NameSpecification,
    initial: Bool?,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity,
    help: ArgumentHelp?
  ) {
    self.init(_parsedValue: .init { key in
      .flag(
        key: key,
        name: name,
        default: initial,
        required: initial == nil,
        inversion: inversion,
        exclusivity: exclusivity,
        help: help)
      })
  }

  /// Creates a Boolean property with default value provided by standard Swift default value syntax that reads its value from the presence of one or more inverted flags.
  ///
  /// Use this initializer to create a Boolean flag with an on/off pair.
  /// With the following declaration, for example, the user can specify either `--use-https` or `--no-use-https` to set the `useHTTPS` flag to `true` or `false`, respectively.
  ///
  /// ```swift
  /// @Flag(inversion: .prefixedNo)
  /// var useHTTPS: Bool = true
  /// ````
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this flag.
  ///   - wrappedValue: A default value to use for this property, provided implicitly by the compiler during property wrapper initialization.
  ///   - inversion: The method for converting this flag's name into an on/off pair.
  ///   - exclusivity: The behavior to use when an on/off pair of flags is specified.
  ///   - help: Information about how to use this flag.
  public init(
    wrappedValue: Bool,
    name: NameSpecification = .long,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity = .chooseLast,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      name: name,
      initial: wrappedValue,
      inversion: inversion,
      exclusivity: exclusivity,
      help: help
    )
  }

  /// Creates a Boolean property with no default value that reads its value from the presence of one or more inverted flags.
  ///
  /// Use this initializer to create a Boolean flag with an on/off pair.
  /// With the following declaration, for example, the user can specify either `--use-https` or `--no-use-https` to set the `useHTTPS` flag to `true` or `false`, respectively.
  ///
  /// ```swift
  /// @Flag(inversion: .prefixedNo)
  /// var useHTTPS: Bool
  /// ````
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this flag.
  ///   - wrappedValue: A default value to use for this property, provided implicitly by the compiler during property wrapper initialization.
  ///   - inversion: The method for converting this flag's name into an on/off pair.
  ///   - exclusivity: The behavior to use when an on/off pair of flags is specified.
  ///   - help: Information about how to use this flag.
  public init(
    name: NameSpecification = .long,
    inversion: FlagInversion,
    exclusivity: FlagExclusivity = .chooseLast,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      name: name,
      initial: nil,
      inversion: inversion,
      exclusivity: exclusivity,
      help: help
    )
  }
}

extension Flag where Value == Int {
  /// Creates an integer property that gets its value from the number of times
  /// a flag appears.
  ///
  /// This property defaults to a value of zero.
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this flag.
  ///   - help: Information about how to use this flag.
  public init(
    name: NameSpecification = .long,
    help: ArgumentHelp? = nil
  ) {
    self.init(_parsedValue: .init { key in
      .counter(key: key, name: name, help: help)
    })
  }
}

// - MARK: EnumerableFlag

extension Flag where Value: EnumerableFlag {
  /// Creates a property with an optional default value, intended to be called by other constructors to centralize logic.
  ///
  /// This private `init` allows us to expose multiple other similar constructors to allow for standard default property initialization while reducing code duplication.
  private init(
    initial: Value?,
    exclusivity: FlagExclusivity,
    help: ArgumentHelp?
  ) {
    self.init(_parsedValue: .init { key in
      // Create a string representation of the default value. Since this is a
      // flag, the default value to show to the user is the `--value-name`
      // flag that a user would provide on the command line, not a Swift value.
      let defaultValueFlag = initial.flatMap { value -> String? in
        let defaultKey = InputKey(name: String(describing: value), parent: key)
        let defaultNames = Value.name(for: value).makeNames(defaultKey)
        return defaultNames.first?.synopsisString
      }

      let caseHelps = Value.allCases.map { Value.help(for: $0) }
      let hasCustomCaseHelp = caseHelps.contains(where: { $0 != nil })
      
      let args = Value.allCases.enumerated().map { (i, value) -> ArgumentDefinition in
        let caseKey = InputKey(name: String(describing: value), parent: key)
        let name = Value.name(for: value)
        
        let helpForCase = caseHelps[i] ?? help
        var defaultValueString: String? = nil
        if hasCustomCaseHelp {
          if value == initial {
            defaultValueString = defaultValueFlag
          }
        } else {
          defaultValueString = defaultValueFlag
        }
        
        let help = ArgumentDefinition.Help(
          allValueStrings: [],
          options: initial != nil ? .isOptional : [],
          help: helpForCase,
          defaultValue: defaultValueString,
          key: key,
          isComposite: !hasCustomCaseHelp)
        
        return ArgumentDefinition.flag(
          name: name,
          key: key,
          caseKey: caseKey,
          help: help,
          parsingStrategy: .default,
          initialValue: initial,
          update: .nullary({ (origin, name, values) in
            try ArgumentSet.updateFlag(key: key, value: value, origin: origin, values: &values, exclusivity: exclusivity)
          })
        )
      }
      return ArgumentSet(args)
      })
  }

  /// Creates a property with a default value provided by standard Swift default value syntax that gets its value from the presence of a flag.
  ///
  /// Use this initializer to customize the name and number of states further than using a `Bool`.
  /// To use, define an `EnumerableFlag` enumeration with a case for each state, and use that as the type for your flag.
  /// In this case, the user can specify either `--use-production-server` or `--use-development-server` to set the flag's value.
  ///
  /// ```swift
  /// enum ServerChoice: EnumerableFlag {
  ///   case useProductionServer
  ///   case useDevelopmentServer
  /// }
  ///
  /// @Flag var serverChoice: ServerChoice = .useProductionServer
  /// ```
  ///
  /// - Parameters:
  ///   - wrappedValue: A default value to use for this property, provided implicitly by the compiler during property wrapper initialization.
  ///   - exclusivity: The behavior to use when multiple flags are specified.
  ///   - help: Information about how to use this flag.
  public init(
    wrappedValue: Value,
    exclusivity: FlagExclusivity = .exclusive,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      initial: wrappedValue,
      exclusivity: exclusivity,
      help: help
    )
  }

  /// Creates a property with no default value that gets its value from the presence of a flag.
  ///
  /// Use this initializer to customize the name and number of states further than using a `Bool`.
  /// To use, define an `EnumerableFlag` enumeration with a case for each state, and use that as the type for your flag.
  /// In this case, the user can specify either `--use-production-server` or `--use-development-server` to set the flag's value.
  ///
  /// ```swift
  /// enum ServerChoice: EnumerableFlag {
  ///   case useProductionServer
  ///   case useDevelopmentServer
  /// }
  ///
  /// @Flag var serverChoice: ServerChoice
  /// ```
  ///
  /// - Parameters:
  ///   - exclusivity: The behavior to use when multiple flags are specified.
  ///   - help: Information about how to use this flag.
  public init(
    exclusivity: FlagExclusivity = .exclusive,
    help: ArgumentHelp? = nil
  ) {
    self.init(
      initial: nil,
      exclusivity: exclusivity,
      help: help
    )
  }
}

extension Flag {
  /// Creates a property that gets its value from the presence of a flag,
  /// where the allowed flags are defined by an `EnumerableFlag` type.
  public init<Element>(
    exclusivity: FlagExclusivity = .exclusive,
    help: ArgumentHelp? = nil
  ) where Value == Element?, Element: EnumerableFlag {
    self.init(_parsedValue: .init { parentKey in
      let caseHelps = Element.allCases.map { Element.help(for: $0) }
      let hasCustomCaseHelp = caseHelps.contains(where: { $0 != nil })

      let args = Element.allCases.enumerated().map { (i, value) -> ArgumentDefinition in
        let caseKey = InputKey(name: String(describing: value), parent: parentKey)
        let name = Element.name(for: value)
        let helpForCase = hasCustomCaseHelp ? (caseHelps[i] ?? help) : help

        let help = ArgumentDefinition.Help(
          allValueStrings: [],
          options: [.isOptional],
          help: helpForCase,
          defaultValue: nil,
          key: parentKey,
          isComposite: !hasCustomCaseHelp)

        return ArgumentDefinition.flag(name: name, key: parentKey, caseKey: caseKey, help: help, parsingStrategy: .default, initialValue: nil as Element?, update: .nullary({ (origin, name, values) in
          try ArgumentSet.updateFlag(key: parentKey, value: value, origin: origin, values: &values, exclusivity: exclusivity)
        }))

      }
      return ArgumentSet(args)
      })
  }

  /// Creates an array property with an optional default value, intended to be called by other constructors to centralize logic.
  ///
  /// This private `init` allows us to expose multiple other similar constructors to allow for standard default property initialization while reducing code duplication.
  private init<Element>(
    initial: [Element]?,
    help: ArgumentHelp? = nil
  ) where Value == Array<Element>, Element: EnumerableFlag {
    self.init(_parsedValue: .init { parentKey in
      let caseHelps = Element.allCases.map { Element.help(for: $0) }
      let hasCustomCaseHelp = caseHelps.contains(where: { $0 != nil })

      let args = Element.allCases.enumerated().map { (i, value) -> ArgumentDefinition in
        let caseKey = InputKey(name: String(describing: value), parent: parentKey)
        let name = Element.name(for: value)
        let helpForCase = hasCustomCaseHelp ? (caseHelps[i] ?? help) : help
        let help = ArgumentDefinition.Help(
          allValueStrings: [],
          options: [.isOptional],
          help: helpForCase,
          defaultValue: nil,
          key: parentKey,
          isComposite: !hasCustomCaseHelp)

        return ArgumentDefinition.flag(name: name, key: parentKey, caseKey: caseKey, help: help, parsingStrategy: .default, initialValue: initial, update: .nullary({ (origin, name, values) in
          values.update(forKey: parentKey, inputOrigin: origin, initial: [Element](), closure: {
            $0.append(value)
          })
        }))
      }
      return ArgumentSet(args)
    })
  }

  /// Creates an array property that gets its values from the presence of
  /// zero or more flags, where the allowed flags are defined by an
  /// `EnumerableFlag` type.
  ///
  /// This property has an empty array as its default value.
  ///
  /// - Parameters:
  ///   - name: A specification for what names are allowed for this flag.
  ///   - help: Information about how to use this flag.
  public init<Element>(
    wrappedValue: [Element],
    help: ArgumentHelp? = nil
  ) where Value == Array<Element>, Element: EnumerableFlag {
    self.init(
      initial: wrappedValue,
      help: help
    )
  }

  /// Creates an array property with no default value that gets its values from the presence of zero or more flags, where the allowed flags are defined by an `EnumerableFlag` type.
  ///
  /// This method is called to initialize an array `Flag` with no default value such as:
  /// ```swift
  /// @Flag
  /// var foo: [CustomFlagType]
  /// ```
  ///
  /// - Parameters:
  ///   - help: Information about how to use this flag.
  public init<Element>(
    help: ArgumentHelp? = nil
  ) where Value == Array<Element>, Element: EnumerableFlag {
    self.init(
      initial: nil,
      help: help
    )
  }
}

extension ArgumentDefinition {
  static func flag<V>(name: NameSpecification, key: InputKey, caseKey: InputKey, help: Help, parsingStrategy: ArgumentDefinition.ParsingStrategy, initialValue: V?, update: Update) -> ArgumentDefinition {
    return ArgumentDefinition(kind: .name(key: caseKey, specification: name), help: help, completion: .default, parsingStrategy: parsingStrategy, update: update, initial: { origin, values in
      if let initial = initialValue {
        values.set(initial, forKey: key, inputOrigin: origin)
      }
    })
  }
}

