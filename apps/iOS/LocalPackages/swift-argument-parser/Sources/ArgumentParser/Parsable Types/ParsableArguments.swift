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

/// A type that can be parsed from a program's command-line arguments.
///
/// When you implement a `ParsableArguments` type, all properties must be declared with
/// one of the four property wrappers provided by the `ArgumentParser` library.
public protocol ParsableArguments: Decodable {
  /// Creates an instance of this parsable type using the definitions
  /// given by each property's wrapper.
  init()
  
  /// Validates the properties of the instance after parsing.
  ///
  /// Implement this method to perform validation or other processing after
  /// creating a new instance from command-line arguments.
  mutating func validate() throws
  
  /// The label to use for "Error: ..." messages from this type. (experimental)
  static var _errorLabel: String { get }
}

/// A type that provides the `ParsableCommand` interface to a `ParsableArguments` type.
struct _WrappedParsableCommand<P: ParsableArguments>: ParsableCommand {
  static var _commandName: String {
    let name = String(describing: P.self).convertedToSnakeCase()
    
    // If the type is named something like "TransformOptions", we only want
    // to use "transform" as the command name.
    if let optionsRange = name.range(of: "_options"),
      optionsRange.upperBound == name.endIndex
    {
      return String(name[..<optionsRange.lowerBound])
    } else {
      return name
    }
  }
  
  @OptionGroup var options: P
}

extension ParsableArguments {
  public mutating func validate() throws {}
  
  /// This type as-is if it conforms to `ParsableCommand`, or wrapped in the
  /// `ParsableCommand` wrapper if not.
  internal static var asCommand: ParsableCommand.Type {
    self as? ParsableCommand.Type ?? _WrappedParsableCommand<Self>.self
  }
  
  public static var _errorLabel: String {
    "Error"
  }
}

// MARK: - API

extension ParsableArguments {
  /// Parses a new instance of this type from command-line arguments.
  ///
  /// - Parameter arguments: An array of arguments to use for parsing. If
  ///   `arguments` is `nil`, this uses the program's command-line arguments.
  /// - Returns: A new instance of this type.
  public static func parse(
    _ arguments: [String]? = nil
  ) throws -> Self {
    // Parse the command and unwrap the result if necessary.
    switch try self.asCommand.parseAsRoot(arguments) {
    case let helpCommand as HelpCommand:
      throw ParserError.helpRequested(visibility: helpCommand.visibility)
    case let result as _WrappedParsableCommand<Self>:
      return result.options
    case var result as Self:
      do {
        try result.validate()
      } catch {
        throw ParserError.userValidationError(error)
      }
      return result
    default:
      // TODO: this should be a "wrong command" message
      throw ParserError.invalidState
    }
  }
  
  /// Returns a brief message for the given error.
  ///
  /// - Parameter error: An error to generate a message for.
  /// - Returns: A message that can be displayed to the user.
  public static func message(
    for error: Error
  ) -> String {
    MessageInfo(error: error, type: self).message
  }
  
  @available(*, deprecated, renamed: "fullMessage(for:columns:)")
  @_disfavoredOverload
  public static func fullMessage(
    for _error: Error
  ) -> String {
    MessageInfo(error: _error, type: self).fullText(for: self)
  }

  /// Returns a full message for the given error, including usage information,
  /// if appropriate.
  ///
  /// - Parameters:
  ///   - error: An error to generate a message for.
  ///   - columns: The column width to use when wrapping long line in the
  ///     help screen. If `columns` is `nil`, uses the current terminal
  ///     width, or a default value of `80` if the terminal width is not
  ///     available.
  /// - Returns: A message that can be displayed to the user.
  public static func fullMessage(
    for error: Error,
    columns: Int? = nil
  ) -> String {
    MessageInfo(error: error, type: self, columns: columns).fullText(for: self)
  }

  /// Returns the text of the help screen for this type.
  ///
  /// - Parameters:
  ///   - columns: The column width to use when wrapping long line in the
  ///     help screen. If `columns` is `nil`, uses the current terminal
  ///     width, or a default value of `80` if the terminal width is not
  ///     available.
  /// - Returns: The full help screen for this type.
  @_disfavoredOverload
  @available(*, deprecated, message: "Use helpMessage(includeHidden:columns:) instead.")
  public static func helpMessage(
    columns _columns: Int?
  ) -> String {
    helpMessage(includeHidden: false, columns: _columns)
  }

  /// Returns the text of the help screen for this type.
  ///
  /// - Parameters:
  ///   - includeHidden: Include hidden help information in the generated
  ///     message.
  ///   - columns: The column width to use when wrapping long line in the
  ///     help screen. If `columns` is `nil`, uses the current terminal
  ///     width, or a default value of `80` if the terminal width is not
  ///     available.
  /// - Returns: The full help screen for this type.
  public static func helpMessage(
    includeHidden: Bool = false,
    columns: Int? = nil
  ) -> String {
    HelpGenerator(self, visibility: includeHidden ? .hidden : .default)
      .rendered(screenWidth: columns)
  }

  /// Returns the JSON representation of this type.
  public static func _dumpHelp() -> String {
    DumpHelpGenerator(self).rendered()
  }

  /// Returns the exit code for the given error.
  ///
  /// The returned code is the same exit code that is used if `error` is passed
  /// to `exit(withError:)`.
  ///
  /// - Parameter error: An error to generate an exit code for.
  /// - Returns: The exit code for `error`.
  public static func exitCode(
    for error: Error
  ) -> ExitCode {
    MessageInfo(error: error, type: self).exitCode
  }
    
  /// Returns a shell completion script for the specified shell.
  ///
  /// - Parameter shell: The shell to generate a completion script for.
  /// - Returns: The completion script for `shell`.
  public static func completionScript(for shell: CompletionShell) -> String {
    let completionsGenerator = try! CompletionsGenerator(command: self.asCommand, shell: shell)
    return completionsGenerator.generateCompletionScript()
  }

  /// Terminates execution with a message and exit code that is appropriate
  /// for the given error.
  ///
  /// If the `error` parameter is `nil`, this method prints nothing and exits
  /// with code `EXIT_SUCCESS`. If `error` represents a help request or
  /// another `CleanExit` error, this method prints help information and
  /// exits with code `EXIT_SUCCESS`. Otherwise, this method prints a relevant
  /// error message and exits with code `EX_USAGE` or `EXIT_FAILURE`.
  ///
  /// - Parameter error: The error to use when exiting, if any.
  public static func exit(
    withError error: Error? = nil
  ) -> Never {
    guard let error = error else {
      Platform.exit(ExitCode.success.rawValue)
    }
    
    let messageInfo = MessageInfo(error: error, type: self)
    let fullText = messageInfo.fullText(for: self)
    if !fullText.isEmpty {
      if messageInfo.shouldExitCleanly {
        print(fullText)
      } else {
        var errorOut = Platform.standardError
        print(fullText, to: &errorOut)
      }
    }
    Platform.exit(messageInfo.exitCode.rawValue)
  }
  
  /// Parses a new instance of this type from command-line arguments or exits
  /// with a relevant message.
  ///
  /// - Parameter arguments: An array of arguments to use for parsing. If
  ///   `arguments` is `nil`, this uses the program's command-line arguments.
  public static func parseOrExit(
    _ arguments: [String]? = nil
  ) -> Self {
    do {
      return try parse(arguments)
    } catch {
      exit(withError: error)
    }
  }

  /// Returns the usage text for this type.
  ///
  /// - Parameters:
  ///   - includeHidden: Include hidden help information in the generated
  ///     message.
  /// - Returns: The usage text for this type.
  public static func usageString(
    includeHidden: Bool = false
  ) -> String {
    HelpGenerator(self, visibility: includeHidden ? .hidden : .default).usage
  }
}

/// Unboxes the given value if it is a `nil` value.
///
/// If the value passed is the `.none` case of any optional type, this function
/// returns `nil`.
///
///     let intAsAny = (1 as Int?) as Any
///     let nilAsAny = (nil as Int?) as Any
///     nilOrValue(intAsAny)      // Optional(1) as Any?
///     nilOrValue(nilAsAny)      // nil as Any?
func nilOrValue(_ value: Any) -> Any? {
  if case Optional<Any>.none = value {
    return nil
  } else {
    return value
  }
}

/// Existential protocol for property wrappers, so that they can provide
/// the argument set that they define.
protocol ArgumentSetProvider {
  func argumentSet(for key: InputKey) -> ArgumentSet
    
  var _visibility: ArgumentVisibility { get }
}

extension ArgumentSetProvider {
  var _visibility: ArgumentVisibility { .default }
}

extension ArgumentSet {
  init(_ type: ParsableArguments.Type, visibility: ArgumentVisibility, parent: InputKey?) {
    #if DEBUG
    do {
      try type._validate(parent: parent)
    } catch {
      assertionFailure("\(error)")
    }
    #endif
    
    let a: [ArgumentSet] = Mirror(reflecting: type.init())
      .children
      .compactMap { child -> ArgumentSet? in
        guard let codingKey = child.label else { return nil }
        
        if let parsed = child.value as? ArgumentSetProvider {
          guard parsed._visibility.isAtLeastAsVisible(as: visibility)
            else { return nil }

          let key = InputKey(name: codingKey, parent: parent)
          return parsed.argumentSet(for: key)
        } else {
          let arg = ArgumentDefinition(
            unparsedKey: codingKey,
            default: nilOrValue(child.value), parent: parent)

          // Save a non-wrapped property as is
          return ArgumentSet(arg)
        }
      }
    self.init(
      a.joined().filter { $0.help.visibility.isAtLeastAsVisible(as: visibility) })
  }
}

/// The fatal error message to display when someone accesses a
/// `ParsableArguments` type after initializing it directly.
internal let directlyInitializedError = """

  --------------------------------------------------------------------
  Can't read a value from a parsable argument definition.

  This error indicates that a property declared with an `@Argument`,
  `@Option`, `@Flag`, or `@OptionGroup` property wrapper was neither
  initialized to a value nor decoded from command-line arguments.

  To get a valid value, either call one of the static parsing methods
  (`parse`, `parseAsRoot`, or `main`) or define an initializer that
  initializes _every_ property of your parsable type.
  --------------------------------------------------------------------

  """
