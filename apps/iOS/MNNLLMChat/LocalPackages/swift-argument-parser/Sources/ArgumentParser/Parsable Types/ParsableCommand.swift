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

/// A type that can be executed as part of a nested tree of commands.
public protocol ParsableCommand: ParsableArguments {
  /// Configuration for this command, including subcommands and custom help
  /// text.
  static var configuration: CommandConfiguration { get }
  
  /// *For internal use only:* The name for the command on the command line.
  ///
  /// This is generated from the configuration, if given, or from the type
  /// name if not. This is a customization point so that a WrappedParsable
  /// can pass through the wrapped type's name.
  static var _commandName: String { get }
  
  /// The behavior or functionality of this command.
  ///
  /// Implement this method in your `ParsableCommand`-conforming type with the
  /// functionality that this command represents.
  ///
  /// This method has a default implementation that prints the help screen for
  /// this command.
  mutating func run() throws
}

// MARK: - Default implementations

extension ParsableCommand {
  public static var _commandName: String {
    configuration.commandName ??
      String(describing: Self.self).convertedToSnakeCase(separator: "-")
  }
  
  public static var configuration: CommandConfiguration {
    CommandConfiguration()
  }

  public mutating func run() throws {
    throw CleanExit.helpRequest(self)
  }
}

// MARK: - API

extension ParsableCommand {
  /// Parses an instance of this type, or one of its subcommands, from
  /// command-line arguments.
  ///
  /// - Parameter arguments: An array of arguments to use for parsing. If
  ///   `arguments` is `nil`, this uses the program's command-line arguments.
  /// - Returns: A new instance of this type, one of its subcommands, or a
  ///   command type internal to the `ArgumentParser` library.
  public static func parseAsRoot(
    _ arguments: [String]? = nil
  ) throws -> ParsableCommand {
    var parser = CommandParser(self)
    let arguments = arguments ?? Array(CommandLine._staticArguments.dropFirst())
    return try parser.parse(arguments: arguments).get()
  }
  
  /// Returns the text of the help screen for the given subcommand of this
  /// command.
  ///
  /// - Parameters:
  ///   - subcommand: The subcommand to generate the help screen for.
  ///     `subcommand` must be declared in the subcommand tree of this
  ///     command.
  ///   - columns: The column width to use when wrapping long line in the
  ///     help screen. If `columns` is `nil`, uses the current terminal
  ///     width, or a default value of `80` if the terminal width is not
  ///     available.
  /// - Returns: The full help screen for this type.
  @_disfavoredOverload
  @available(*, deprecated, renamed: "helpMessage(for:includeHidden:columns:)")
  public static func helpMessage(
    for _subcommand: ParsableCommand.Type,
    columns: Int? = nil
  ) -> String {
    helpMessage(for: _subcommand, includeHidden: false, columns: columns)
  }

  /// Returns the text of the help screen for the given subcommand of this
  /// command.
  ///
  /// - Parameters:
  ///   - subcommand: The subcommand to generate the help screen for.
  ///     `subcommand` must be declared in the subcommand tree of this
  ///     command.
  ///   - includeHidden: Include hidden help information in the generated
  ///     message.
  ///   - columns: The column width to use when wrapping long line in the
  ///     help screen. If `columns` is `nil`, uses the current terminal
  ///     width, or a default value of `80` if the terminal width is not
  ///     available.
  /// - Returns: The full help screen for this type.
  public static func helpMessage(
    for subcommand: ParsableCommand.Type,
    includeHidden: Bool = false,
    columns: Int? = nil
  ) -> String {
    HelpGenerator(
      commandStack: CommandParser(self).commandStack(for: subcommand),
      visibility: includeHidden ? .hidden : .default)
        .rendered(screenWidth: columns)
  }

  /// Returns the usage text for the given subcommand of this command.
  ///
  /// - Parameters:
  ///   - includeHidden: Include hidden help information in the generated
  ///     message.
  /// - Returns: The usage text for this type.
  public static func usageString(
    for subcommand: ParsableCommand.Type,
    includeHidden: Bool = false
  ) -> String {
    HelpGenerator(
      commandStack: CommandParser(self).commandStack(for: subcommand),
      visibility: includeHidden ? .hidden : .default)
        .usage
  }

  /// Executes this command, or one of its subcommands, with the given
  /// arguments.
  ///
  /// This method parses an instance of this type, one of its subcommands, or
  /// another built-in `ParsableCommand` type, from command-line arguments,
  /// and then calls its `run()` method, exiting with a relevant error message
  /// if necessary.
  ///
  /// - Parameter arguments: An array of arguments to use for parsing. If
  ///   `arguments` is `nil`, this uses the program's command-line arguments.
  public static func main(_ arguments: [String]?) {
    
#if DEBUG
    if #available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *) {
      if let asyncCommand = firstAsyncSubcommand(self) {
        if Self() is AsyncParsableCommand {
          failAsyncPlatform(rootCommand: self)
        } else {
          failAsyncHierarchy(rootCommand: self, subCommand: asyncCommand)
        }
      }
    }
#endif
    
    do {
      var command = try parseAsRoot(arguments)
      try command.run()
    } catch {
      exit(withError: error)
    }
  }

  /// Executes this command, or one of its subcommands, with the program's
  /// command-line arguments.
  ///
  /// Instead of calling this method directly, you can add `@main` to the root
  /// command for your command-line tool.
  ///
  /// This method parses an instance of this type, one of its subcommands, or
  /// another built-in `ParsableCommand` type, from command-line arguments,
  /// and then calls its `run()` method, exiting with a relevant error message
  /// if necessary.
  public static func main() {
    self.main(nil)
  }
}

// MARK: - Internal API

extension ParsableCommand {
  /// `true` if this command contains any array arguments that are declared
  /// with `.unconditionalRemaining`.
  internal static var includesPassthroughArguments: Bool {
    ArgumentSet(self, visibility: .private, parent: nil).contains(where: {
      $0.isRepeatingPositional && $0.parsingStrategy == .allRemainingInput
    })
  }
  
  internal static var includesAllUnrecognizedArgument: Bool {
    ArgumentSet(self, visibility: .private, parent: nil).contains(where: {
      $0.isRepeatingPositional && $0.parsingStrategy == .allUnrecognized
    })
  }
  
  /// `true` if this command's default subcommand contains any array arguments
  /// that are declared with `.unconditionalRemaining`. This is `false` if
  /// there's no default subcommand.
  internal static var defaultIncludesPassthroughArguments: Bool {
    configuration.defaultSubcommand?.includesPassthroughArguments == true
  }
  
#if DEBUG
  @available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
  internal static func checkAsyncHierarchy(_ command: ParsableCommand.Type, root: String) {
    for sub in command.configuration.subcommands {
      checkAsyncHierarchy(sub, root: root)

      guard sub.configuration.subcommands.isEmpty else { continue }
      guard sub is AsyncParsableCommand.Type else { continue }

      fatalError("""

      --------------------------------------------------------------------
      Asynchronous subcommand of a synchronous root.

      The asynchronous command `\(sub)` is declared as a subcommand of the synchronous root command `\(root)`.

      With this configuration, your asynchronous `run()` method will not be called. To fix this issue, change `\(root)`'s `ParsableCommand` conformance to `AsyncParsableCommand`.
      --------------------------------------------------------------------

      """.wrapped(to: 70))
    }
  }

  @available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
  internal static func firstAsyncSubcommand(_ command: ParsableCommand.Type) -> AsyncParsableCommand.Type? {
    for sub in command.configuration.subcommands {
      if let asyncCommand = sub as? AsyncParsableCommand.Type,
         sub.configuration.subcommands.isEmpty
      {
        return asyncCommand
      }
      
      if let asyncCommand = firstAsyncSubcommand(sub) {
        return asyncCommand
      }
    }
    
    return nil
  }
#endif
}

// MARK: Async Configuration Errors

func failAsyncHierarchy(
  rootCommand: ParsableCommand.Type, subCommand: ParsableCommand.Type
) -> Never {
  fatalError("""

  --------------------------------------------------------------------
  Asynchronous subcommand of a synchronous root.

  The asynchronous command `\(subCommand)` is declared as a subcommand of the synchronous root command `\(rootCommand)`.

  With this configuration, your asynchronous `run()` method will not be called. To fix this issue, change `\(rootCommand)`'s `ParsableCommand` conformance to `AsyncParsableCommand`.
  --------------------------------------------------------------------

  """.wrapped(to: 70))
}

func failAsyncPlatform(rootCommand: ParsableCommand.Type) -> Never {
  fatalError("""

  --------------------------------------------------------------------
  Asynchronous root command needs availability annotation.

  The asynchronous root command `\(rootCommand)` needs an availability annotation in order to be executed asynchronously. To fix this issue, add the following availability attribute to your `\(rootCommand)` declaration or set the minimum platform in your "Package.swift" file.
  
  """.wrapped(to: 70)
  + """
  
  @available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
  --------------------------------------------------------------------
  
  """)
}
