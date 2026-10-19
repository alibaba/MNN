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

/// A type that can be executed asynchronously, as part of a nested tree of
/// commands.
@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
public protocol AsyncParsableCommand: ParsableCommand {
  /// The behavior or functionality of this command.
  ///
  /// Implement this method in your `ParsableCommand`-conforming type with the
  /// functionality that this command represents.
  ///
  /// This method has a default implementation that prints the help screen for
  /// this command.
  mutating func run() async throws
}

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
extension AsyncParsableCommand {
  /// Executes this command, or one of its subcommands, with the given arguments.
  ///
  /// This method parses an instance of this type, one of its subcommands, or
  /// another built-in `AsyncParsableCommand` type, from command-line
  /// (or provided) arguments, and then calls its `run()` method, exiting
  /// with a relevant error message if necessary.
  ///
  /// - Parameter arguments: An array of arguments to use for parsing. If
  ///   `arguments` is `nil`, this uses the program's command-line arguments.
  public static func main(_ arguments: [String]?) async {
    do {
      var command = try parseAsRoot(arguments)
      if var asyncCommand = command as? AsyncParsableCommand {
        try await asyncCommand.run()
      } else {
        try command.run()
      }
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
  /// another built-in `AsyncParsableCommand` type, from command-line arguments,
  /// and then calls its `run()` method, exiting with a relevant error message
  /// if necessary.
  public static func main() async {
    await self.main(nil)
  }
}

/// A type that can designate an `AsyncParsableCommand` as the program's
/// entry point.
///
/// See the ``AsyncParsableCommand`` documentation for usage information.
@available(
  swift, deprecated: 5.6,
  message: "Use @main directly on your root `AsyncParsableCommand` type.")
public protocol AsyncMainProtocol {
  associatedtype Command: ParsableCommand
}

@available(swift, deprecated: 5.6)
@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
extension AsyncMainProtocol {
  /// Executes the designated command type, or one of its subcommands, with
  /// the program's command-line arguments.
  public static func main() async {
    do {
      var command = try Command.parseAsRoot()
      if var asyncCommand = command as? AsyncParsableCommand {
        try await asyncCommand.run()
      } else {
        try command.run()
      }
    } catch {
      Command.exit(withError: error)
    }
  }
}
