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

/// A type that represents the different possible flags to be used by a
/// `@Flag` property.
///
/// For example, the `Size` enumeration declared here can be used as the type of
/// a `@Flag` property:
///
/// ```swift
/// enum Size: String, EnumerableFlag {
///     case small, medium, large, extraLarge
/// }
///
/// struct Example: ParsableCommand {
///     @Flag var sizes: [Size]
///
///     mutating func run() {
///         print(sizes)
///     }
/// }
/// ```
///
/// By default, each case name is converted to a flag by using the `.long` name
/// specification, so a user can call `example` like this:
///
///     $ example --small --large
///     [.small, .large]
///
/// Provide alternative or additional name specifications for each case by
/// implementing the `name(for:)` static method on your `EnumerableFlag` type.
///
/// ```swift
/// extension Size {
///     static func name(for value: Self) -> NameSpecification {
///         switch value {
///         case .extraLarge:
///             return [.customShort("x"), .long]
///         default:
///             return .shortAndLong
///         }
///     }
/// }
/// ```
/// 
/// With this extension, a user can use short or long versions of the flags:
///
///     $ example -s -l -x --medium
///     [.small, .large, .extraLarge, .medium]
public protocol EnumerableFlag: CaseIterable, Equatable {
  /// Returns the name specification to use for the given flag.
  ///
  /// The default implementation for this method always returns `.long`.
  /// Implement this method for your custom `EnumerableFlag` type to provide
  /// different name specifications for different cases.
  static func name(for value: Self) -> NameSpecification
  
  /// Returns the help information to show for the given flag.
  ///
  /// The default implementation for this method always returns `nil`, which
  /// groups the flags together with the help provided in the `@Flag`
  /// declaration. Implement this method for your custom type to provide
  /// different help information for each flag.
  static func help(for value: Self) -> ArgumentHelp?
}

extension EnumerableFlag {
  public static func name(for value: Self) -> NameSpecification {
    .long
  }
  
  public static func help(for value: Self) -> ArgumentHelp? {
    nil
  }
}
