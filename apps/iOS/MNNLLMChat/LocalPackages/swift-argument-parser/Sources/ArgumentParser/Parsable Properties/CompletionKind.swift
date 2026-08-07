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

/// The type of completion to use for an argument or option.
public struct CompletionKind {
  internal enum Kind {
    /// Use the default completion kind for the value's type.
    case `default`

    /// Use the specified list of completion strings.
    case list([String])

    /// Complete file names with the specified extensions.
    case file(extensions: [String])

    /// Complete directory names that match the specified pattern.
    case directory

    /// Call the given shell command to generate completions.
    case shellCommand(String)

    /// Generate completions using the given closure.
    case custom(@Sendable ([String]) -> [String])
  }
  
  internal var kind: Kind
  
  /// Use the default completion kind for the value's type.
  public static var `default`: CompletionKind {
    CompletionKind(kind: .default)
  }
  
  /// Use the specified list of completion strings.
  public static func list(_ words: [String]) -> CompletionKind {
    CompletionKind(kind: .list(words))
  }
  
  /// Complete file names.
  public static func file(extensions: [String] = []) -> CompletionKind {
    CompletionKind(kind: .file(extensions: extensions))
  }

  /// Complete directory names.
  public static var directory: CompletionKind {
    CompletionKind(kind: .directory)
  }
  
  /// Call the given shell command to generate completions.
  public static func shellCommand(_ command: String) -> CompletionKind {
    CompletionKind(kind: .shellCommand(command))
  }
  
  /// Generate completions using the given closure.
  @preconcurrency
  public static func custom(_ completion: @Sendable @escaping ([String]) -> [String]) -> CompletionKind {
    CompletionKind(kind: .custom(completion))
  }
}

extension CompletionKind: Sendable { }
extension CompletionKind.Kind: Sendable { }
