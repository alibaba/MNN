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

/// Visibility level of an argument's help.
public struct ArgumentVisibility: Hashable {
  /// Internal implementation of `ArgumentVisibility` to allow for easier API
  /// evolution.
  internal enum Representation {
    case `default`
    case hidden
    case `private`
  }

  internal var base: Representation

  /// Show help for this argument whenever appropriate.
  public static let `default` = Self(base: .default)

  /// Only show help for this argument in the extended help screen.
  public static let hidden = Self(base: .hidden)

  /// Never show help for this argument.
  public static let `private` = Self(base: .private)
}

extension ArgumentVisibility: Sendable { }

extension ArgumentVisibility.Representation {
  /// A raw Integer value that represents each visibility level.
  ///
  /// `_comparableLevel` can be used to test if a Visibility case is more or
  /// less visible than another, without committing this behavior to API.
  /// A lower `_comparableLevel` indicates that the case is less visible (more
  /// secret).
  internal var _comparableLevel: Int {
    switch self {
    case .default:
      return 2
    case .hidden:
      return 1
    case .private:
      return 0
    }
  }
}

extension ArgumentVisibility {
  /// - Returns: true if `self` is at least as visible as the supplied argument.
  internal func isAtLeastAsVisible(as other: Self) -> Bool {
    self.base._comparableLevel >= other.base._comparableLevel
  }
}
