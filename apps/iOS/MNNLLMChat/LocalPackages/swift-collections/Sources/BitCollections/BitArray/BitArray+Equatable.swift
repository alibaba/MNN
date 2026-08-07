//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension BitArray: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// Equality is the inverse of inequality. For any values `a` and `b`,
  /// `a == b` implies that `a != b` is `false`.
  ///
  /// - Parameter lhs: A value to compare.
  /// - Parameter rhs: Another value to compare.
  /// - Complexity: O(left.count)
  public static func ==(left: Self, right: Self) -> Bool {
    guard left._count == right._count else { return false }
    return left._storage == right._storage
  }
}
