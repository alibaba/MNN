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

extension OrderedSet: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// Two ordered sets are considered equal if they contain the same
  /// elements in the same order.
  ///
  /// - Note: This operator implements different behavior than the
  ///    `isEqualSet(to:)` method -- the latter implements an unordered
  ///    comparison, to match the behavior of members like `isSubset(of:)`,
  ///    `isStrictSuperset(of:)` etc.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._elements == right._elements
  }
}
