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

extension Deque: Equatable where Element: Equatable {
  /// Returns a Boolean value indicating whether two values are equal. Two
  /// deques are considered equal if they contain the same elements in the same
  /// order.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    let lhsCount = left.count
    if lhsCount != right.count {
      return false
    }

    // Test referential equality.
    if lhsCount == 0 || left._storage.isIdentical(to: right._storage) {
      return true
    }

    return left.elementsEqual(right)
  }
}
