//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2019 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension TreeDictionary: Equatable where Value: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// Two persistent dictionaries are considered equal if they contain the same
  /// key-value pairs, but not necessarily in the same order.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable
  public static func == (left: Self, right: Self) -> Bool {
    left._root.isEqualSet(to: right._root, by: { $0 == $1 })
  }
}
