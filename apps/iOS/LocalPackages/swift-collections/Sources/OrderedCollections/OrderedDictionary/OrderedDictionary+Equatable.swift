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

extension OrderedDictionary: Equatable where Value: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// Two ordered dictionaries are considered equal if they contain the same
  /// key-value pairs, in the same order.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._keys == right._keys && left._values == right._values
  }
}
