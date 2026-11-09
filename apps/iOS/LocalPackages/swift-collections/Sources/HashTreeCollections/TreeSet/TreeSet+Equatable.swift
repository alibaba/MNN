//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension TreeSet: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// Two persistent sets are considered equal if they contain the same
  /// elements, but not necessarily in the same order.
  ///
  /// - Note: This simply forwards to the ``isEqualSet(to:)-4bc1i`` method.
  /// That method has additional overloads that can be used to compare
  /// persistent sets with additional types.
  ///
  /// - Complexity: O(`min(left.count, right.count)`)
  @inlinable @inline(__always)
  public static func == (left: Self, right: Self) -> Bool {
    left.isEqualSet(to: right)
  }
}
