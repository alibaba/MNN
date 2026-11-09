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

extension OrderedSet {
  /// Returns a Boolean value indicating whether two set values contain the
  /// same elements, but not necessarily in the same order.
  ///
  /// - Note: This member implements different behavior than the `==(_:_:)`
  ///    operator -- the latter implements an ordered comparison, matching
  ///    the stricter concept of equality expected of an ordered collection
  ///    type.
  ///
  /// - Complexity: O(`min(left.count, right.count)`), as long as`Element`
  ///    properly implements hashing.
  public func isEqualSet(to other: Self) -> Bool {
    self.unordered == other.unordered
  }

  /// Returns a Boolean value indicating whether two set values contain the
  /// same elements, but not necessarily in the same order.
  ///
  /// - Complexity: O(`min(left.count, right.count)`), as long as`Element`
  ///    properly implements hashing.
  public func isEqualSet(to other: UnorderedView) -> Bool {
    self.unordered == other
  }

  /// Returns a Boolean value indicating whether an ordered set contains the
  /// same values as a given sequence, but not necessarily in the same
  /// order.
  ///
  /// Duplicate items in `other` do not prevent it from comparing equal to
  /// `self`.
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in
  ///    `other`, as long as`Element` properly implements hashing.
  public func isEqualSet(to other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return isEqualSet(to: other)
    }

    if self.isEmpty {
      return other.allSatisfy { _ in false }
    }

    if other is _UniqueCollection {
      // We don't need to create a temporary set.
      guard other.underestimatedCount <= self.count else { return false }
      var seen = 0
      for item in other {
        guard self.contains(item) else { return false }
        seen &+= 1
      }
      precondition(
        seen <= self.count,
        // Otherwise other.underestimatedCount != other.count
        "Invalid Collection '\(type(of: other))' (bad underestimatedCount)")
      return seen == self.count
    }

    return _UnsafeBitSet.withTemporaryBitSet(capacity: count) { seen in
      var c = 0
      for item in other {
        guard let index = _find(item).index else { return false }
        if seen.insert(index) {
          c &+= 1
        }
      }
      return c == count
    }
  }
}
