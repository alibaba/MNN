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

// FIXME: These are non-standard extensions generalizing ==.
extension BitSet {
  /// Returns a Boolean value indicating whether two bit sets are equal. Two
  /// bit sets are considered equal if they contain the same elements.
  ///
  /// - Complexity: O(*max*), where *max* is value of the largest member of
  ///     either set.
  public func isEqualSet(to other: Self) -> Bool {
    self._storage == other._storage
  }

  /// Returns a Boolean value indicating whether a bit set is equal to a counted
  /// bit set, i.e., whether they contain the same values.
  ///
  /// - Complexity: O(*max*), where *max* is value of the largest member of
  ///     either set.
  public func isEqualSet(to other: BitSet.Counted) -> Bool {
    self.isEqualSet(to: other._bits)
  }

  /// Returns a Boolean value indicating whether a bit set is equal to a range
  /// of integers, i.e., whether they contain the same values.
  ///
  /// - Complexity: O(min(*max*, `other.upperBound`), where *max* is the largest
  ///    member of `self`.
  public func isEqualSet(to other: Range<Int>) -> Bool {
    guard let other = other._toUInt() else { return false }
    return _read { $0.isEqualSet(to: other) }
  }

  /// Returns a Boolean value indicating whether this bit set contains the same
  /// elements as the given `other` sequence.
  ///
  /// Duplicate items in `other` do not prevent it from comparing equal to
  /// `self`.
  ///
  ///     let bits: BitSet = [0, 1, 5, 6]
  ///     let other = [5, 5, 0, 1, 1, 6, 5, 0, 1, 6, 6, 5]
  ///
  ///     bits.isEqualSet(to: other) // true
  ///
  /// - Complexity: O(*n*), where *n* is the number of items in `other`.
  public func isEqualSet(to other: some Sequence<Int>) -> Bool {
    if let other = _specialize(other, for: BitSet.self) {
      return isEqualSet(to: other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return isEqualSet(to: other)
    }
    if let other = _specialize(other, for: Range<Int>.self) {
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
        guard let item = UInt(exactly: item) else { return false }
        guard self._contains(item) else { return false}
        seen &+= 1
      }
      precondition(
        seen <= self.count,
        // Otherwise other.underestimatedCount != other.count
        "Invalid Collection '\(type(of: other))' (bad underestimatedCount)")
      return seen == self.count
    }

    var seen: BitSet? = BitSet(reservingCapacity: self.max()!)
    var it = other.makeIterator()
    while let item = it.next() {
      guard let item = UInt(exactly: item) else { return false }
      guard self._contains(item) else { return false}
      seen!._insert(item) // Ignore dupes
      if seen!.count == self.count {
        // We've seen them all. Stop further accounting.
        seen = nil
        break
      }
    }
    guard seen == nil else { return false }
    while let item = it.next() {
      guard let item = UInt(exactly: item) else { return false }
      guard self._contains(item) else { return false}
    }
    return true
  }
}
