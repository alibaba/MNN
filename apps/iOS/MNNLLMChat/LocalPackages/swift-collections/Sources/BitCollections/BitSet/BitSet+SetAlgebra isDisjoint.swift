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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension BitSet {
  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func isDisjoint(with other: Self) -> Bool {
    self._read { first in
      other._read { second in
        let w1 = first._words
        let w2 = second._words
        for i in 0 ..< Swift.min(w1.count, w2.count) {
          if !w1[i].intersection(w2[i]).isEmpty { return false }
        }
        return true
      }
    }
  }

  /// Returns a Boolean value that indicates whether a bit set has no members
  /// in common with the given counted bit set.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet.Counted = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: A counted bit set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func isDisjoint(with other: BitSet.Counted) -> Bool {
    self.isDisjoint(with: other._bits)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given range of integers.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     a.isDisjoint(with: -10 ..< 0) // true
  ///
  /// - Parameter other: A range of arbitrary integers.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isDisjoint(with other: Range<Int>) -> Bool {
    _read { $0.isDisjoint(with: other._clampedToUInt()) }
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given sequence of integers.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [5, 6, -10, 42]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isDisjoint(with other: some Sequence<Int>) -> Bool {
    if let other = _specialize(other, for: BitSet.self) {
      return self.isDisjoint(with: other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return self.isDisjoint(with: other)
    }
    if let other = _specialize(other, for: Range<Int>.self) {
      return self.isDisjoint(with: other)
    }
    for value in other {
      guard !contains(value) else { return false }
    }
    return true
  }
}
