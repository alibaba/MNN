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
  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [1, 2, 4]
  ///     let c: BitSet = [0, 1]
  ///     a.isSuperset(of: a) // true
  ///     a.isSuperset(of: b) // true
  ///     a.isSuperset(of: c) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isSuperset(of other: Self) -> Bool {
    other.isSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  /// - Parameter other: A counted bit set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `other`.
  public func isSuperset(of other: BitSet.Counted) -> Bool {
    isSuperset(of: other._bits)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// a given range of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a: BitSet = [0, 1, 2, 3, 4, 10]
  ///     a.isSuperset(of: 0 ..< 4) // true
  ///     a.isSuperset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(`range.count`)
  public func isSuperset(of other: Range<Int>) -> Bool {
    if other.isEmpty { return true }
    guard let r = other._toUInt() else { return false }
    return _read { $0.isSuperset(of: r) }
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the values in a given sequence of integers.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*.
  ///
  ///     let a = [1, 2, 3]
  ///     let b: BitSet = [0, 1, 2, 3, 4]
  ///     let c: BitSet = [0, 1, 2]
  ///     b.isSuperset(of: a) // true
  ///     c.isSuperset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers, some of whose members
  ///    may appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: The same as the complexity of iterating over all elements
  ///    in `other`.
  @inlinable
  public func isSuperset(of other: some Sequence<Int>) -> Bool {
    if let other = _specialize(other, for: BitSet.self) {
      return self.isSuperset(of: other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return self.isSuperset(of: other)
    }
    if let other = _specialize(other, for: Range<Int>.self)  {
      return self.isSuperset(of: other)
    }
    for i in other {
      guard let i = UInt(exactly: i) else { return false }
      if !_contains(i) { return false }
    }
    return true
  }
}
