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
  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [1, 2, 4]
  ///     let c: BitSet = [0, 1]
  ///     a.isSubset(of: a) // true
  ///     b.isSubset(of: a) // true
  ///     c.isSubset(of: a) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: Self) -> Bool {
    self._read { first in
      other._read { second in
        let w1 = first._words
        let w2 = second._words
        if w1.count > w2.count {
          return false
        }
        for i in 0 ..< w1.count {
          if !w1[i].subtracting(w2[i]).isEmpty {
            return false
          }
        }
        return true
      }
    }
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  /// - Parameter other: A counted bit set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: BitSet.Counted) -> Bool {
    self.isSubset(of: other._bits)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given range of integers.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let b: BitSet = [0, 1, 2]
  ///     let c: BitSet = [2, 3, 4]
  ///     b.isSubset(of: -10 ..< 4) // true
  ///     c.isSubset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isSubset(of other: Range<Int>) -> Bool {
    _read { $0.isSubset(of: other._clampedToUInt()) }
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the values in a given sequence of integers.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*.
  ///
  ///     let a = [1, 2, 3, 4, -10]
  ///     let b: BitSet = [1, 2, 4]
  ///     let c: BitSet = [0, 1]
  ///     b.isSubset(of: a) // true
  ///     c.isSubset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isSubset(of other: some Sequence<Int>) -> Bool {
    if let other = _specialize(other, for: BitSet.self) {
      return self.isSubset(of: other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return self.isSubset(of: other)
    }
    if let other = _specialize(other, for: Range<Int>.self)  {
      return self.isSubset(of: other)
    }

    var it = self.makeIterator()
    guard let first = it.next() else { return true }
    if let match = other._customContainsEquatableElement(first) {
      // Fast path: the sequence has fast containment checks.
      guard match else { return false }
      while let item = it.next() {
        guard other.contains(item) else { return false }
      }
      return true
    }

    var t = self
    for i in other {
      guard let i = UInt(exactly: i) else { continue }
      if t._remove(i), t.isEmpty { return true }
    }
    assert(!t.isEmpty)
    return false
  }
}
