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
  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let a: BitSet = [1, 2, 3, 4]
  ///     let b: BitSet = [1, 2, 4]
  ///     let c: BitSet = [0, 1]
  ///     a.isStrictSubset(of: a) // false
  ///     b.isStrictSubset(of: a) // true
  ///     c.isStrictSubset(of: a) // false
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: Self) -> Bool {
    self._read { first in
      other._read { second in
        let w1 = first._words
        let w2 = second._words
        if w1.count > w2.count {
          return false
        }
        var strict = w1.count < w2.count
        for i in 0 ..< w1.count {
          if !w1[i].subtracting(w2[i]).isEmpty {
            return false
          }
          strict = strict || w1[i] != w2[i]
        }
        return strict
      }
    }
  }

  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  /// - Parameter other: A counted bit set.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: BitSet.Counted) -> Bool {
    isStrictSubset(of: other._bits)
  }

  /// Returns a Boolean value that indicates whether this set is a strict
  /// subset of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let b: BitSet = [0, 1, 2]
  ///     let c: BitSet = [2, 3, 4]
  ///     b.isStrictSubset(of: -10 ..< 4) // true
  ///     c.isStrictSubset(of: -10 ..< 4) // false
  ///
  /// - Parameter other: An arbitrary range of integers.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public func isStrictSubset(of other: Range<Int>) -> Bool {
    isSubset(of: other) && !isSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether this bit set is a strict
  /// subset of the values in a given sequence of integers.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*.
  ///
  ///     let a = [1, 2, 3, 4, -10]
  ///     let b: BitSet = [1, 2, 4]
  ///     let c: BitSet = [0, 1]
  ///     b.isStrictSubset(of: a) // true
  ///     c.isStrictSubset(of: a) // false
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: `true` if the set is a strict subset of `other`;
  ///     otherwise, `false`.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func isStrictSubset(of other: some Sequence<Int>) -> Bool {
    if let other = _specialize(other, for: BitSet.self) {
      return isStrictSubset(of: other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return isStrictSubset(of: other)
    }
    if let other = _specialize(other, for: Range<Int>.self) {
      return isStrictSubset(of: other)
    }

    if isEmpty {
      var it = other.makeIterator()
      return it.next() != nil
    }

    let selfCount = self.count
    return _UnsafeHandle.withTemporaryBitSet(
      wordCount: _storage.count
    ) { seen in
      var strict = false
      var it = other.makeIterator()
      var c = 0
      while let i = it.next() {
        guard self.contains(i) else {
          strict = true
          continue
        }
        if seen.insert(UInt(i)) {
          c &+= 1
          if c == selfCount {
            while !strict, let i = it.next() {
              strict = !self.contains(i)
            }
            return strict
          }
        }
      }
      assert(c < selfCount)
      return false
    }
  }
}
