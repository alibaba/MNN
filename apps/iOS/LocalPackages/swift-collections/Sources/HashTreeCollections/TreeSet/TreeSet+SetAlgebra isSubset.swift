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

extension TreeSet {
  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isSubset(of other: Self) -> Bool {
    self._root.isSubset(.top, of: other._root)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isSubset<Value>(
    of other: TreeDictionary<Element, Value>.Keys
  ) -> Bool {
    self._root.isSubset(.top, of: other._base._root)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given sequence.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: A sequence of elements, some of whose elements may
  ///    appear more than once.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: In the worst case, this makes O(*n*) calls to
  ///    `self.contains` (where *n* is the number of elements in `other`),
  ///    and it constructs a temporary persistent set containing every
  ///    element of the sequence.
  @inlinable
  public func isSubset(of other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return isSubset(of: other)
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

    // FIXME: Would making this a BitSet of seen positions be better?
    var seen: _Node = ._emptyNode()
    for item in other {
      let hash = _Hash(item)
      guard _root.containsKey(.top, item, hash) else { continue }
      guard seen.insert(.top, (item, ()), hash).inserted else { continue }
      if seen.count == self.count {
        return true
      }
    }
    return false
  }
}
