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
  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isStrictSubset(of other: Self) -> Bool {
    guard self.count < other.count else { return false }
    return isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isStrictSubset<Value>(
    of other: TreeDictionary<Element, Value>.Keys
  ) -> Bool {
    guard self.count < other.count else { return false }
    return isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: A sequence of elements, some of whose elements may
  ///    appear more than once.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: In the worst case, this makes O(*n*) calls to
  ///    `self.contains` (where *n* is the number of elements in `other`),
  ///    and it constructs a temporary persistent set containing every
  ///    element of the sequence.
  @inlinable
  public func isStrictSubset(of other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return isStrictSubset(of: other)
    }

    do {
      var it = self.makeIterator()
      guard let first = it.next() else {
        return other.contains(where: { _ in true })
      }
      if let match = other._customContainsEquatableElement(first) {
        // Fast path: the sequence has fast containment checks.
        guard match else { return false }
        while let item = it.next() {
          guard other.contains(item) else { return false }
        }
        return !other.allSatisfy { self.contains($0) }
      }
    }

    // FIXME: Would making this a BitSet of seen positions be better?
    var seen: _Node = ._emptyNode()
    var doneCollecting = false
    var isStrict = false
    var it = other.makeIterator()
    while let item = it.next() {
      let hash = _Hash(item)
      if self._root.containsKey(.top, item, hash) {
        if
          !doneCollecting,
          seen.insert(.top, (item, ()), hash).inserted,
          seen.count == self.count
        {
          if isStrict { return true }
          // Stop collecting seen items -- we just need to decide
          // strictness now.
          seen = ._emptyNode()
          doneCollecting = true
        }
      } else {
        isStrict = true
        if doneCollecting { return true }
      }
    }

    return false
  }
}
