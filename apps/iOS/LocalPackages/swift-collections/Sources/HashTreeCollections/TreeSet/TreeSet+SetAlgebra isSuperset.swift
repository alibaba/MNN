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
  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isSuperset(of other: Self) -> Bool {
    other._root.isSubset(.top, of: self._root)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing. The implementation is careful to make
  ///    the best use of hash tree structure to minimize work when possible,
  ///    e.g. by skipping over parts of the input trees.
  @inlinable
  public func isSuperset<Value>(
    of other: TreeDictionary<Element, Value>.Keys
  ) -> Bool {
    other._base._root.isSubset(.top, of: self._root)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: A sequence of elements, some of whose elements may
  ///    appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: O(*n*) calls to `self.contains`, where *n* is the number
  ///    of elements in `other`.
  @inlinable
  public func isSuperset(of other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return isSuperset(of: other)
    }

    return other.allSatisfy { self.contains($0) }
  }
}
