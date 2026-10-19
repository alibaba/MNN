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

// `OrderedSet` does not directly conform to `SetAlgebra` because its definition
// of equality conflicts with `SetAlgebra` requirements. However, it still
// implements most `SetAlgebra` requirements (except `insert`, which is replaced
// by `append`).
//
// `OrderedSet` also provides an `unordered` view that explicitly conforms to
// `SetAlgebra`. That view implements `Equatable` by ignoring element order,
// so it can satisfy `SetAlgebra` requirements.

extension OrderedSet {
  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSuperset(of other: Self) -> Bool {
    self.count > other.count && other.isSubset(of: self)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     a.isStrictSuperset(of: b.unordered) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public func isStrictSuperset(of other: UnorderedView) -> Bool {
    isStrictSuperset(of: other._base)
  }

  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given set.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Set = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSuperset(of other: Set<Element>) -> Bool {
    self.count > other.count && other.isSubset(of: self)
  }

  /// Returns a Boolean value that indicates whether the set is a strict
  /// superset of the given sequence.
  ///
  /// Set *A* is a strict superset of another set *B* if every member of *B* is
  /// also a member of *A* and *A* contains at least one element that is *not*
  /// a member of *B*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Array = [4, 2, 1]
  ///     a.isStrictSuperset(of: b) // true
  ///
  /// - Parameter other: A finite sequence of elements, some of whose members
  ///    may appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if `self` is a strict superset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isStrictSuperset(
    of other: some Sequence<Element>
  ) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return self.isStrictSuperset(of: other)
    }
    if let other = _specialize(other, for: Set<Element>.self) {
      return self.isStrictSuperset(of: other)
    }

    var it = self.makeIterator()
    guard let first = it.next() else { return false }
    if let match = other._customContainsEquatableElement(first) {
      // Fast path: the sequence has fast containment checks.
      guard other.allSatisfy({ self.contains($0) }) else { return false }
      guard match else { return true }
      while let item = it.next() {
        guard other.contains(item) else { return true }
      }
      return false
    }

    return _UnsafeBitSet.withTemporaryBitSet(capacity: count) { seen in
      // Mark elements in `self` that we've seen in `other`.
      var c = 0
      for item in other {
        guard let index = _find(item).index else {
          return false
        }
        if seen.insert(index) {
          c &+= 1
          if c == self.count {
            // We've seen enough.
            return false
          }
        }
      }
      return c < self.count
    }
  }
}

