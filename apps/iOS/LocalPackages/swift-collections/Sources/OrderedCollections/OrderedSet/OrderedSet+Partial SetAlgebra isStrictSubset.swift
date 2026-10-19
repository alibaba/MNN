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
  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSubset(of other: Self) -> Bool {
    self.count < other.count && self.isSubset(of: other)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a.unordered) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public func isStrictSubset(of other: UnorderedView) -> Bool {
    isStrictSubset(of: other._base)
  }

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given set.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: Set = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isStrictSubset(of other: Set<Element>) -> Bool {
    self.count < other.count && self.isSubset(of: other)
  }

  /// Returns a Boolean value that indicates whether the set is a strict subset
  /// of the given sequence.
  ///
  /// Set *A* is a strict subset of another set *B* if every member of *A* is
  /// also a member of *B* and *B* contains at least one element that is not a
  /// member of *A*. (Ignoring the order the elements appear in the sets.)
  ///
  ///     let a: Array = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isStrictSubset(of: a) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if `self` is a strict subset of `other`; otherwise,
  ///    `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isStrictSubset(
    of other: some Sequence<Element>
  ) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return self.isStrictSubset(of: other)
    }
    if let other = _specialize(other, for: Set<Element>.self) {
      return self.isStrictSubset(of: other)
    }

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

    return _UnsafeBitSet.withTemporaryBitSet(capacity: count) { seen in
      // Mark elements in `self` that we've seen in `other`.
      var isKnownStrict = false
      var c = 0
      for item in other {
        if let index = _find(item).index {
          if seen.insert(index) {
            c &+= 1
            if c == self.count, isKnownStrict {
              // We've seen enough.
              return true
            }
          }
        } else {
          if !isKnownStrict, c == self.count { return true }
          isKnownStrict = true
        }
      }
      return false
    }
  }
}
