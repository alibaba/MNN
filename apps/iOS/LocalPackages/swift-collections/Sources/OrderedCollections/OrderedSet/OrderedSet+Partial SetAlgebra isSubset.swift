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
  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSubset(of other: Self) -> Bool {
    guard other.count >= self.count else { return false }
    for item in self {
      guard other.contains(item) else { return false }
    }
    return true
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isSubset(of: a.unordered) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public func isSubset(of other: UnorderedView) -> Bool {
    isSubset(of: other._base)
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the given set.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Set = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSubset(of other: Set<Element>) -> Bool {
    guard other.count >= self.count else { return false }
    for item in self {
      guard other.contains(item) else { return false }
    }
    return true
  }

  /// Returns a Boolean value that indicates whether this set is a subset of
  /// the elements in the given sequence.
  ///
  /// Set *A* is a subset of another set *B* if every member of *A* is also a
  /// member of *B*, ignoring the order they appear in the two sets.
  ///
  ///     let a: Array = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     b.isSubset(of: a) // true
  ///
  /// - Parameter other: A finite sequence.
  ///
  /// - Returns: `true` if the set is a subset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func isSubset(
    of other: some Sequence<Element>
  ) -> Bool {
    guard !isEmpty else { return true }

    if let other = _specialize(other, for: Self.self) {
      return isSubset(of: other)
    }

    var it = self.makeIterator()
    let first = it.next()!
    if let match = other._customContainsEquatableElement(first) {
      // Fast path: the sequence has fast containment checks.
      guard match else { return false }
      while let item = it.next() {
        guard other.contains(item) else { return false }
      }
      return true
    }

    return _UnsafeBitSet.withTemporaryBitSet(capacity: count) { seen in
      // Mark elements in `self` that we've seen in `other`.
      var c = 0
      for item in other {
        if let index = _find(item).index {
          if seen.insert(index) {
            c &+= 1
            if c == self.count {
              // We've seen enough.
              return true
            }
          }
        }
      }
      return false
    }
  }
}
