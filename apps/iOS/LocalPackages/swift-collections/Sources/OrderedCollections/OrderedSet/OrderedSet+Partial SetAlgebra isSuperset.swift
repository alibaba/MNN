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
  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: Self) -> Bool {
    other.isSubset(of: self)
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Set = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: UnorderedView) -> Bool {
    isSuperset(of: other._base)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given set.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Set = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: Set<Element>) -> Bool {
    guard self.count >= other.count else { return false }
    return _isSuperset(of: other)
  }

  /// Returns a Boolean value that indicates whether this set is a superset of
  /// the given sequence.
  ///
  /// Set *A* is a superset of another set *B* if every member of *B* is also a
  /// member of *A*, ignoring the order they appear in the two sets.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Array = [4, 2, 1]
  ///     a.isSuperset(of: b) // true
  ///
  /// - Parameter other: A finite sequence of elements, some of whose members
  ///    may appear more than once. (Duplicate items are ignored.)
  ///
  /// - Returns: `true` if the set is a superset of `other`; otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(*n*) on average, where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public func isSuperset(of other: some Sequence<Element>) -> Bool {
    _isSuperset(of: other)
  }

  @inlinable
  internal func _isSuperset(of other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return self.isSuperset(of: other)
    }
    for item in other {
      guard self.contains(item) else { return false }
    }
    return true
  }
}
