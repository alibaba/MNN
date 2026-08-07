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
  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(min(`self.count`, `other.count`)) on
  ///    average, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(with other: Self) -> Bool {
    guard !self.isEmpty && !other.isEmpty else { return true }
    if self.count <= other.count {
      for item in self {
        if other.contains(item) { return false }
      }
    } else {
      for item in other {
        if self.contains(item) { return false }
      }
    }
    return true
  }

  // Generalizations

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [5, 6]
  ///     a.isDisjoint(with: b.unordered) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(min(`self.count`, `other.count`)) on
  ///    average, if `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public func isDisjoint(with other: UnorderedView) -> Bool {
    isDisjoint(with: other._base)
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given set.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Set = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(min(`self.count`, `other.count`)) on
  ///    average, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(with other: Set<Element>) -> Bool {
    guard !self.isEmpty && !other.isEmpty else { return true }
    if self.count <= other.count {
      for item in self {
        if other.contains(item) { return false }
      }
    } else {
      for item in other {
        if self.contains(item) { return false }
      }
    }
    return true
  }

  /// Returns a Boolean value that indicates whether the set has no members in
  /// common with the given sequence.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Array = [5, 6]
  ///     a.isDisjoint(with: b) // true
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: `true` if `self` has no elements in common with `other`;
  ///   otherwise, `false`.
  ///
  /// - Complexity: Expected to be O(*n*) on average, where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public func isDisjoint(
    with other: some Sequence<Element>
  ) -> Bool {
    guard !self.isEmpty else { return true }
    for item in other {
      if self.contains(item) { return false }
    }
    return true
  }
}
