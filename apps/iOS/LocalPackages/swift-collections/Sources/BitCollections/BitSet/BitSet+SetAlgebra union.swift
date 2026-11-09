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
  /// Returns a new set with the elements of both this and the given set.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: Self) -> Self {
    self._read { first in
      other._read { second in
        Self(
          _combining: (first, second),
          includingTail: true,
          using: { $0.union($1) })
      }
    }
  }

  /// Returns a new set with the elements of both this and the given set.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: BitSet.Counted) -> Self {
    union(other._bits)
  }

  /// Returns a new set with the elements of both this set and the given
  /// range of integers.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     set.union(3 ..< 7) // [1, 2, 3, 4, 5, 6]
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func union(_ other: Range<Int>) -> Self {
    var result = self
    result.formUnion(other)
    return result
  }

  /// Returns a new set with the elements of both this set and the given
  /// sequence of integers.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.union(other) // [0, 1, 2, 3, 4, 6]
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public func union(_ other: __owned some Sequence<Int>) -> Self {
    if let other = _specialize(other, for: BitSet.self) {
      return union(other)
    }
    if let other = _specialize(other, for: BitSet.Counted.self) {
      return union(other)
    }
    if let other = _specialize(other, for: Range<Int>.self) {
      return union(other)
    }
    var result = self
    result.formUnion(other)
    return result
  }
}
