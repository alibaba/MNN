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
  /// Returns a new set containing the elements of this set that do not occur
  /// in the given other set.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func subtracting(_ other: Self) -> Self {
    self._read { first in
      other._read { second in
        Self(
          _combining: (first, second),
          includingTail: true,
          using: { $0.subtracting($1) })
      }
    }
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given other set.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another bit set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public func subtracting(_ other: BitSet.Counted) -> Self {
    subtracting(other._bits)
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given range of integers.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     set.subtracting(-10 ..< 3) // [3, 4]
  ///
  /// - Parameter other: A range of arbitrary integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in self.
  public func subtracting(_ other: Range<Int>) -> Self {
    var result = self
    result.subtract(other)
    return result
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given sequence of integers.
  ///
  ///     let set: BitSet = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, -2, -4]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: A sequence of arbitrary integers.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///    and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public func subtracting(_ other: __owned some Sequence<Int>) -> Self {
    var result = self
    result.subtract(other)
    return result
  }
}
