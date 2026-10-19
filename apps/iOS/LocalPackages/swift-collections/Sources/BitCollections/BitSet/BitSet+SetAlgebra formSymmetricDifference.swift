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
  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formSymmetricDifference(_ other: Self) {
    _ensureCapacity(limit: other._capacity)
    _updateThenShrink { target, shrink in
      other._read { source in
        target.combineSharedPrefix(
          with: source, using: { $0.formSymmetricDifference($1) })
      }
    }
  }

  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formSymmetricDifference(_ other: BitSet.Counted) {
    formSymmetricDifference(other._bits)
  }

  /// Replace this set with the elements contained in this set or the given
  /// range of integers, but not both.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     set.formSymmetricDifference(3 ..< 7)
  ///     // set is now [1, 2, 5, 6]
  ///
  /// - Parameter other: A range of nonnegative integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either input.
  public mutating func formSymmetricDifference(_ other: Range<Int>) {
    guard let other = other._toUInt() else {
      preconditionFailure("Invalid range")
    }
    guard !other.isEmpty else { return }
    _ensureCapacity(limit: other.upperBound)
    _updateThenShrink { handle, shrink in
      handle.formSymmetricDifference(other)
    }
  }

  /// Replace this set with the elements contained in this set or the given
  /// sequence, but not both.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other = [6, 4, 2, 0, 2, 0]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [0, 1, 3, 6]
  ///
  /// - Parameter other: A sequence of nonnegative integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in either
  ///    input, and *k* is the complexity of iterating over all elements in
  ///    `other`.
  @inlinable
  public mutating func formSymmetricDifference(
    _ other: __owned some Sequence<Int>
  ) {
    if let other = _specialize(other, for: Range<Int>.self) {
      formSymmetricDifference(other)
      return
    }
    // Note: BitSet & BitSet.Counted are handled in the BitSet initializer below
    formSymmetricDifference(BitSet(other))
  }
}
