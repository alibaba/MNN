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
  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet = [0, 2, 4, 6]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formIntersection(_ other: Self) {
    other._read { source in
      if source.wordCount < _storage.count {
        self._storage.removeLast(_storage.count - source.wordCount)
      }
      _updateThenShrink { target, shrink in
        target.combineSharedPrefix(
          with: source, using: { $0.formIntersection($1) })
      }
    }
  }

  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other: BitSet.Counted = [0, 2, 4, 6]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A bit set.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in either set.
  public mutating func formIntersection(_ other: BitSet.Counted) {
    formIntersection(other._bits)
  }

  /// Removes the elements of this set that aren't also in the given range.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     set.formIntersection(-10 ..< 3)
  ///     // set is now [3, 4]
  ///
  /// - Parameter other: A range of integers.
  ///
  /// - Complexity: O(*max*), where *max* is the largest item in `self`.
  public mutating func formIntersection(_ other: Range<Int>) {
    let other = other._clampedToUInt()
    guard let last = other.last else {
      self = BitSet()
      return
    }
    let lastWord = _UnsafeHandle.Index(last).word
    if _storage.count - lastWord - 1 > 0 {
      _storage.removeLast(_storage.count - lastWord - 1)
    }
    _updateThenShrink { handle, shrink in
      handle.formIntersection(other)
    }
  }

  /// Removes the elements of this set that aren't also in the given sequence.
  ///
  ///     var set: BitSet = [1, 2, 3, 4]
  ///     let other: Set<Int> = [6, 4, 2, 0]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// - Parameter other: A sequence of integers.
  ///
  /// - Complexity: O(*max*) + *k*, where *max* is the largest item in `self`,
  ///     and *k* is the complexity of iterating over all elements in `other`.
  @inlinable
  public mutating func formIntersection(
    _ other: __owned some Sequence<Int>
  ) {
    if let other = _specialize(other, for: Range<Int>.self) {
      formIntersection(other)
      return
    }
    // Note: BitSet & BitSet.Counted are handled in the BitSet initializer below
    formIntersection(BitSet(_validMembersOf: other))
  }
}
