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
  /// Returns a new set containing the elements of this set that do not occur
  /// in the given set.
  ///
  /// The result contains elements in the same order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.subtracting(other) // [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func subtracting(_ other: Self) -> Self {
    _subtracting(other)
  }

  // Generalizations

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given set.
  ///
  /// The result contains elements in the same order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.subtracting(other.unordered) // [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func subtracting(_ other: UnorderedView) -> Self {
    subtracting(other._base)
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given sequence.
  ///
  /// The result contains elements in the same order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     set.subtracting([6, 4, 2, 0] as Array) // [1, 3]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func subtracting(_ other: some Sequence<Element>) -> Self {
    _subtracting(other)
  }

  @inlinable
  __consuming func _subtracting(_ other: some Sequence<Element>) -> Self {
    guard count > 0 else { return Self() }
    return _UnsafeBitSet.withTemporaryBitSet(capacity: count) { difference in
      difference.insertAll(upTo: count)
      var c = count
      for item in other {
        if let index = self._find(item).index {
          if difference.remove(index) {
            c &-= 1
            if c == 0 {
              return Self()
            }
          }
        }
      }
      assert(c > 0)
      return _extractSubset(using: difference, count: c)
    }
  }
}
