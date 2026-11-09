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
  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  /// The result contains elements from `self` followed by elements in `other`,
  /// in the same order they appeared in the original sets.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.symmetricDifference(other) // [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public __consuming func symmetricDifference(_ other: __owned Self) -> Self {
    _UnsafeBitSet.withTemporaryBitSet(capacity: self.count) { bitset1 in
      _UnsafeBitSet.withTemporaryBitSet(capacity: other.count) { bitset2 in
        bitset1.insertAll(upTo: self.count)
        for item in other {
          if let index = self._find(item).index {
            bitset1.remove(index)
          }
        }
        bitset2.insertAll(upTo: other.count)
        for item in self {
          if let index = other._find(item).index {
            bitset2.remove(index)
          }
        }
        var result = self._extractSubset(using: bitset1,
                                         extraCapacity: bitset2.count)
        for offset in bitset2 {
          result._appendNew(other._elements[Int(bitPattern: offset)])
        }
        result._checkInvariants()
        return result
      }
    }
  }

  // Generalizations

  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  /// The result contains elements from `self` followed by elements in `other`,
  /// in the same order they appeared in the original sets.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.symmetricDifference(other.unordered) // [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func symmetricDifference(
    _ other: __owned UnorderedView
  ) -> Self {
    symmetricDifference(other._base)
  }

  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  /// The result contains elements from `self` followed by elements in `other`,
  /// in the same order they appeared in the original input values.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: Array = [6, 4, 2, 0]
  ///     set.symmetricDifference(other) // [1, 3, 6, 0]
  ///
  /// In case the sequence contains duplicate elements, only the first instance
  /// matters -- the second and subsequent instances are ignored by this method.
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average where *n* is
  ///    the number of elements in `other`, if `Element` implements high-quality
  ///    hashing.
  @inlinable
  public __consuming func symmetricDifference(
    _ other: __owned some Sequence<Element>
  ) -> Self {
    _UnsafeBitSet.withTemporaryBitSet(capacity: self.count) { bitset in
      var new = Self()
      bitset.insertAll(upTo: self.count)
      for item in other {
        if let index = self._find(item).index {
          bitset.remove(index)
        } else {
          new.append(item)
        }
      }
      var result = _extractSubset(using: bitset, extraCapacity: new.count)
      for item in new._elements {
        result._appendNew(item)
      }
      result._checkInvariants()
      return result
    }
  }
}
