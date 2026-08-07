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
  /// Returns a new set with the elements that are common to both this set and
  /// the provided other one, in the order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.intersection(other) // [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public __consuming func intersection(_ other: Self) -> Self {
    var result = Self()
    for item in self {
      if other.contains(item) {
        result._appendNew(item)
      }
    }
    result._checkInvariants()
    return result
  }

  // Generalizations

  /// Returns a new set with the elements that are common to both this set and
  /// the provided other one, in the order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.intersection(other) // [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func intersection(_ other: UnorderedView) -> Self {
    intersection(other._base)
  }

  /// Returns a new set with the elements that are common to both this set and
  /// the provided sequence, in the order they appear in `self`.
  ///
  ///     let set: OrderedSet = [1, 2, 3, 4]
  ///     set.intersection([6, 4, 2, 0] as Array) // [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Returns: A new set.
  ///
  /// - Complexity: Expected to be O(*n*) on average where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func intersection(
    _ other: some Sequence<Element>
  ) -> Self {
    _UnsafeBitSet.withTemporaryBitSet(capacity: self.count) { bitset in
      for item in other {
        if let index = self._find_inlined(item).index {
          bitset.insert(index)
        }
      }
      return self._extractSubset(using: bitset)
    }
  }
}

