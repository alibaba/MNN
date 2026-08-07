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

// `OrderedSet` does not directly conform to `SetAlgebra` because its definition
// of equality conflicts with `SetAlgebra` requirements. However, it still
// implements most `SetAlgebra` requirements (except `insert`, which is replaced
// by `append`).
//
// `OrderedSet` also provides an `unordered` view that explicitly conforms to
// `SetAlgebra`. That view implements `Equatable` by ignoring element order,
// so it can satisfy `SetAlgebra` requirements.

extension OrderedSet {
  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  /// On return, `self` contains elements originally from `self` followed by
  /// elements in `other`, in the same order they appeared in the input values.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.formSymmetricDifference(other)
  ///     // set is now [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  public mutating func formSymmetricDifference(_ other: __owned Self) {
    self = self.symmetricDifference(other)
  }

  // Generalizations

  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  /// On return, `self` contains elements originally from `self` followed by
  /// elements in `other`, in the same order they appeared in the input values.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.formSymmetricDifference(other.unordered)
  ///     // set is now [1, 3, 6, 0]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func formSymmetricDifference(_ other: __owned UnorderedView) {
    formSymmetricDifference(other._base)
  }

  /// Replace this set with the elements contained in this set or the given
  /// sequence, but not both.
  ///
  /// On return, `self` contains elements originally from `self` followed by
  /// elements in `other`, in the same order they first appeared in the input
  /// values.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     set.formSymmetricDifference([6, 4, 2, 0] as Array)
  ///     // set is now [1, 3, 6, 0]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average where *n* is
  ///    the number of elements in `other`, if `Element` implements high-quality
  ///    hashing.
  @inlinable
  public mutating func formSymmetricDifference(
    _ other: __owned some Sequence<Element>
  ) {
    self = self.symmetricDifference(other)
  }
}
