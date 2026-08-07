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
  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: A set of elements.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public mutating func formIntersection(_ other: Self) {
    self = self.intersection(other)
  }

  // Generalizations

  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.formIntersection(other)
  ///     // set is now [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: A set of elements.
  ///
  /// - Complexity: Expected to be O(`self.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func formIntersection(_ other: UnorderedView) {
    formIntersection(other._base)
  }

  /// Removes the elements of this set that aren't also in the given sequence.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     set.formIntersection([6, 4, 2, 0] as Array)
  ///     // set is now [2, 4]
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(*n*) on average where *n* is the number of
  ///    elements in `other`, if `Element` implements high-quality hashing.
  @inlinable
  public mutating func formIntersection(
    _ other: some Sequence<Element>
  ) {
    self = self.intersection(other)
  }
}
