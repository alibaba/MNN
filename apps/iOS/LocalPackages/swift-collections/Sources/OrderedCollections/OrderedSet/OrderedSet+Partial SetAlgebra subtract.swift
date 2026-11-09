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
  /// Removes the elements of the given set from this set.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.subtract(other)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func subtract(_ other: Self) {
    self = subtracting(other)
  }

  // Generalizations

  /// Removes the elements of the given set from this set.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     let other: OrderedSet = [6, 4, 2, 0]
  ///     set.subtract(other.unordered)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: Another set.
  ///
  /// - Complexity: Expected to be O(`self.count + other.count`) on average, if
  ///    `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func subtract(_ other: UnorderedView) {
    subtract(other._base)
  }

  /// Removes the elements of the given sequence from this set.
  ///
  ///     var set: OrderedSet = [1, 2, 3, 4]
  ///     set.subtract([6, 4, 2, 0] as Array)
  ///     // set is now [1, 3]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + *n*) on average, where *n*
  ///    is the number of elements in `other`, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func subtract(_ other: some Sequence<Element>) {
    self = _subtracting(other)
  }
}
