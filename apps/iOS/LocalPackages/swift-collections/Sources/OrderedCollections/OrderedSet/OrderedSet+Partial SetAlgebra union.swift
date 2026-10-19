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
  /// Returns a new set with the elements of both this and the given set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the result, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [0, 2, 4, 6]
  ///     a.union(b) // [1, 2, 3, 4, 0, 6]
  ///
  /// - Parameter other: The set of elements to add.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func union(_ other: __owned Self) -> Self {
    var result = self
    result.formUnion(other)
    return result
  }

  // Generalizations

  /// Returns a new set with the elements of both this and the given set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the result, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [0, 2, 4, 6]
  ///     a.union(b.unordered) // [1, 2, 3, 4, 0, 6]
  ///
  /// - Parameter other: The set of elements to add.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public __consuming func union(_ other: __owned UnorderedView) -> Self {
    union(other._base)
  }

  /// Returns a new set with the elements of both this and the given set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the result, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Array = [0, 2, 4, 6]
  ///     a.union(b) // [1, 2, 3, 4, 0, 6]
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func union(
    _ other: __owned some Sequence<Element>
  ) -> Self {
    var result = self
    result.formUnion(other)
    return result
  }

  /// Returns a new set with the contents of a sequence appended to the end of the set, excluding
  /// elements that are already members.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [0, 2, 4, 6]
  ///     a.union(b) // [1, 2, 3, 4, 0, 6]
  ///
  /// This is functionally equivalent to `self.union(elements)`, but it's
  /// more explicit about how the new members are ordered in the new set.
  ///
  /// - Parameter elements: A finite sequence of elements to append.
  ///
  /// - Complexity: Expected to be O(`self.count` + `elements.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public __consuming func appending(
    contentsOf elements: __owned some Sequence<Element>
  ) -> Self {
    var result = self
    result.append(contentsOf: elements)
    return result
  }
}

