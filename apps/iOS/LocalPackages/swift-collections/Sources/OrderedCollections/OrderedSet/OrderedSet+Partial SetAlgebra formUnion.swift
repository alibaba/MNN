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
  /// Adds the elements of the given set to this set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the set, in the order they appear in `other`.
  ///
  ///     var a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [0, 2, 4, 6]
  ///     a.formUnion(b)
  ///     // `a` is now `[1, 2, 3, 4, 0, 6]`
  ///
  /// For values that are members of both sets, this operation preserves the
  /// instances that were originally in `self`. (This matters if equal members
  /// can be distinguished by comparing their identities, or by some other
  /// means.)
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: Expected to be O(`other.count`) on average, if `Element`
  ///    implements high-quality hashing.
  @inlinable
  public mutating func formUnion(_ other: __owned Self) {
    append(contentsOf: other)
  }

  // Generalizations

  /// Adds the elements of the given set to this set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the set, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: OrderedSet = [0, 2, 4, 6]
  ///     a.formUnion(b.unordered)
  ///     // a is now [1, 2, 3, 4, 0, 6]
  ///
  /// For values that are members of both inputs, this operation preserves the
  /// instances that were originally in `self`. (This matters if equal members
  /// can be distinguished by comparing their identities, or by some other
  /// means.)
  ///
  /// - Parameter other: The set of elements to add.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public mutating func formUnion(_ other: __owned UnorderedView) {
    formUnion(other._base)
  }

  /// Adds the elements of the given sequence to this set.
  ///
  /// Members of `other` that aren't already in `self` get appended to the end
  /// of the set, in the order they appear in `other`.
  ///
  ///     let a: OrderedSet = [1, 2, 3, 4]
  ///     let b: Array = [0, 2, 4, 6]
  ///     a.formUnion(b)
  ///     // a is now [1, 2, 3, 4, 0, 6]
  ///
  /// For values that are members of both inputs, this operation preserves the
  /// instances that were originally in `self`. (This matters if equal members
  /// can be distinguished by comparing their identities, or by some other
  /// means.)
  ///
  /// If some of the values that are missing from `self` have multiple copies
  /// in `other`, then the result of this function always contains the first
  /// instances in the sequence -- the second and subsequent copies are ignored.
  ///
  /// - Parameter other: A finite sequence of elements.
  ///
  /// - Complexity: Expected to be O(`self.count` + `other.count`) on average,
  ///    if `Element` implements high-quality hashing.
  @inlinable
  public mutating func formUnion(
    _ other: __owned some Sequence<Element>
  ) {
    append(contentsOf: other)
  }
}

