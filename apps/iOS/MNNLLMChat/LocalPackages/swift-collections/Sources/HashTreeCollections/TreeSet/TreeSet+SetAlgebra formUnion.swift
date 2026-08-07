//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension TreeSet {
  /// Adds the elements of the given set to this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     a.formUnion(b)
  ///     // `a` is now some permutation of `[0, 1, 2, 3, 4, 6]`
  ///
  /// For values that are members of both sets, this operation preserves the
  /// instances that were originally in `self`. (This matters if equal members
  /// can be distinguished by comparing their identities, or by some other
  /// means.)
  ///
  /// - Parameter other: The set of elements to insert.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formUnion(_ other: __owned Self) {
    self = union(other)
  }

  /// Adds the elements of the given keys view of a persistent dictionary
  /// to this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     a.formUnion(b.keys)
  ///     // `a` is now some permutation of `[0, 1, 2, 3, 4, 6]`
  ///
  /// For values that are members of both inputs, this operation preserves the
  /// instances that were originally in `self`. (This matters if equal members
  /// can be distinguished by comparing their identities, or by some other
  /// means.)
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formUnion<Value>(
    _ other: __owned TreeDictionary<Element, Value>.Keys
  ) {
    self = union(other)
  }

  /// Adds the elements of the given sequence to this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6, 0, 2]
  ///     a.formUnion(b)
  ///     // `a` is now some permutation of `[0, 1, 2, 3, 4, 6]`
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
  /// - Parameter other: An arbitrary finite sequence of items,
  ///    possibly containing duplicate values.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formUnion(_ other: __owned some Sequence<Element>) {
    self = union(other)
  }
}
