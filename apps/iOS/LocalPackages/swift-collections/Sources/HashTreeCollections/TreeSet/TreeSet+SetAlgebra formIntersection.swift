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
  /// Removes the elements of this set that aren't also in the given one.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     a.formIntersection(b)
  ///     // `a` is now some permutation of `[2, 4]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formIntersection(_ other: Self) {
    // FIXME: Implement in-place reductions
    self = intersection(other)
  }

  /// Removes the elements of this set that aren't also in the given keys view
  /// of a persistent dictionary.
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formIntersection<Value>(
    _ other: TreeDictionary<Element, Value>.Keys
  ) {
    // FIXME: Implement in-place reductions
    self = intersection(other)
  }

  /// Removes the elements of this set that aren't also in the given sequence.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6]
  ///     a.formIntersection(b)
  ///     // `a` is now some permutation of `[2, 4]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: An arbitrary finite sequence of items,
  ///    possibly containing duplicate values.
  @inlinable
  public mutating func formIntersection(_ other: some Sequence<Element>) {
    self = intersection(other)
  }
}
