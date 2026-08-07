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
  /// Replace this set with the elements contained in this set or the given
  /// set, but not both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     a.formSymmetricDifference(b)
  ///     // `a` is now some permutation of `[0, 1, 3, 6]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formSymmetricDifference(_ other: __owned Self) {
    self = symmetricDifference(other)
  }

  /// Replace this set with the elements contained in this set or the given
  /// keys view of a persistent dictionary, but not both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     a.formSymmetricDifference(b.keys)
  ///     // `a` is now some permutation of `[0, 1, 3, 6]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func formSymmetricDifference<Value>(
    _ other: __owned TreeDictionary<Element, Value>.Keys
  ) {
    self = symmetricDifference(other)
  }

  /// Replace this set with the elements contained in this set or the given
  /// sequence, but not both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6, 2, 4, 6]
  ///     a.formSymmetricDifference(b)
  ///     // `a` is now some permutation of `[0, 1, 3, 6]`
  ///
  /// - Parameter other: A finite sequence of elements, possibly containing
  ///     duplicate values.
  @inlinable
  public mutating func formSymmetricDifference(
    _ other: __owned some Sequence<Element>
  ) {
    self = symmetricDifference(other)
  }
}
