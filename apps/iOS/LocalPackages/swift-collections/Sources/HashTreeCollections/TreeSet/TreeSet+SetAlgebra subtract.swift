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
  /// Removes the elements of the given set from this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     a.subtract(b)
  ///     // `a` is now some permutation of `[1, 3]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func subtract(_ other: Self) {
    // FIXME: Implement in-place reductions
    self = subtracting(other)
  }

  /// Removes the elements of the given keys view of a persistent dictionary
  /// from this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     a.subtract(b.keys)
  ///     // `a` is now some permutation of `[1, 3]`
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public mutating func subtract<Value>(
    _ other: TreeDictionary<Element, Value>.Keys
  ) {
    // FIXME: Implement in-place reductions
    self = subtracting(other)
  }

  /// Removes the elements of the given sequence from this set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6]
  ///     a.subtract(b)
  ///     // `a` is now some permutation of `[1, 3]`
  ///
  /// - Parameter other: An arbitrary finite sequence.
  ///
  /// - Complexity: O(*n*) where *n* is the number of elements in `other`,
  ///    as long as `Element` properly implements hashing.
  @inlinable
  public mutating func subtract(_ other: some Sequence<Element>) {
    self = subtracting(other)
  }
}
