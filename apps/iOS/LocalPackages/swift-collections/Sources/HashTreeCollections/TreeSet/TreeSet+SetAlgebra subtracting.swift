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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension TreeSet {
  /// Returns a new set containing the elements of this set that do not occur
  /// in the given other set.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     let c = a.subtracting(b)
  ///     // `c` is some permutation of `[1, 3]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public __consuming func subtracting(_ other: Self) -> Self {
    _subtracting(other._root)
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given keys view of a persistent dictionary.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     let c = a.subtracting(b.keys)
  ///     // `c` is some permutation of `[1, 3]`
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public __consuming func subtracting<V>(
    _ other: TreeDictionary<Element, V>.Keys
  ) -> Self {
    _subtracting(other._base._root)
  }

  @inlinable
  internal __consuming func _subtracting<V>(
    _ other: _HashNode<Element, V>
  ) -> Self {
    guard let r = _root.subtracting(.top, other) else { return self }
    return Self(_new: r)
  }

  /// Returns a new set containing the elements of this set that do not occur
  /// in the given sequence.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6]
  ///     let c = a.subtracting(b)
  ///     // `c` is some permutation of `[1, 3]`
  ///
  /// - Parameter other: An arbitrary finite sequence.
  ///
  /// - Complexity: O(*n*) where *n* is the number of elements in `other`,
  ///    as long as `Element` properly implements hashing.
  @inlinable
  public __consuming func subtracting(_ other: some Sequence<Element>) -> Self {
    if let other = _specialize(other, for: Self.self) {
      return subtracting(other)
    }

    guard let first = self.first else { return Self() }
    if other._customContainsEquatableElement(first) != nil {
      // Fast path: the sequence has fast containment checks.
      return self.filter { !other.contains($0) }
    }

    var root = self._root
    for item in other {
      let hash = _Hash(item)
      _ = root.remove(.top, item, hash)
    }
    return Self(_new: root)
  }
}
