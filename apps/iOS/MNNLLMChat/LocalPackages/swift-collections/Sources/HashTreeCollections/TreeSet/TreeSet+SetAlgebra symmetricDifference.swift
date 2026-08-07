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
  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     let c = a.symmetricDifference(b)
  ///     // `c` is some permutation of `[0, 1, 3, 6]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public func symmetricDifference(_ other: __owned Self) -> Self {
    let branch = _root.symmetricDifference(.top, other._root)
    guard let branch = branch else { return self }
    let root = branch.finalize(.top)
    root._fullInvariantCheck()
    return TreeSet(_new: root)
  }

  /// Returns a new set with the elements that are either in this set or in
  /// `other`, but not in both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     let c = a.symmetricDifference(b.keys)
  ///     // `c` is some permutation of `[0, 1, 3, 6]`
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public func symmetricDifference<Value>(
    _ other: __owned TreeDictionary<Element, Value>.Keys
  ) -> Self {
    let branch = _root.symmetricDifference(.top, other._base._root)
    guard let branch = branch else { return self }
    let root = branch.finalize(.top)
    root._fullInvariantCheck()
    return TreeSet(_new: root)
  }

  /// Returns a new set with the elements that are either in this set or in
  /// the given sequence, but not in both.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6, 6, 2]
  ///     let c = a.symmetricDifference(b)
  ///     // `c` is some permutation of `[0, 1, 3, 6]`
  ///
  /// In case the sequence contains duplicate elements, only the first instance
  /// matters -- the second and subsequent instances are ignored by this method.
  ///
  /// - Parameter other: A finite sequence of elements, possibly containing
  ///     duplicates.
  ///
  /// - Complexity: Expected complexity is O(`self.count` + `other.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable
  public func symmetricDifference(
    _ other: __owned some Sequence<Element>
  ) -> Self {
    if let other = _specialize(other, for: Self.self) {
      return symmetricDifference(other)
    }

    if other is _UniqueCollection {
      // Fast path: we can do a simple in-place loop.
      var root = self._root
      for item in other {
        let hash = _Hash(item)
        var state = root.prepareValueUpdate(item, hash)
        if state.found {
          state.value = nil
        } else {
          state.value = ()
        }
        root.finalizeValueUpdate(state)
      }
      return Self(_new: root)
    }

    // If `other` may contain duplicates, we need to be more
    // careful (and slower).
    let other = TreeSet(other)
    return self.symmetricDifference(other)
  }
}
