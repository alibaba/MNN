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
  /// Returns a new set with the elements that are common to both this set and
  /// the provided other one.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeSet = [0, 2, 4, 6]
  ///     let c = a.intersection(b)
  ///     // `c` is some permutation of `[2, 4]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: An arbitrary set of elements.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable @inline(__always)
  public func intersection(_ other: Self) -> Self {
    _intersection(other._root)
  }

  /// Returns a new set with the elements that are common to both this set and
  /// the provided keys view of a persistent dictionary.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b: TreeDictionary = [0: "a", 2: "b", 4: "c", 6: "d"]
  ///     let c = a.intersection(b.keys)
  ///     // `c` is some permutation of `[2, 4]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  @inlinable @inline(__always)
  public func intersection<Value>(
    _ other: TreeDictionary<Element, Value>.Keys
  ) -> Self {
    _intersection(other._base._root)
  }

  @inlinable
  internal func _intersection<V>(_ other: _HashNode<Element, V>) -> Self {
    guard let r = _root.intersection(.top, other) else { return self }
    return Self(_new: r)
  }

  /// Returns a new set with the elements that are common to both this set and
  /// the provided sequence.
  ///
  ///     var a: TreeSet = [1, 2, 3, 4]
  ///     let b = [0, 2, 4, 6]
  ///     let c = a.intersection(b)
  ///     // `c` is some permutation of `[2, 4]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: An arbitrary finite sequence of items,
  ///    possibly containing duplicate values.
  @inlinable
  public func intersection(
    _ other: some Sequence<Element>
  ) -> Self {
    if let other = _specialize(other, for: Self.self) {
      return intersection(other)
    }

    guard let first = self.first else { return Self() }
    if other._customContainsEquatableElement(first) != nil {
      // Fast path: the sequence has fast containment checks.
      return self.filter { other.contains($0) }
    }

    var result: _Node = ._emptyNode()
    for item in other {
      let hash = _Hash(item)
      if let r = self._root.lookup(.top, item, hash) {
        let itemInSelf = _UnsafeHandle.read(r.node) { $0[item: r.slot] }
        _ = result.updateValue(.top, forKey: itemInSelf.key, hash) {
          $0.initialize(to: itemInSelf)
        }
      }
    }
    return Self(_new: result)
  }
}
