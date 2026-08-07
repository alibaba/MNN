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

// FIXME: These are non-standard extensions generalizing ==.
extension TreeSet {
  /// Returns a Boolean value indicating whether persistent sets are equal. Two
  /// persistent sets are considered equal if they contain the same elements,
  /// but not necessarily in the same order.
  ///
  /// - Parameter other: Another set.
  ///
  /// - Returns: `true` if the set is equal to `other`; otherwise, `false`.
  ///
  /// - Complexity: Generally O(`count`), as long as`Element` properly
  ///    implements hashing. That said, the implementation is careful to take
  ///    every available shortcut to reduce complexity, e.g. by skipping
  ///    comparing elements in shared subtrees.
  @inlinable
  public func isEqualSet(to other: Self) -> Bool {
    _root.isEqualSet(to: other._root, by: { _, _ in true })
  }

  /// Returns a Boolean value indicating whether a persistent set compares equal
  /// to the given keys view of a persistent dictionary. The two input
  /// collections are considered equal if they contain the same elements,
  /// but not necessarily in the same order.
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Returns: `true` if the set contains exactly the same members as `other`;
  ///    otherwise, `false`.
  ///
  /// - Complexity: Generally O(`count`), as long as`Element` properly
  ///    implements hashing. That said, the implementation is careful to take
  ///    every available shortcut to reduce complexity, e.g. by skipping
  ///    comparing elements in shared subtrees.
  @inlinable
  public func isEqualSet<Value>(
    to other: TreeDictionary<Element, Value>.Keys
  ) -> Bool {
    _root.isEqualSet(to: other._base._root, by: { _, _ in true })
  }

  /// Returns a Boolean value indicating whether this persistent set contains
  /// the same elements as the given `other` sequence, but not necessarily
  /// in the same order.
  ///
  /// Duplicate items in `other` do not prevent it from comparing equal to
  /// `self`.
  ///
  ///     let this: TreeSet = [0, 1, 5, 6]
  ///     let that = [5, 5, 0, 1, 1, 6, 5, 0, 1, 6, 6, 5]
  ///
  ///     this.isEqualSet(to: that) // true
  ///
  /// - Parameter other: The keys view of a persistent dictionary.
  ///
  /// - Returns: `true` if the set contains exactly the same members as `other`;
  ///    otherwise, `false`. This function does not consider the order of
  ///    elements and it ignores duplicate items in `other`.
  ///
  /// - Complexity: Generally O(*n*), where *n* is the number of items in
  ///    `other`, as long as`Element` properly implements hashing.
  ///    That said, the implementation is careful to take
  ///    every available shortcut to reduce complexity, e.g. by skipping
  ///    comparing elements in shared subtrees.
  @inlinable
  public func isEqualSet(to other: some Sequence<Element>) -> Bool {
    if let other = _specialize(other, for: Self.self) {
      return isEqualSet(to: other)
    }

    if self.isEmpty {
      return other.allSatisfy { _ in false }
    }

    if other is _UniqueCollection {
      // We don't need to create a temporary set.
      guard other.underestimatedCount <= self.count else { return false }
      var seen = 0
      for item in other {
        guard self.contains(item) else { return false }
        seen &+= 1
      }
      precondition(
        seen <= self.count,
        // Otherwise other.underestimatedCount != other.count
        "Invalid Collection '\(type(of: other))' (bad underestimatedCount)")
      return seen == self.count
    }

    // FIXME: Would making this a BitSet of seen positions be better?
    var seen: _Node? = ._emptyNode()
    var it = other.makeIterator()
    while let item = it.next() {
      let hash = _Hash(item)
      guard self._root.containsKey(.top, item, hash) else { return false }
      _ = seen!.insert(.top, (item, ()), hash) // Ignore dupes
      if seen!.count == self.count {
        // We've seen them all. Stop further accounting.
        seen = nil
        break
      }
    }
    guard seen == nil else { return false }
    while let item = it.next() {
      guard self.contains(item) else { return false }
    }
    return true
  }
}
