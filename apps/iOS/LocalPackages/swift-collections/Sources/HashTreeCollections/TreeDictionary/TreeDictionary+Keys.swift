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

extension TreeDictionary {
  /// A view of a persistent dictionary’s keys, as a standalone collection.
  @frozen
  public struct Keys {
    @usableFromInline
    internal typealias _Node = _HashNode<Key, Value>

    @usableFromInline
    internal var _base: TreeDictionary

    @inlinable
    internal init(_base: TreeDictionary) {
      self._base = _base
    }
  }
  
  /// A collection containing just the keys of the dictionary.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var keys: Keys {
    Keys(_base: self)
  }
}

extension TreeDictionary.Keys: Sendable
where Key: Sendable, Value: Sendable {}

extension TreeDictionary.Keys: _UniqueCollection {}

extension TreeDictionary.Keys: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: self)
  }
}

extension TreeDictionary.Keys: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension TreeDictionary.Keys: Sequence {
  /// The element type of the collection.
  public typealias Element = Key

  /// The type that allows iteration over the elements of the keys view
  /// of a persistent dictionary.
  @frozen
  public struct Iterator: IteratorProtocol {
    public typealias Element = Key

    @usableFromInline
    internal var _base: TreeDictionary.Iterator

    @inlinable
    internal init(_base: TreeDictionary.Iterator) {
      self._base = _base
    }

    @inlinable
    public mutating func next() -> Element? {
      _base.next()?.key
    }
  }

  @inlinable
  public func makeIterator() -> Iterator {
    Iterator(_base: _base.makeIterator())
  }

  @inlinable
  public func _customContainsEquatableElement(
    _ element: Element
  ) -> Bool? {
    _base._root.containsKey(.top, element, _Hash(element))
  }

  /// Returns a Boolean value that indicates whether the given key exists
  /// in the dictionary.
  ///
  /// - Parameter element: A key to look for in the dictionary/
  ///
  /// - Returns: `true` if `element` exists in the set; otherwise, `false`.
  ///
  /// - Complexity: This operation is expected to perform O(1) hashing and
  ///    comparison operations on average, provided that `Element` implements
  ///    high-quality hashing.
  @inlinable
  public func contains(_ element: Element) -> Bool {
    _base._root.containsKey(.top, element, _Hash(element))
  }
}

extension TreeDictionary.Keys.Iterator: Sendable
where Key: Sendable, Value: Sendable {}

extension TreeDictionary.Keys: Collection {
  public typealias Index = TreeDictionary.Index

  @inlinable
  public var isEmpty: Bool { _base.isEmpty }

  @inlinable
  public var count: Int { _base.count }

  @inlinable
  public var startIndex: Index { _base.startIndex }

  @inlinable
  public var endIndex: Index { _base.endIndex }

  @inlinable
  public subscript(index: Index) -> Element {
    _base[index].key
  }

  @inlinable
  public func formIndex(after i: inout Index) {
    _base.formIndex(after: &i)
  }

  @inlinable
  public func index(after i: Index) -> Index {
    _base.index(after: i)
  }

  @inlinable
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    _base.index(i, offsetBy: distance)
  }

  @inlinable
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    _base.index(i, offsetBy: distance, limitedBy: limit)
  }

  @inlinable
  public func distance(from start: Index, to end: Index) -> Int {
    _base.distance(from: start, to: end)
  }
}

#if false
extension TreeDictionary.Keys: BidirectionalCollection {
  // Note: Let's not do this. `BidirectionalCollection` would imply that
  // the ordering of elements would be meaningful, which isn't true for
  // `TreeDictionary.Keys`.
  @inlinable
  public func formIndex(before i: inout Index) {
    _base.formIndex(before: &i)
  }

  @inlinable
  public func index(before i: Index) -> Index {
    _base.index(before: i)
  }
}
#endif

extension TreeDictionary.Keys {
  /// Returns a new keys view with the elements that are common to both this
  /// view and the provided other one.
  ///
  ///     var a: TreeDictionary = ["a": 1, "b": 2, "c": 3]
  ///     let b: TreeDictionary = ["b": 4, "d": 5, "e": 6]
  ///     let c = a.keys.intersection(b.keys)
  ///     // `c` is `["b"]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: The keys view of a persistent dictionary with the same
  ///    `Key` type.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  public func intersection<Value2>(
    _ other: TreeDictionary<Key, Value2>.Keys
  ) -> Self {
    guard let r = _base._root.intersection(.top, other._base._root) else {
      return self
    }
    let d = TreeDictionary(_new: r)
    d._invariantCheck()
    return d.keys
  }

  /// Returns a new keys view with the elements that are common to both this
  /// view and the provided persistent set.
  ///
  ///     var a: TreeDictionary = ["a": 1, "b": 2, "c": 3]
  ///     let b: TreeSet = ["b", "d", "e"]
  ///     let c = a.keys.intersection(b)
  ///     // `c` is `["b"]`
  ///
  /// The result will only contain instances that were originally in `self`.
  /// (This matters if equal members can be distinguished by comparing their
  /// identities, or by some other means.)
  ///
  /// - Parameter other: A persistent set whose `Element` type is `Key`.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  public func intersection(_ other: TreeSet<Key>) -> Self {
    guard let r = _base._root.intersection(.top, other._root) else {
      return self
    }
    let d = TreeDictionary(_new: r)
    d._invariantCheck()
    return d.keys
  }

  /// Returns a new keys view containing the elements of `self` that do not
  /// occur in the provided other one.
  ///
  ///     var a: TreeDictionary = ["a": 1, "b": 2, "c": 3]
  ///     let b: TreeDictionary = ["b": 4, "d": 5, "e": 6]
  ///     let c = a.keys.subtracting(b.keys)
  ///     // `c` is some permutation of `["a", "c"]`
  ///
  /// - Parameter other: The keys view of a persistent dictionary with the same
  ///    `Key` type.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  public func subtracting<Value2>(
    _ other: TreeDictionary<Key, Value2>.Keys
  ) -> Self {
    guard let r = _base._root.subtracting(.top, other._base._root) else {
      return self
    }
    let d = TreeDictionary(_new: r)
    d._invariantCheck()
    return d.keys
  }

  /// Returns a new keys view containing the elements of `self` that do not
  /// occur in the provided persistent set.
  ///
  ///     var a: TreeDictionary = ["a": 1, "b": 2, "c": 3]
  ///     let b: TreeSet = ["b", "d", "e"]
  ///     let c = a.keys.subtracting(b)
  ///     // `c` is some permutation of `["a", "c"]`
  ///
  /// - Parameter other: A persistent set whose `Element` type is `Key`.
  ///
  /// - Complexity: Expected complexity is O(`self.count`) in
  ///     the worst case, if `Element` properly implements hashing.
  ///     However, the implementation is careful to make the best use of
  ///     hash tree structure to minimize work when possible, e.g. by linking
  ///     parts of the input trees directly into the result.
  public func subtracting(_ other: TreeSet<Key>) -> Self {
    guard let r = _base._root.subtracting(.top, other._root) else {
      return self
    }
    let d = TreeDictionary(_new: r)
    d._invariantCheck()
    return d.keys
  }
}

extension TreeDictionary.Keys: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  /// 
  /// Equality is the inverse of inequality. For any values `a` and `b`,
  /// `a == b` implies that `a != b` is `false`.
  ///
  /// - Parameter lhs: A value to compare.
  /// - Parameter rhs: Another value to compare.
  ///
  /// - Complexity: Generally O(`count`), as long as`Element` properly
  ///    implements hashing. That said, the implementation is careful to take
  ///    every available shortcut to reduce complexity, e.g. by skipping
  ///    comparing elements in shared subtrees.
  @inlinable
  public static func == (left: Self, right: Self) -> Bool {
    left._base._root.isEqualSet(to: right._base._root, by: { _, _ in true })
  }
}

extension TreeDictionary.Keys: Hashable {
  /// Hashes the essential components of this value by feeding them into the
  /// given hasher.
  ///
  /// Complexity: O(`count`)
  @inlinable
  public func hash(into hasher: inout Hasher) {
    let copy = hasher
    let seed = copy.finalize()

    var hash = 0
    for member in self {
      hash ^= member._rawHashValue(seed: seed)
    }
    hasher.combine(hash)
  }
}
