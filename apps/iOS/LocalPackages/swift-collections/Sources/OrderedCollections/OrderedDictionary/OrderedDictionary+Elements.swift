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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension OrderedDictionary {
  /// A view of the contents of an ordered dictionary as a random-access
  /// collection.
  @frozen
  public struct Elements {
    @usableFromInline
    internal var _base: OrderedDictionary

    @inlinable
    @inline(__always)
    internal init(_base: OrderedDictionary) {
      self._base = _base
    }
  }
}

extension OrderedDictionary.Elements: Sendable
where Key: Sendable, Value: Sendable {}

extension OrderedDictionary {
  /// A view of the contents of this dictionary as a random-access collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var elements: Elements {
    get {
      Elements(_base: self)
    }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      var elements = Elements(_base: self)
      self = Self()
      defer { self = elements._base }
      yield &elements
    }
  }
}

extension OrderedDictionary.Elements {
  /// A read-only collection view containing the keys in this collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var keys: OrderedSet<Key> {
    _base._keys
  }

  /// A mutable collection view containing the values in this collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var values: OrderedDictionary.Values {
    get {
      _base.values
    }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      var values = OrderedDictionary.Values(_base: _base)
      self = Self(_base: .init())
      defer { self._base = values._base }
      yield &values
    }
  }
}

extension OrderedDictionary.Elements {
  /// Returns the index for the given key.
  ///
  /// If the given key is found in the dictionary, this method returns an index
  /// into the dictionary that corresponds with the key-value pair.
  ///
  ///     let countryCodes: OrderedDictionary = ["BR": "Brazil", "GH": "Ghana", "JP": "Japan"]
  ///     let index = countryCodes.elements.index(forKey: "JP")
  ///
  ///     let (key, value) = countryCodes.elements[index!]
  ///     print("Country code for \(value): '\(key)'.")
  ///     // Prints "Country code for Japan: 'JP'."
  ///
  /// - Parameter key: The key to find in the dictionary.
  ///
  /// - Returns: The index for `key` and its associated value if `key` is in
  ///    the dictionary; otherwise, `nil`.
  ///
  /// - Complexity: Expected to be O(1) on average, if `Key` implements
  ///    high-quality hashing.
  @inlinable
  public func index(forKey key: Key) -> Int? {
    _base.index(forKey: key)
  }
}

extension OrderedDictionary.Elements: Sequence {
  /// The element type of the collection.
  public typealias Element = (key: Key, value: Value)

  @inlinable
  public var underestimatedCount: Int { _base.count }

  @inlinable
  public func makeIterator() -> OrderedDictionary<Key, Value>.Iterator {
    _base.makeIterator()
  }
}

extension OrderedDictionary.Elements: RandomAccessCollection {
  /// The index type for an ordered dictionary: `Int`.
  ///
  /// Indices in `Elements` are integer offsets from the start of the
  /// collection.
  public typealias Index = Int

  /// The type that represents the indices that are valid for subscripting the
  /// `Elements` collection, in ascending order.
  public typealias Indices = Range<Int>

  /// The position of the first element in a nonempty dictionary.
  ///
  /// For an instance of `OrderedDictionary.Elements`, `startIndex` is always
  /// zero. If the dictionary is empty, `startIndex` is equal to `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var startIndex: Int { 0 }

  /// The collection's "past the end" position---that is, the position one
  /// greater than the last valid subscript argument.
  ///
  /// In `OrderedDictionary.Elements`, `endIndex` always equals the count of
  /// elements. If the dictionary is empty, `endIndex` is equal to `startIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var endIndex: Int { _base.count }

  /// Returns the position immediately after the given index.
  ///
  /// The specified index must be a valid index less than `endIndex`, or the
  /// returned value won't be a valid index in the collection.
  ///
  /// - Parameter i: A valid index of the collection.
  ///
  /// - Returns: The index immediately after `i`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func index(after i: Int) -> Int { i + 1 }

  /// Returns the position immediately before the given index.
  ///
  /// The specified index must be a valid index greater than `startIndex`, or
  /// the returned value won't be a valid index in the collection.
  ///
  /// - Parameter i: A valid index of the collection.
  ///
  /// - Returns: The index immediately before `i`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func index(before i: Int) -> Int { i - 1 }

  /// Replaces the given index with its successor.
  ///
  /// The specified index must be a valid index less than `endIndex`, or the
  /// returned value won't be a valid index in the collection.
  ///
  /// - Parameter i: A valid index of the collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func formIndex(after i: inout Int) { i += 1 }

  /// Replaces the given index with its predecessor.
  ///
  /// The specified index must be a valid index greater than `startIndex`, or
  /// the returned value won't be a valid index in the collection.
  ///
  /// - Parameter i: A valid index of the collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func formIndex(before i: inout Int) { i -= 1 }

  /// Returns an index that is the specified distance from the given index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection, or the returned value will not be a valid index.
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`.
  ///
  /// - Returns: An index offset by `distance` from the index `i`. If `distance`
  ///   is positive, this is the same value as the result of `distance` calls to
  ///   `index(after:)`. If `distance` is negative, this is the same value as
  ///   the result of `abs(distance)` calls to `index(before:)`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func index(_ i: Int, offsetBy distance: Int) -> Int {
    i + distance
  }

  /// Returns an index that is the specified distance from the given index,
  /// unless that distance is beyond a given limiting index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection, unless the index passed as `limit` prevents offsetting
  /// beyond those bounds. (Otherwise the returned value won't be a valid index
  /// in the collection.)
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`.
  ///   - limit: A valid index of the collection to use as a limit. If
  ///     `distance > 0`, `limit` has no effect if it is less than `i`.
  ///     Likewise, if `distance < 0`, `limit` has no effect if it is greater
  ///     than `i`.
  /// - Returns: An index offset by `distance` from the index `i`, unless that
  ///   index would be beyond `limit` in the direction of movement. In that
  ///   case, the method returns `nil`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func index(
    _ i: Int,
    offsetBy distance: Int,
    limitedBy limit: Int
  ) -> Int? {
    _base._values.index(i, offsetBy: distance, limitedBy: limit)
  }

  /// Returns the distance between two indices.
  ///
  /// - Parameters:
  ///   - start: A valid index of the collection.
  ///   - end: Another valid index of the collection. If `end` is equal to
  ///     `start`, the result is zero.
  ///
  /// - Returns: The distance between `start` and `end`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func distance(from start: Int, to end: Int) -> Int {
    end - start
  }

  /// Accesses the element at the specified position.
  ///
  /// - Parameter index: The position of the element to access. `index` must be
  ///   greater than or equal to `startIndex` and less than `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public subscript(position: Int) -> Element {
    (_base._keys[position], _base._values[position])
  }

  /// Accesses a contiguous subrange of the dictionary's elements.
  ///
  /// The returned `Subsequence` instance uses the same indices for the same
  /// elements as the original collection. In particular, the slice, unlike an
  /// `Elements`, may have a nonzero `startIndex` and an `endIndex` that is not
  /// equal to `count`. Always use the slice's `startIndex` and `endIndex`
  /// properties instead of assuming that its indices start or end at a
  /// particular value.
  ///
  /// - Parameter bounds: A range of valid indices in the collection.
  ///
  /// - Complexity: O(1)
  public subscript(bounds: Range<Int>) -> SubSequence {
    _failEarlyRangeCheck(bounds, bounds: startIndex ..< endIndex)
    return SubSequence(_base: _base, bounds: bounds)
  }

  /// A Boolean value indicating whether the collection is empty.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var isEmpty: Bool { _base.isEmpty }

  /// The number of elements in the dictionary.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var count: Int { _base.count }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ index: Int, bounds: Range<Int>) {
    _base._values._failEarlyRangeCheck(index, bounds: bounds)
  }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ index: Int, bounds: ClosedRange<Int>) {
    _base._values._failEarlyRangeCheck(index, bounds: bounds)
  }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ range: Range<Int>, bounds: Range<Int>) {
    _base._values._failEarlyRangeCheck(range, bounds: bounds)
  }
}

extension OrderedDictionary.Elements: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _base.description
  }
}

extension OrderedDictionary.Elements: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension OrderedDictionary.Elements: CustomReflectable {
  public var customMirror: Mirror {
    Mirror(self, unlabeledChildren: self, displayStyle: .collection)
  }
}

extension OrderedDictionary.Elements: Equatable where Value: Equatable {
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._base == right._base
  }
}

extension OrderedDictionary.Elements: Hashable where Value: Hashable {
  @inlinable
  public func hash(into hasher: inout Hasher) {
    _base.hash(into: &hasher)
  }
}

// MARK: Partial `MutableCollection`

extension OrderedDictionary.Elements {
  /// Exchanges the key-value pairs at the specified indices of the dictionary.
  ///
  /// Both parameters must be valid indices below `endIndex`. Passing the same
  /// index as both `i` and `j` has no effect.
  ///
  /// - Parameters:
  ///   - i: The index of the first value to swap.
  ///   - j: The index of the second value to swap.
  ///
  /// - Complexity: O(1) when the dictionary's storage isn't shared with another
  ///    value; O(`count`) otherwise.
  @inlinable
  @inline(__always)
  public mutating func swapAt(_ i: Int, _ j: Int) {
    _base.swapAt(i, j)
  }

  /// Reorders the elements of the dictionary such that all the elements that
  /// match the given predicate are after all the elements that don't match.
  ///
  /// After partitioning a collection, there is a pivot index `p` where
  /// no element before `p` satisfies the `belongsInSecondPartition`
  /// predicate and every element at or after `p` satisfies
  /// `belongsInSecondPartition`.
  ///
  /// - Parameter belongsInSecondPartition: A predicate used to partition
  ///   the collection. All elements satisfying this predicate are ordered
  ///   after all elements not satisfying it.
  /// - Returns: The index of the first element in the reordered collection
  ///   that matches `belongsInSecondPartition`. If no elements in the
  ///   collection match `belongsInSecondPartition`, the returned index is
  ///   equal to the collection's `endIndex`.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  @inline(__always)
  public mutating func partition(
    by belongsInSecondPartition: (Element) throws -> Bool
  ) rethrows -> Int {
    try _base.partition(by: belongsInSecondPartition)
  }
}

extension OrderedDictionary.Elements {
  /// Sorts the collection in place, using the given predicate as the
  /// comparison between elements.
  ///
  /// When you want to sort a collection of elements that don't conform to
  /// the `Comparable` protocol, pass a closure to this method that returns
  /// `true` when the first element should be ordered before the second.
  ///
  /// Alternatively, use this method to sort a collection of elements that do
  /// conform to `Comparable` when you want the sort to be descending instead
  /// of ascending. Pass the greater-than operator (`>`) operator as the
  /// predicate.
  ///
  /// `areInIncreasingOrder` must be a *strict weak ordering* over the
  /// elements. That is, for any elements `a`, `b`, and `c`, the following
  /// conditions must hold:
  ///
  /// - `areInIncreasingOrder(a, a)` is always `false`. (Irreflexivity)
  /// - If `areInIncreasingOrder(a, b)` and `areInIncreasingOrder(b, c)` are
  ///   both `true`, then `areInIncreasingOrder(a, c)` is also `true`.
  ///   (Transitive comparability)
  /// - Two elements are *incomparable* if neither is ordered before the other
  ///   according to the predicate. If `a` and `b` are incomparable, and `b`
  ///   and `c` are incomparable, then `a` and `c` are also incomparable.
  ///   (Transitive incomparability)
  ///
  /// The sorting algorithm is guaranteed to be stable. A stable sort
  /// preserves the relative order of elements for which
  /// `areInIncreasingOrder` does not establish an order.
  ///
  /// - Parameter areInIncreasingOrder: A predicate that returns `true` if its
  ///   first argument should be ordered before its second argument;
  ///   otherwise, `false`. If `areInIncreasingOrder` throws an error during
  ///   the sort, the elements may be in a different order, but none will be
  ///   lost.
  ///
  /// - Complexity: O(*n* log *n*), where *n* is the length of the collection.
  @inlinable
  @inline(__always)
  public mutating func sort(
    by areInIncreasingOrder: (Element, Element) throws -> Bool
  ) rethrows {
    try _base.sort(by: areInIncreasingOrder)
  }
}

extension OrderedDictionary.Elements where Key: Comparable {
  /// Sorts the dictionary in place.
  ///
  /// You can sort an ordered dictionary of keys that conform to the
  /// `Comparable` protocol by calling this method. The key-value pairs are
  /// sorted in ascending order. (`Value` doesn't need to conform to
  /// `Comparable` because the keys are guaranteed to be unique.)
  ///
  /// The sorting algorithm is guaranteed to be stable. A stable sort
  /// preserves the relative order of elements that compare as equal.
  ///
  /// - Complexity: O(*n* log *n*), where *n* is the length of the collection.
  @inlinable
  @inline(__always)
  public mutating func sort() {
    _base.sort()
  }
}

extension OrderedDictionary.Elements {
  /// Shuffles the collection in place.
  ///
  /// Use the `shuffle()` method to randomly reorder the elements of an ordered
  /// dictionary.
  ///
  /// This method is equivalent to calling `shuffle(using:)`, passing in the
  /// system's default random generator.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  @inlinable
  public mutating func shuffle() {
    _base.shuffle()
  }

  /// Shuffles the collection in place, using the given generator as a source
  /// for randomness.
  ///
  /// You use this method to randomize the elements of a collection when you
  /// are using a custom random number generator. For example, you can use the
  /// `shuffle(using:)` method to randomly reorder the elements of an array.
  ///
  /// - Parameter generator: The random number generator to use when shuffling
  ///   the collection.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  ///
  /// - Note: The algorithm used to shuffle a collection may change in a future
  ///   version of Swift. If you're passing a generator that results in the
  ///   same shuffled order each time you run your program, that sequence may
  ///   change when your program is compiled using a different version of
  ///   Swift.
  @inlinable
  public mutating func shuffle(
    using generator: inout some RandomNumberGenerator
  ) {
    _base.shuffle(using: &generator)
  }
}

extension OrderedDictionary.Elements {
  /// Reverses the elements of the ordered dictionary in place.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func reverse() {
    _base.reverse()
  }
}

// MARK: Partial `RangeReplaceableCollection`

extension OrderedDictionary.Elements {
  /// Removes all members from the dictionary.
  ///
  /// - Parameter keepingCapacity: If `true`, the dictionary's storage capacity
  ///   is preserved; if `false`, the underlying storage is released. The
  ///   default is `false`.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    _base.removeAll(keepingCapacity: keepCapacity)
  }

  /// Removes and returns the element at the specified position.
  ///
  /// All the elements following the specified position are moved to close the
  /// resulting gap.
  ///
  /// - Parameter index: The position of the element to remove. `index` must be
  ///    a valid index of the collection that is not equal to the collection's
  ///    end index.
  ///
  /// - Returns: The removed element.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  @discardableResult
  public mutating func remove(at index: Int) -> Element {
    _base.remove(at: index)
  }

  /// Removes the specified subrange of elements from the collection.
  ///
  /// All the elements following the specified subrange are moved to close the
  /// resulting gap.
  ///
  /// - Parameter bounds: The subrange of the collection to remove. The bounds
  ///   of the range must be valid indices of the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeSubrange(_ bounds: Range<Int>) {
    _base.removeSubrange(bounds)
  }

  /// Removes the specified subrange of elements from the collection.
  ///
  /// All the elements following the specified subrange are moved to close the
  /// resulting gap.
  ///
  /// - Parameter bounds: The subrange of the collection to remove. The bounds
  ///   of the range must be valid indices of the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeSubrange(
    _ bounds: some RangeExpression<Int>
  ) {
    _base.removeSubrange(bounds)
  }


  /// Removes the last element of a non-empty dictionary.
  ///
  /// - Complexity: Expected to be O(`1`) on average, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  @discardableResult
  public mutating func removeLast() -> Element {
    _base.removeLast()
  }

  /// Removes the last `n` element of the dictionary.
  ///
  /// - Parameter n: The number of elements to remove from the collection.
  ///   `n` must be greater than or equal to zero and must not exceed the
  ///   number of elements in the collection.
  ///
  /// - Complexity: Expected to be O(`n`) on average, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public mutating func removeLast(_ n: Int) {
    _base.removeLast(n)
  }

  /// Removes the first element of a non-empty dictionary.
  ///
  /// The members following the removed key-value pair need to be moved to close
  /// the resulting gaps in the storage arrays.
  ///
  /// - Complexity: O(`count`).
  @inlinable
  @discardableResult
  public mutating func removeFirst() -> Element {
    _base.removeFirst()
  }

  /// Removes the first `n` elements of the dictionary.
  ///
  /// The members following the removed items need to be moved to close the
  /// resulting gaps in the storage arrays.
  ///
  /// - Parameter n: The number of elements to remove from the collection.
  ///   `n` must be greater than or equal to zero and must not exceed the
  ///   number of elements in the set.
  ///
  /// - Complexity: O(`count`).
  @inlinable
  public mutating func removeFirst(_ n: Int) {
    _base.removeFirst(n)
  }

  /// Removes all the elements that satisfy the given predicate.
  ///
  /// Use this method to remove every element in a collection that meets
  /// particular criteria. The order of the remaining elements is preserved.
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   dictionary as its argument and returns a Boolean value indicating
  ///   whether the element should be removed from the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(
    where shouldBeRemoved: (Self.Element) throws -> Bool
  ) rethrows {
    try _base.removeAll(where: shouldBeRemoved)
  }
}

