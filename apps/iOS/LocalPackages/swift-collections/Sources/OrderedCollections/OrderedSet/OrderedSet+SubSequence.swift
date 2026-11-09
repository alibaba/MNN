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

extension OrderedSet {
  /// A collection that represents a contiguous slice of an ordered set.
  ///
  /// Ordered set slices are random access collections that support efficient
  /// membership testing. `contains(_:)` and `firstIndex(of:)`/`lastIndex(of:)`
  /// are expected to have a complexity of O(1), as long as `Element` has
  /// high-quality hashing.
  @frozen
  public struct SubSequence {
    @usableFromInline
    internal var _base: OrderedSet

    @usableFromInline
    internal var _bounds: Range<Int>

    @inlinable
    @inline(__always)
    internal init(_base: OrderedSet, bounds: Range<Int>) {
      self._base = _base
      self._bounds = bounds
    }
  }
}

extension OrderedSet.SubSequence: Sendable where Element: Sendable {}

extension OrderedSet.SubSequence {
  @inlinable
  internal var _slice: Array<Element>.SubSequence {
    _base._elements[_bounds]
  }

  @inlinable
  internal func _index(of element: Element) -> Int? {
    guard let index = _base._find(element).index else { return nil }
    guard _bounds.contains(index) else { return nil }
    return index
  }
}

extension OrderedSet.SubSequence: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: self)
  }
}

extension OrderedSet.SubSequence: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension OrderedSet.SubSequence: Sequence {
  // A type representing the collection’s elements.
  public typealias Element = OrderedSet.Element
  /// The type that allows iteration over the collection's elements.
  public typealias Iterator = IndexingIterator<Self>

  @inlinable
  public func _customContainsEquatableElement(_ element: Element) -> Bool? {
    _index(of: element) != nil
  }

  @inlinable
  public __consuming func _copyToContiguousArray() -> ContiguousArray<Element> {
    _slice._copyToContiguousArray()
  }

  @inlinable
  public __consuming func _copyContents(
    initializing ptr: UnsafeMutableBufferPointer<Element>
  ) -> (Iterator, UnsafeMutableBufferPointer<Element>.Index) {
    guard !isEmpty else { return (makeIterator(), 0) }
    let copied: Int = _slice.withUnsafeBufferPointer { buffer in
      guard let p = ptr.baseAddress else {
        preconditionFailure("Attempt to copy contents into nil buffer pointer")
      }
      let c = Swift.min(buffer.count, ptr.count)
      if c > 0 {
        p.initialize(from: buffer.baseAddress!, count: c)
      }
      return c
    }
    return (Iterator(_elements: self, _position: _bounds.lowerBound + copied),
            copied)
  }

  /// Call `body(p)`, where `p` is a buffer pointer to the collection’s
  /// contiguous storage. Ordered sets always have contiguous storage.
  ///
  /// - Parameter body: A function to call. The function must not escape its
  ///    unsafe buffer pointer argument.
  ///
  /// - Returns: The value returned by `body`.
  ///
  /// - Complexity: O(1) (ignoring time spent in `body`)
  @inlinable
  public func withContiguousStorageIfAvailable<R>(
    _ body: (UnsafeBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    try _slice.withContiguousStorageIfAvailable(body)
  }
}

extension OrderedSet.SubSequence: _UniqueCollection {}

extension OrderedSet.SubSequence: RandomAccessCollection {
  /// The index type for ordered sets, `Int`.
  ///
  /// Indices in the order set are integer offsets from the start of the
  /// collection, starting at zero for the first element (if exists).
  public typealias Index = Int

  /// The type that represents the indices that are valid for subscripting an
  /// ordered set, in ascending order.
  public typealias Indices = Array<Element>.SubSequence.Indices

  /// Ordered set subsequences are self-slicing.
  public typealias SubSequence = Self

  /// The position of the first element in a nonempty ordered set slice.
  ///
  /// Note that instances of `OrderedSet.SubSequence` generally aren't indexed
  /// from zero.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var startIndex: Int { _bounds.lowerBound }

  /// The "past the end" position---that is, the position one greater
  /// than the last valid subscript argument.
  ///
  /// Note that instances of `OrderedSet.SubSequence` generally aren't indexed
  /// from zero, so `endIndex` may differ from the `count`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var endIndex: Int { _bounds.upperBound }

  /// The indices that are valid for subscripting the collection, in ascending
  /// order.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var indices: Indices { _slice.indices }

  /// Returns the position immediately after the given index.
  ///
  /// The specified index must be a valid index less than `endIndex`, or the
  /// returned value won't be a valid index in the set.
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
  /// the returned value won't be a valid index in the set.
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
  /// returned value won't be a valid index in the set.
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
  /// the returned value won't be a valid index in the set.
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
  ///   - i: A valid index of the set.
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
  /// in the set.)
  ///
  /// - Parameters:
  ///   - i: A valid index of the set.
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
    _slice.index(i, offsetBy: distance, limitedBy: limit)
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
    _slice[position]
  }

  /// Accesses a contiguous subrange of the set's elements.
  ///
  /// The returned `Subsequence` instance uses the same indices for the same
  /// elements as the original set. In particular, that slice, unlike an
  /// `OrderedSet`, may have a nonzero `startIndex` and an `endIndex` that is
  /// not equal to `count`. Always use the slice's `startIndex` and `endIndex`
  /// properties instead of assuming that its indices start or end at a
  /// particular value.
  ///
  /// - Parameter bounds: A range of valid indices in the set.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public subscript(bounds: Range<Int>) -> SubSequence {
    _failEarlyRangeCheck(bounds, bounds: startIndex ..< endIndex)
    return SubSequence(_base: _base, bounds: bounds)
  }

  /// A Boolean value indicating whether the collection is empty.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var isEmpty: Bool { _bounds.isEmpty }

  /// The number of elements in the set.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var count: Int { _bounds.count }

  @inlinable
  @inline(__always)
  public func _customIndexOfEquatableElement(_ element: Element) -> Int?? {
    .some(_index(of: element))
  }

  @inlinable
  @inline(__always)
  public func _customLastIndexOfEquatableElement(_ element: Element) -> Int?? {
    .some(_index(of: element))
  }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ index: Int, bounds: Range<Int>) {
    _slice._failEarlyRangeCheck(index, bounds: bounds)
  }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ index: Int, bounds: ClosedRange<Int>) {
    _slice._failEarlyRangeCheck(index, bounds: bounds)
  }

  @inlinable
  @inline(__always)
  public func _failEarlyRangeCheck(_ range: Range<Int>, bounds: Range<Int>) {
    _slice._failEarlyRangeCheck(range, bounds: bounds)
  }
}

extension OrderedSet.SubSequence: Equatable {
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._base._elements[left._bounds] == right._base._elements[right._bounds]
  }
}

extension OrderedSet.SubSequence: Hashable {
  @inlinable
  public func hash(into hasher: inout Hasher) {
    hasher.combine(count)
    for item in self {
      hasher.combine(item)
    }
  }
}
