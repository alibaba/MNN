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
  /// A view of an ordered dictionary's values as a standalone collection.
  @frozen
  public struct Values {
    @usableFromInline
    internal var _base: OrderedDictionary

    @inlinable
    @inline(__always)
    internal init(_base: OrderedDictionary) {
      self._base = _base
    }
  }
}

extension OrderedDictionary.Values: Sendable
where Key: Sendable, Value: Sendable {}

extension OrderedDictionary.Values: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: self)
  }
}

extension OrderedDictionary.Values: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension OrderedDictionary.Values {
  /// A read-only view of the contents of this collection as an array value.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var elements: Array<Value> {
    Array(_base._values)
  }
}

extension OrderedDictionary.Values {
  /// Calls a closure with a pointer to the collection's contiguous storage.
  ///
  /// Often, the optimizer can eliminate bounds checks within a collection
  /// algorithm, but when that fails, invoking the same algorithm on the
  /// buffer pointer passed into your closure lets you trade safety for speed.
  ///
  /// The pointer passed as an argument to `body` is valid only during the
  /// execution of `withUnsafeBufferPointer(_:)`. Do not store or return the
  /// pointer for later use.
  ///
  /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that
  ///   points to the contiguous storage for the collection. If `body` has a
  ///   return value, that value is also used as the return value for the
  ///   `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only
  ///   for the duration of the method's execution.
  ///
  /// - Returns: The return value, if any, of the `body` closure parameter.
  ///
  /// - Complexity: O(1) (not counting the closure call)
  @inlinable
  @inline(__always)
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Element>) throws -> R
  ) rethrows -> R {
    try _base._values.withUnsafeBufferPointer(body)
  }

  /// Calls the given closure with a pointer to the collection's mutable
  /// contiguous storage.
  ///
  /// Often, the optimizer can eliminate bounds checks within a collection
  /// algorithm, but when that fails, invoking the same algorithm on the buffer
  /// pointer passed into your closure lets you trade safety for speed.
  ///
  /// The pointer passed as an argument to `body` is valid only during the
  /// execution of `withUnsafeMutableBufferPointer(_:)`. Do not store or return
  /// the pointer for later use.
  ///
  /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter
  ///   that points to the contiguous storage for the collection. If `body` has
  ///   a return value, that value is also used as the return value for the
  ///   `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is
  ///   valid only for the duration of the method's execution.
  ///
  /// - Returns: The return value, if any, of the `body` closure parameter.
  ///
  /// - Complexity: O(1) (not counting the closure call)
  @inlinable
  @inline(__always)
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R {
    try _base._values.withUnsafeMutableBufferPointer(body)
  }
}

extension OrderedDictionary.Values: Sequence {
  /// The element type of the collection.
  public typealias Element = Value

  /// The type that allows iteration over the collection's elements.
  public typealias Iterator = IndexingIterator<Self>
}

extension OrderedDictionary.Values: RandomAccessCollection {
  /// The index type for a dictionary's values view, `Int`.
  ///
  /// Indices in `Values` are integer offsets from the start of the collection.
  public typealias Index = Int

  /// The type that represents the indices that are valid for subscripting the
  /// `Values` collection, in ascending order.
  public typealias Indices = Range<Int>

  /// The position of the first element in a nonempty dictionary.
  ///
  /// For an instance of `OrderedDictionary.Values`, `startIndex` is always
  /// zero. If the dictionary is empty, `startIndex` is equal to `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var startIndex: Int { 0 }

  /// The collection's "past the end" position---that is, the position one
  /// greater than the last valid subscript argument.
  ///
  /// In `OrderedDictionary.Values`, `endIndex` always equals the count of
  /// elements. If the dictionary is empty, `endIndex` is equal to `startIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var endIndex: Int { _base._values.count }

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

  /// Call `body(p)`, where `p` is a buffer pointer to the collection’s
  /// contiguous storage. `OrderedDictionary.Values` values always have
  /// contiguous storage.
  ///
  /// - Parameter body: A function to call. The function must not escape its
  ///    unsafe buffer pointer argument.
  ///
  /// - Returns: The value returned by `body`.
  ///
  /// - Complexity: O(1) (ignoring time spent in `body`)
  @inlinable
  @inline(__always)
  public func withContiguousStorageIfAvailable<R>(
    _ body: (UnsafeBufferPointer<Value>) throws -> R
  ) rethrows -> R? {
    try _base._values.withUnsafeBufferPointer(body)
  }
}

extension OrderedDictionary.Values: MutableCollection {
  /// Accesses the element at the specified position. This can be used to
  /// perform in-place mutations on dictionary values.
  ///
  /// - Parameter index: The position of the element to access. `index` must be
  ///   greater than or equal to `startIndex` and less than `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public subscript(position: Int) -> Value {
    get {
      _base._values[position]
    }
    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      yield &_base._values[position]
    }
  }

  /// Exchanges the values at the specified indices of the collection. (Leaving
  /// their associated keys in the underlying dictionary at their original
  /// position.)
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
    _base._values.swapAt(i, j)
  }

  /// Reorders the elements of the collection such that all the elements that
  /// match the given predicate are after all the elements that don't match.
  ///
  /// This operation does not reorder the keys of the underlying dictionary,
  /// just their associated values.
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
    by belongsInSecondPartition: (Value) throws -> Bool
  ) rethrows -> Int {
    try _base._values.partition(by: belongsInSecondPartition)
  }

  /// Call `body(b)`, where `b` is an unsafe buffer pointer to the collection's
  /// mutable contiguous storage. `OrderedDictionary.Values` always stores its
  /// elements in contiguous storage.
  ///
  /// The supplied buffer pointer is only valid for the duration of the call.
  ///
  /// Often, the optimizer can eliminate bounds- and uniqueness-checks within an
  /// algorithm, but when that fails, invoking the same algorithm on the unsafe
  /// buffer supplied to `body` lets you trade safety for speed.
  ///
  /// - Parameters:
  ///   - body: The function to invoke.
  ///
  /// - Returns: The value returned by `body`, or `nil` if `body` wasn't called.
  ///
  /// - Complexity: O(1) when this instance has a unique reference to its
  ///    underlying storage; O(`count`) otherwise. (Not counting the call to
  ///    `body`.)
  @inlinable
  @inline(__always)
  public mutating func withContiguousMutableStorageIfAvailable<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    try _base._values.withUnsafeMutableBufferPointer(body)
  }
}

extension OrderedDictionary.Values: Equatable where Value: Equatable {
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._base._values == right._base._values
  }
}

extension OrderedDictionary.Values: Hashable where Value: Hashable {
  @inlinable
  public func hash(into hasher: inout Hasher) {
    hasher.combine(count) // Discriminator
    for item in self {
      hasher.combine(item)
    }
  }
}
