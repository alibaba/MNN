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

extension BitArray: Sequence {
  /// The Boolean type representing the bit array's elements.
  public typealias Element = Bool
  /// The iterator type for a bit array.
  public typealias Iterator = IndexingIterator<BitArray>
}

extension BitArray: RandomAccessCollection, MutableCollection {
  /// The type representing a position in a bit array.
  public typealias Index = Int
  
  /// A collection representing a contiguous subrange of this collection's
  /// elements. The subsequence shares indices with the original collection.
  ///
  /// The subsequence type for bit arrays is the default `Slice`.
  public typealias SubSequence = Slice<BitArray>

  /// A type that represents the indices that are valid for subscripting the
  /// collection, in ascending order.
  public typealias Indices = Range<Int>

  /// The number of elements in the bit array.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var count: Int {
    Int(_count)
  }

  /// The position of the first element in a nonempty bit array, or `endIndex`
  /// if the array is empty.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public var startIndex: Int { 0 }

  /// The collection’s “past the end” position--that is, the position one step
  /// after the last valid subscript argument.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public var endIndex: Int { count }

  /// Returns the position immediately after the given index.
  ///
  /// - Parameter `index`: A valid index of the bit set. `index` must be less than `endIndex`.
  ///
  /// - Returns: The valid index immediately after `index`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func index(after i: Int) -> Int { i + 1 }

  /// Returns the position immediately before the given index.
  ///
  /// - Parameter `index`: A valid index of the bit set. `index` must be greater
  ///    than `startIndex`.
  ///
  /// - Returns: The valid index immediately before `index`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func index(before i: Int) -> Int { i - 1 }

  /// Replaces the given index with its successor.
  ///
  /// - Parameter i: A valid index of the collection. `i` must be less than
  ///   `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func formIndex(after i: inout Int) {
    i += 1
  }

  /// Replaces the given index with its predecessor.
  ///
  /// - Parameter i: A valid index of the collection. `i` must be greater than
  ///   `startIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func formIndex(before i: inout Int) {
    i -= 1
  }

  /// Returns an index that is the specified distance from the given index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection.
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`.
  /// - Returns: An index offset by `distance` from the index `i`. If
  ///   `distance` is positive, this is the same value as the result of
  ///   `distance` calls to `index(after:)`. If `distance` is negative, this
  ///   is the same value as the result of `abs(distance)` calls to
  ///   `index(before:)`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func index(_ i: Int, offsetBy distance: Int) -> Int {
    i + distance
  }
  
  /// Returns an index that is the specified distance from the given index,
  /// unless that distance is beyond a given limiting index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection, unless the index passed as `limit` prevents offsetting
  /// beyond those bounds.
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`.
  ///   - limit: A valid index of the collection to use as a limit. If
  ///     `distance > 0`, a limit that is less than `i` has no effect.
  ///     Likewise, if `distance < 0`, a limit that is greater than `i` has no
  ///     effect.
  /// - Returns: An index offset by `distance` from the index `i`, unless that
  ///   index would be beyond `limit` in the direction of movement. In that
  ///   case, the method returns `nil`.
  ///
  /// - Complexity: O(1)
  @inlinable
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    let l = self.distance(from: i, to: limit)
    if distance > 0 ? l >= 0 && l < distance : l <= 0 && distance < l {
      return nil
    }
    return index(i, offsetBy: distance)
  }

  /// Returns the distance between two indices.
  ///
  /// - Parameters:
  ///   - start: A valid index of the collection.
  ///   - end: Another valid index of the collection. If `end` is equal to
  ///     `start`, the result is zero.
  /// - Returns: The distance between `start` and `end`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func distance(from start: Int, to end: Int) -> Int {
    end - start
  }

  /// Accesses the element at the specified position.
  ///
  /// You can subscript a collection with any valid index other than the
  /// collection's end index. The end index refers to the position one past
  /// the last element of a collection, so it doesn't correspond with an
  /// element.
  ///
  /// - Parameter position: The position of the element to access. `position`
  ///   must be a valid index of the collection that is not equal to the
  ///   `endIndex` property.
  ///
  /// - Complexity: O(1)
  public subscript(position: Int) -> Bool {
    get {
      precondition(position >= 0 && position < _count, "Index out of bounds")
      return _read { handle in
        handle[position]
      }
    }
    set {
      precondition(position >= 0 && position < _count, "Index out of bounds")
      return _update { handle in
        handle[position] = newValue
      }
    }
  }
}
