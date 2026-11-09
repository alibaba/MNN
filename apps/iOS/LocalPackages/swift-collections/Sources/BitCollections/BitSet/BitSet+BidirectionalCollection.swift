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

extension BitSet: Sequence {
  /// The type representing the bit set's elements.
  /// Bit sets are collections of nonnegative integers.
  public typealias Element = Int

  /// Returns the exact count of the bit set.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var underestimatedCount: Int {
    return count
  }

  /// Returns an iterator over the elements of the bit set.
  ///
  /// - Complexity: O(1)
  @inlinable
  public func makeIterator() -> Iterator {
    return Iterator(self)
  }

  public func _customContainsEquatableElement(
    _ element: Int
  ) -> Bool? {
    guard let element = UInt(exactly: element) else { return false }
    return _contains(element)
  }

  /// An iterator over the members of a bit set.
  public struct Iterator: IteratorProtocol {
    internal typealias _UnsafeHandle = BitSet._UnsafeHandle

    internal let bitset: BitSet
    internal var index: Int
    internal var word: _Word

    @usableFromInline
    internal init(_ bitset: BitSet) {
      self.bitset = bitset
      self.index = 0
      self.word = bitset._read { handle in
        guard handle.wordCount > 0 else { return .empty }
        return handle._words[0]
      }
    }

    /// Advances to the next element and returns it, or `nil` if no next element
    /// exists.
    ///
    /// Once `nil` has been returned, all subsequent calls return `nil`.
    ///
    /// - Complexity:
    ///   Each individual call has a worst case time complexity of O(*n*),
    ///   where *n* is largest element in the set, as each call needs to
    ///   search for the next `true` bit in the underlying storage.
    ///   However, each storage bit is only visited once, so iterating over the
    ///   entire set has the same O(*n*) complexity.
    @_effects(releasenone)
    public mutating func next() -> Int? {
      if let bit = word.next() {
        let i = _UnsafeHandle.Index(word: index, bit: bit)
        return Int(truncatingIfNeeded: i.value)
      }
      return bitset._read { handle in
        while (index + 1) < handle.wordCount {
          index += 1
          word = handle._words[index]
          if let bit = word.next() {
            let i = _UnsafeHandle.Index(word: index, bit: bit)
            return Int(truncatingIfNeeded: i.value)
          }
        }
        return nil
      }
    }
  }
}

extension BitSet.Iterator: Sendable {}

extension BitSet: Collection, BidirectionalCollection {
  /// A Boolean value indicating whether the collection is empty.
  ///
  /// - Complexity: O(*min*) where *min* is the value of the first element.
  ///    (The complexity is O(1) if the set is empty.)
  public var isEmpty: Bool { _storage.firstIndex { !$0.isEmpty } == nil }

  /// The number of elements in the bit set.
  ///
  /// - Complexity: O(*max*) where *max* is the value of the largest element.
  ///    (The complexity is O(1) if the set is empty.)
  ///
  /// - Note: `BitSet.Counted` is a variant of this type that keeps a running
  ///    total of its element count, for use cases that require an O(1) count.
  public var count: Int {
    return _read { $0.count }
  }

  /// The position of the first element in a nonempty set, or `endIndex`
  /// if the collection is empty.
  ///
  /// - Complexity: O(*min*) where *min* is the value of the first element.
  public var startIndex: Index {
    Index(_position: _read { $0.startIndex })
  }

  /// The collection’s “past the end” position--that is, the position one step
  /// after the last valid subscript argument.
  ///
  /// - Complexity: O(1)
  public var endIndex: Index {
    Index(_position: .init(word: _storage.count, bit: 0))
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
  public subscript(position: Index) -> Int {
    let v = Int(bitPattern: position._value)
    assert(contains(v))
    return v
  }
  
  /// Returns the position immediately after the given index.
  ///
  /// - Parameter `index`: A valid index of the bit set. `index` must be less
  ///    than `endIndex`.
  ///
  /// - Returns: The valid index immediately after `index`.
  ///
  /// - Complexity:
  ///   O(*d*), where *d* is difference between the value of the member
  ///   addressed by `index` and the member following it in the set.
  ///   (Each call needs to search for the next `true` bit in the underlying
  ///   storage.)
  public func index(after index: Index) -> Index {
    _read { handle in
      assert(handle._isReachable(index._position), "Invalid index")
      let pos = handle.index(after: index._position)
      return Index(_position: pos)
    }
  }
  
  /// Returns the position immediately before the given index.
  ///
  /// - Parameter `index`: A valid index of the bit set.
  ///    `index` must be greater than `startIndex`.
  ///
  /// - Returns: The preceding valid index immediately before `index`.
  ///
  /// - Complexity:
  ///   O(*d*), where *d* is difference between the value of the member
  ///   addressed by `index` and the member preceding it in the set.
  ///   (Each call needs to search for the next `true` bit in the underlying
  ///   storage.)
  public func index(before index: Index) -> Index {
    _read { handle in
      assert(handle._isReachable(index._position), "Invalid index")
      let pos = handle.index(before: index._position)
      return Index(_position: pos)
    }
  }

  /// Returns the distance between two indices.
  ///
  /// - Parameters:
  ///   - start: A valid index of the collection.
  ///   - end: Another valid index of the collection. If `end` is equal to
  ///     `start`, the result is zero.
  /// - Returns: The distance between `start` and `end`.
  ///
  /// - Complexity: O(*d*), where *d* is the difference of the values
  ///    addressed by the two input indices.
  public func distance(from start: Index, to end: Index) -> Int {
    _read { handle in
      assert(handle._isReachable(start._position), "Invalid start index")
      assert(handle._isReachable(end._position), "Invalid end index")
      return handle.distance(from: start._position, to: end._position)
    }
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
  /// - Complexity: O(*d*), where *d* is the difference of the values
  ///    addressed by `index` and the returned result.
  public func index(_ index: Index, offsetBy distance: Int) -> Index {
    _read { handle in
      assert(handle._isReachable(index._position), "Invalid index")
      let pos = handle.index(index._position, offsetBy: distance)
      return Index(_position: pos)
    }
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
  /// - Complexity: O(*d*), where *d* is the difference of the values
  ///    addressed by `index` and the returned result.
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    _read { handle in
      assert(handle._isReachable(i._position), "Invalid index")
      assert(handle._isReachable(limit._position), "Invalid limit index")
      return handle.index(
        i._position, offsetBy: distance, limitedBy: limit._position
      ).map { Index(_position: $0) }
    }
  }

  public func _customIndexOfEquatableElement(_ element: Int) -> Index?? {
    guard contains(element) else { return .some(nil) }
    return Index(_value: UInt(bitPattern: element))
  }

  public func _customLastIndexOfEquatableElement(_ element: Int) -> Index?? {
    _customIndexOfEquatableElement(element)
  }
}
