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

extension OrderedDictionary: Sequence {
  /// The element type of a dictionary: a tuple containing an individual
  /// key-value pair.
  public typealias Element = (key: Key, value: Value)

  /// The type that allows iteration over an ordered dictionary's elements.
  @frozen
  public struct Iterator: IteratorProtocol {
    @usableFromInline
    internal let _base: OrderedDictionary

    @usableFromInline
    internal var _position: Int

    @inlinable
    @inline(__always)
    internal init(_base: OrderedDictionary) {
      self._base = _base
      self._position = 0
    }

    /// Advances to the next element and returns it, or nil if no next
    /// element exists.
    ///
    /// - Complexity: O(1)
    @inlinable
    public mutating func next() -> Element? {
      guard _position < _base._values.count else { return nil }
      let result = (_base._keys[_position], _base._values[_position])
      _position += 1
      return result
    }
  }

  /// The number of elements in the collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public var underestimatedCount: Int {
    count
  }

  /// Returns an iterator over the elements of this collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public func makeIterator() -> Iterator {
    Iterator(_base: self)
  }
}

extension OrderedDictionary.Iterator: Sendable
where Key: Sendable, Value: Sendable {}
