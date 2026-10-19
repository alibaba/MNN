//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2019 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension TreeDictionary: Sequence {
  /// The element type of a dictionary: a tuple containing an individual
  /// key-value pair.
  public typealias Element = (key: Key, value: Value)

  /// The type that allows iteration over a persistent dictionary's elements.
  @frozen
  public struct Iterator {
    // Fixed-stack iterator for traversing a hash tree.
    // The iterator performs a pre-order traversal, with items at a node visited
    // before any items within children.

    @usableFromInline
    internal typealias _UnsafeHandle = _Node.UnsafeHandle

    @usableFromInline
    internal var _it: _HashTreeIterator

    @inlinable
    internal init(_root: _RawHashNode) {
      self._it = _HashTreeIterator(root: _root)
    }
  }

  /// A value less than or equal to the number of elements in the sequence,
  /// calculated nondestructively.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var underestimatedCount: Int {
    _root.count
  }

  /// Returns an iterator over the elements of this collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  public __consuming func makeIterator() -> Iterator {
    return Iterator(_root: _root.raw)
  }
}

extension TreeDictionary.Iterator: @unchecked Sendable
where Key: Sendable, Value: Sendable {}

extension TreeDictionary.Iterator: IteratorProtocol {
  /// The element type of a dictionary: a tuple containing an individual
  /// key-value pair.
  public typealias Element = (key: Key, value: Value)

  /// Advances to the next element and returns it, or `nil` if no next
  /// element exists.
  ///
  /// Once `nil` has been returned, all subsequent calls return `nil`.
  ///
  /// - Complexity: O(1)
  @inlinable
  public mutating func next() -> Element? {
    guard let (node, slot) = _it.next() else { return nil }
    return _UnsafeHandle.read(node) { $0[item: slot] }
  }
}
