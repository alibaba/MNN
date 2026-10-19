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

extension TreeSet: Sequence {
  /// An iterator over the members of a `TreeSet`.
  @frozen
  public struct Iterator: IteratorProtocol {
    @usableFromInline
    internal typealias _UnsafeHandle = _Node.UnsafeHandle

    @usableFromInline
    internal var _it: _HashTreeIterator

    @inlinable
    internal init(_root: _RawHashNode) {
      _it = _HashTreeIterator(root: _root)
    }

    /// Advances to the next element and returns it, or `nil` if no next element
    /// exists.
    ///
    /// Once `nil` has been returned, all subsequent calls return `nil`.
    ///
    /// - Complexity: O(1)
    @inlinable
    public mutating func next() -> Element? {
      guard let (node, slot) = _it.next() else { return nil }
      return _UnsafeHandle.read(node) { $0[item: slot].key }
    }
  }

  /// Returns an iterator over the members of the set.
  @inlinable
  public func makeIterator() -> Iterator {
    Iterator(_root: _root.raw)
  }

  @inlinable
  public func _customContainsEquatableElement(_ element: Element) -> Bool? {
    _root.containsKey(.top, element, _Hash(element))
  }
}

extension TreeSet.Iterator: @unchecked Sendable
where Element: Sendable {}
