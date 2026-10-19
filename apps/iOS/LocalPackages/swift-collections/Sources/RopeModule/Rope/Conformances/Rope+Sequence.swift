//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Rope: Sequence {
  @inlinable
  public func makeIterator() -> Iterator {
    Iterator(self, from: self.startIndex)
  }

  @inlinable
  public func makeIterator(from start: Index) -> Iterator {
    Iterator(self, from: start)
  }

  @frozen // Not really! This module isn't ABI stable.
  public struct Iterator: IteratorProtocol {
    @usableFromInline
    internal let _rope: Rope

    @usableFromInline
    internal var _index: Index

    @inlinable
    internal init(_ rope: Rope, from start: Index) {
      rope.validate(start)
      self._rope = rope
      self._index = start
      self._rope.grease(&_index)
    }

    @inlinable
    public mutating func next() -> Element? {
      guard let leaf = _index._leaf else { return nil }
      let item = leaf.read { $0.children[_index._path[0]].value }
      _rope.formIndex(after: &_index)
      return item
    }
  }
}
