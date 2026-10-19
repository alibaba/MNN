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
  /// A view of a dictionary’s values.
  @frozen
  public struct Values {
    @usableFromInline
    internal typealias _Node = TreeDictionary._Node

    @usableFromInline
    internal typealias _UnsafeHandle = _Node.UnsafeHandle

    @usableFromInline
    internal var _base: TreeDictionary

    @inlinable
    internal init(_base: TreeDictionary) {
      self._base = _base
    }
  }

  /// A collection containing just the values of the dictionary.
  @inlinable
  public var values: Values {
    // Note: this property is kept read only for now until we decide whether
    // it's worth providing setters without a `MutableCollection` conformance.
    get {
      Values(_base: self)
    }
  }
}

extension TreeDictionary.Values: Sendable
where Key: Sendable, Value: Sendable {}

extension TreeDictionary.Values: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _arrayDescription(for: self)
  }
}

extension TreeDictionary.Values: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension TreeDictionary.Values: Sequence {
  public typealias Element = Value

  @frozen
  public struct Iterator: IteratorProtocol {
    public typealias Element = Value

    @usableFromInline
    internal var _base: TreeDictionary.Iterator

    @inlinable
    internal init(_base: TreeDictionary.Iterator) {
      self._base = _base
    }

    @inlinable
    public mutating func next() -> Element? {
      _base.next()?.value
    }
  }

  @inlinable
  public func makeIterator() -> Iterator {
    Iterator(_base: _base.makeIterator())
  }
}

extension TreeDictionary.Values.Iterator: Sendable
where Key: Sendable, Value: Sendable {}

// Note: This cannot be a MutableCollection because its subscript setter
// needs to invalidate indices.
extension TreeDictionary.Values: Collection {
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
    // The subscript is kept read only for now until we decide whether it's
    // worth providing setters without a `MutableCollection` conformance.
    // (With the current index implementation, mutating values must invalidate
    // indices.)
    get {
      _base[index].value
    }
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
extension TreeDictionary.Values: BidirectionalCollection {
  // Note: Let's not do this. `BidirectionalCollection` would imply that
  // the ordering of elements would be meaningful, which isn't true for
  // `TreeDictionary.Values`.
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
