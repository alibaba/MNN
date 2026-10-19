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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public struct UnicodeScalarView: Sendable {
    var _base: BigString

    @inline(__always)
    init(_base: BigString) {
      self._base = _base
    }
  }

  @inline(__always)
  public var unicodeScalars: UnicodeScalarView {
    get {
      UnicodeScalarView(_base: self)
    }
    set {
      self = newValue._base
    }
    _modify {
      var view = UnicodeScalarView(_base: self)
      self = .init()
      defer {
        self = view._base
      }
      yield &view
    }
  }

  public init(_ view: BigString.UnicodeScalarView) {
    self = view._base
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self.init(value.unicodeScalars)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: CustomStringConvertible {
  public var description: String {
    String(_base)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: CustomDebugStringConvertible {
  public var debugDescription: String {
    description.debugDescription
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    BigString.utf8IsEqual(left._base, to: right._base)
  }

  public func isIdentical(to other: Self) -> Bool {
    self._base.isIdentical(to: other._base)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: Hashable {
  public func hash(into hasher: inout Hasher) {
    _base.hashUTF8(into: &hasher)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: Sequence {
  public typealias Element = UnicodeScalar

  public struct Iterator {
    internal let _base: BigString
    internal var _index: BigString.Index

    internal init(_base: BigString, from start: BigString.Index) {
      self._base = _base
      self._index = _base._unicodeScalarIndex(roundingDown: start)
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_base: self._base, from: self.startIndex)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView.Iterator: IteratorProtocol {
  public typealias Element = UnicodeScalar

  public mutating func next() -> UnicodeScalar? {
    guard _index < _base.endIndex else { return nil }
    let ri = _index._rope!
    var ci = _index._chunkIndex
    let chunk = _base._rope[ri]
    let result = chunk.string.unicodeScalars[ci]

    chunk.string.unicodeScalars.formIndex(after: &ci)
    if ci < chunk.string.endIndex {
      _index = BigString.Index(baseUTF8Offset: _index._utf8BaseOffset, _rope: ri, chunk: ci)
    } else {
      _index = BigString.Index(
        baseUTF8Offset: _index._utf8BaseOffset + chunk.utf8Count,
        _rope: _base._rope.index(after: ri),
        chunk: String.Index(_utf8Offset: 0))
    }
    return result
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: BidirectionalCollection {
  public typealias Index = BigString.Index
  public typealias SubSequence = BigSubstring.UnicodeScalarView

  @inline(__always)
  public var startIndex: Index { _base.startIndex }

  @inline(__always)
  public var endIndex: Index { _base.endIndex }

  public var count: Int { _base._unicodeScalarCount }

  @inline(__always)
  public func index(after i: Index) -> Index {
    _base._unicodeScalarIndex(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    _base._unicodeScalarIndex(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    _base._unicodeScalarIndex(i, offsetBy: distance)
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    _base._unicodeScalarIndex(i, offsetBy: distance, limitedBy: limit)
  }

  public func distance(from start: Index, to end: Index) -> Int {
    _base._unicodeScalarDistance(from: start, to: end)
  }

  public subscript(position: Index) -> UnicodeScalar {
    _base[_unicodeScalar: position]
  }

  public subscript(bounds: Range<Index>) -> BigSubstring.UnicodeScalarView {
    BigSubstring.UnicodeScalarView(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView {
  public func index(roundingDown i: Index) -> Index {
    _base._unicodeScalarIndex(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    _base._unicodeScalarIndex(roundingUp: i)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UnicodeScalarView: RangeReplaceableCollection {
  public init() {
    self._base = BigString()
  }

  public mutating func reserveCapacity(_ n: Int) {
    // Do nothing.
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Index>, 
    with newElements: __owned some Sequence<UnicodeScalar> // Note: Sequence, not Collection
  ) {
    if let newElements = _specialize(
      newElements, for: String.UnicodeScalarView.self
    ) {
      _base._replaceSubrange(subrange, with: String(newElements))
    } else if let newElements = _specialize(
      newElements, for: Substring.UnicodeScalarView.self
    ) {
      _base._replaceSubrange(subrange, with: Substring(newElements))
    } else if let newElements = _specialize(
      newElements, for: BigString.UnicodeScalarView.self
    ) {
      _base._replaceSubrange(subrange, with: newElements._base)
    } else if let newElements = _specialize(
      newElements, for: BigSubstring.UnicodeScalarView.self
    ) {
      _base._replaceSubrange(
        subrange, with: newElements._base, in: newElements._bounds)
    } else {
      _base._replaceSubrange(subrange, with: BigString(_from: newElements))
    }
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Index>, 
    with newElements: __owned String.UnicodeScalarView
  ) {
    _base._replaceSubrange(subrange, with: String(newElements))
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Index>,
    with newElements: __owned Substring.UnicodeScalarView
  ) {
    _base._replaceSubrange(subrange, with: Substring(newElements))
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Index>, 
    with newElements: __owned BigString.UnicodeScalarView
  ) {
    _base._replaceSubrange(subrange, with: newElements._base)
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Index>, 
    with newElements: __owned BigSubstring.UnicodeScalarView
  ) {
    _base._replaceSubrange(subrange, with: newElements._base, in: newElements._bounds)
  }

  public init(_ elements: some Sequence<UnicodeScalar>) {
    if let elements = _specialize(elements, for: String.UnicodeScalarView.self) {
      self.init(_base: BigString(_from: elements))
    } else if let elements = _specialize(elements, for: Substring.UnicodeScalarView.self) {
      self.init(_base: BigString(_from: elements))
    } else if let elements = _specialize(elements, for: BigString.UnicodeScalarView.self) {
      self.init(_base: elements._base)
    } else if let elements = _specialize(elements, for: BigSubstring.UnicodeScalarView.self) {
      self.init(_base: BigString(_from: elements._base, in: elements._bounds))
    } else {
      self.init(_base: BigString(_from: elements))
    }
  }

  public init(_ elements: String.UnicodeScalarView) {
    self.init(_base: BigString(_from: elements))
  }

  public init(_ elements: Substring.UnicodeScalarView) {
    self.init(_base: BigString(_from: elements))
  }

  public init(_ elements: BigString.UnicodeScalarView) {
    self.init(_base: elements._base)
  }

  public init(_ elements: BigSubstring.UnicodeScalarView) {
    self.init(_base: BigString(_from: elements._base, in: elements._bounds))
  }

  public init(repeating repeatedValue: UnicodeScalar, count: Int) {
    self.init(_base: BigString(repeating: BigString(String(repeatedValue)), count: count))
  }

  public init(repeating repeatedValue: some StringProtocol, count: Int) {
    self.init(_base: BigString(repeating: BigString(_from: repeatedValue), count: count))
  }

  public init(repeating value: BigString.UnicodeScalarView, count: Int) {
    self.init(_base: BigString(repeating: value._base, count: count))
  }

  public init(repeating value: BigSubstring.UnicodeScalarView, count: Int) {
    let value = BigString(value)
    self.init(_base: BigString(repeating: value, count: count))
  }

  public mutating func append(_ newElement: __owned UnicodeScalar) {
    _base.append(contentsOf: String(newElement))
  }

  public mutating func append(contentsOf newElements: __owned some Sequence<UnicodeScalar>) {
    if let newElements = _specialize(newElements, for: String.UnicodeScalarView.self) {
      append(contentsOf: newElements)
    } else if let newElements = _specialize(newElements, for: Substring.UnicodeScalarView.self) {
      append(contentsOf: newElements)
    } else if let newElements = _specialize(newElements, for: BigString.UnicodeScalarView.self) {
      append(contentsOf: newElements)
    } else if let newElements = _specialize(newElements, for: BigSubstring.UnicodeScalarView.self) {
      append(contentsOf: newElements)
    } else {
      _base.append(contentsOf: BigString(_from: newElements))
    }
  }

  public mutating func append(contentsOf newElements: __owned String.UnicodeScalarView) {
    _base.append(contentsOf: String(newElements))
  }

  public mutating func append(contentsOf newElements: __owned Substring.UnicodeScalarView) {
    _base.append(contentsOf: Substring(newElements))
  }

  public mutating func append(contentsOf newElements: __owned BigString.UnicodeScalarView) {
    _base.append(contentsOf: newElements._base)
  }

  public mutating func append(contentsOf newElements: __owned BigSubstring.UnicodeScalarView) {
    _base._append(contentsOf: newElements._base, in: newElements._bounds)
  }

  public mutating func insert(_ newElement: UnicodeScalar, at i: Index) {
    _base.insert(contentsOf: String(newElement), at: i)
  }

  public mutating func insert(
    contentsOf newElements: __owned some Sequence<UnicodeScalar>, // Note: Sequence, not Collection
    at i: Index
  ) {
    if let newElements = _specialize(newElements, for: String.UnicodeScalarView.self) {
      insert(contentsOf: newElements, at: i)
    } else if let newElements = _specialize(newElements, for: Substring.UnicodeScalarView.self) {
      insert(contentsOf: newElements, at: i)
    } else if let newElements = _specialize(newElements, for: BigString.UnicodeScalarView.self) {
      insert(contentsOf: newElements, at: i)
    } else if let newElements = _specialize(newElements, for: BigSubstring.UnicodeScalarView.self) {
      insert(contentsOf: newElements, at: i)
    } else {
      _base.insert(contentsOf: BigString(_from: newElements), at: i)
    }
  }

  public mutating func insert(
    contentsOf newElements: __owned String.UnicodeScalarView,
    at i: Index
  ) {
    _base.insert(contentsOf: String(newElements), at: i)
  }

  public mutating func insert(
    contentsOf newElements: __owned Substring.UnicodeScalarView,
    at i: Index
  ) {
    _base.insert(contentsOf: Substring(newElements), at: i)
  }

  public mutating func insert(
    contentsOf newElements: __owned BigString.UnicodeScalarView,
    at i: Index
  ) {
    _base.insert(contentsOf: newElements._base, at: i)
  }

  public mutating func insert(
    contentsOf newElements: __owned BigSubstring.UnicodeScalarView,
    at i: Index
  ) {
    _base._insert(contentsOf: newElements._base, in: newElements._bounds, at: i)
  }

  @discardableResult
  public mutating func remove(at i: Index) -> UnicodeScalar {
    _base.removeUnicodeScalar(at: i)
  }

  public mutating func removeSubrange(_ bounds: Range<Index>) {
    _base._removeSubrange(bounds)
  }

  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    self._base = BigString()
  }
}
