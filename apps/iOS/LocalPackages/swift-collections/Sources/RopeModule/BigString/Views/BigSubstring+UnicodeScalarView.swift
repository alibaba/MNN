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

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring {
  public struct UnicodeScalarView: Sendable {
    internal var _base: BigString
    internal var _bounds: Range<Index>

    public init(_unchecked base: BigString, in bounds: Range<Index>) {
      assert(bounds.lowerBound == base._unicodeScalarIndex(roundingDown: bounds.lowerBound))
      assert(bounds.upperBound == base._unicodeScalarIndex(roundingDown: bounds.upperBound))
      self._base = base
      self._bounds = bounds
    }

    public init(_ base: BigString, in bounds: Range<Index>) {
      self._base = base
      let lower = base._unicodeScalarIndex(roundingDown: bounds.lowerBound)
      let upper = base._unicodeScalarIndex(roundingDown: bounds.upperBound)
      self._bounds = Range(uncheckedBounds: (lower, upper))
    }

    internal init(_substring: BigSubstring) {
      self.init(_unchecked: _substring._base, in: _substring._bounds)
    }
  }

  public var unicodeScalars: UnicodeScalarView {
    get {
      UnicodeScalarView(_substring: self)
    }
    set {
      self = Self(newValue._base, in: newValue._bounds)
    }
    _modify {
      var view = UnicodeScalarView(_unchecked: _base, in: _bounds)
      self = Self()
      defer {
        self = Self(view._base, in: view._bounds)
      }
      yield &view
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public init(_ unicodeScalars: BigSubstring.UnicodeScalarView) {
    self.init(_from: unicodeScalars._base, in: unicodeScalars._bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView {
  public var base: BigString.UnicodeScalarView { _base.unicodeScalars }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self.init(value.unicodeScalars)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: CustomStringConvertible {
  public var description: String {
    String(_from: _base, in: _bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: CustomDebugStringConvertible {
  public var debugDescription: String {
    description.debugDescription
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    BigString.utf8IsEqual(left._base, in: left._bounds, to: right._base, in: right._bounds)
  }

  public func isIdentical(to other: Self) -> Bool {
    guard self._base.isIdentical(to: other._base) else { return false }
    return self._bounds == other._bounds
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: Hashable {
  public func hash(into hasher: inout Hasher) {
    _base.hashUTF8(into: &hasher, from: _bounds.lowerBound, to: _bounds.upperBound)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: Sequence {
  public typealias Element = UnicodeScalar

  public struct Iterator: IteratorProtocol {
    var _it: BigString.UnicodeScalarView.Iterator
    let _end: BigString.Index

    internal init(_substring: BigSubstring.UnicodeScalarView) {
      self._it = .init(_base: _substring._base, from: _substring.startIndex)
      self._end = _substring._base.resolve(_substring.endIndex, preferEnd: true)
    }

    public mutating func next() -> UnicodeScalar? {
      guard _it._index < _end else { return nil }
      return _it.next()
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: BidirectionalCollection {
  public typealias Index = BigString.Index
  public typealias SubSequence = Self

  @inline(__always)
  public var startIndex: Index { _bounds.lowerBound }

  @inline(__always)
  public var endIndex: Index { _bounds.upperBound }

  public var count: Int {
    distance(from: _bounds.lowerBound, to: _bounds.upperBound)
  }

  @inline(__always)
  public func index(after i: Index) -> Index {
    precondition(i < endIndex, "Can't advance above end index")
    return _base._unicodeScalarIndex(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    precondition(i > startIndex, "Can't advance below start index")
    return _base._unicodeScalarIndex(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    let j = _base._unicodeScalarIndex(i, offsetBy: distance)
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    guard let j = _base._unicodeScalarIndex(i, offsetBy: distance, limitedBy: limit) else {
      return nil
    }
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func distance(from start: Index, to end: Index) -> Int {
    precondition(start >= startIndex && start <= endIndex, "Index out of bounds")
    precondition(end >= startIndex && end <= endIndex, "Index out of bounds")
    return _base._unicodeScalarDistance(from: start, to: end)
  }

  public subscript(position: Index) -> UnicodeScalar {
    precondition(position >= startIndex && position < endIndex, "Index out of bounds")
    return _base[_unicodeScalar: position]
  }

  public subscript(bounds: Range<Index>) -> Self {
    precondition(
      bounds.lowerBound >= startIndex && bounds.upperBound <= endIndex,
      "Range out of bounds")
    return Self(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView {
  public func index(roundingDown i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._unicodeScalarIndex(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._unicodeScalarIndex(roundingUp: i)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView {
  /// Run the closure `body` to mutate the contents of this view within `range`, then update
  /// the bounds of this view to maintain their logical position in the resulting string.
  /// The `range` argument is validated to be within the original bounds of the substring.
  internal mutating func _mutateBasePreservingBounds<R>(
    in range: Range<Index>,
    with body: (inout BigString.UnicodeScalarView) -> R
  ) -> R {
    precondition(
      range.lowerBound >= _bounds.lowerBound && range.upperBound <= _bounds.upperBound,
      "Range out of bounds")

    let startOffset = self.startIndex.utf8Offset
    let endOffset = self.endIndex.utf8Offset
    let oldCount = self._base._utf8Count

    var view = BigString.UnicodeScalarView(_base: self._base)
    self._base = BigString()

    defer {
      // The Unicode scalar view is regular -- we just need to maintain the UTF-8 offsets of
      // our bounds across the mutation. No extra adjustment/rounding is necessary.
      self._base = view._base
      let delta = self._base._utf8Count - oldCount
      let start = _base._utf8Index(at: startOffset)._knownScalarAligned()
      let end = _base._utf8Index(at: endOffset + delta)._knownScalarAligned()
      self._bounds = start ..< end
    }
    return body(&view)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UnicodeScalarView: RangeReplaceableCollection {
  public init() {
    self.init(_substring: BigSubstring())
  }
  
  public mutating func reserveCapacity(_ n: Int) {
    // Do nothing.
  }
  
  public mutating func replaceSubrange(
    _ subrange: Range<Index>, 
    with newElements: __owned some Sequence<UnicodeScalar> // Note: Sequence, not Collection
  ) {
    _mutateBasePreservingBounds(in: subrange) { $0.replaceSubrange(subrange, with: newElements) }
  }

  public init(_ elements: some Sequence<UnicodeScalar>) {
    let base = BigString.UnicodeScalarView(elements)
    self.init(base._base, in: base.startIndex ..< base.endIndex)
  }

  public init(repeating repeatedValue: UnicodeScalar, count: Int) {
    let base = BigString.UnicodeScalarView(repeating: repeatedValue, count: count)
    self.init(base._base, in: base.startIndex ..< base.endIndex)
  }

  public mutating func append(_ newElement: UnicodeScalar) {
    let i = endIndex
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(newElement, at: i)
    }
  }

  public mutating func append(
    contentsOf newElements: __owned some Sequence<UnicodeScalar>
  ) {
    let i = endIndex
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(contentsOf: newElements, at: i)
    }
  }

  public mutating func insert(_ newElement: UnicodeScalar, at i: Index) {
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(newElement, at: i)
    }
  }


  public mutating func insert(
    contentsOf newElements: __owned some Sequence<UnicodeScalar>, // Note: Sequence, not Collection
    at i: Index
  ) {
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(contentsOf: newElements, at: i)
    }
  }

  @discardableResult
  public mutating func remove(at i: Index) -> UnicodeScalar {
    let j = self.index(after: i)
    return _mutateBasePreservingBounds(in: i ..< j) {
      $0.remove(at: i)
    }
  }

  public mutating func removeSubrange(_ bounds: Range<Index>) {
    _mutateBasePreservingBounds(in: bounds) {
      $0.removeSubrange(bounds)
    }
  }

  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    let bounds = _bounds
    _mutateBasePreservingBounds(in: bounds) {
      $0.removeSubrange(bounds)
    }
    assert(_bounds.isEmpty)
  }
}
