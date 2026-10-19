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
  public struct UTF8View: Sendable {
    internal var _base: BigString
    internal var _bounds: Range<Index>

    public init(_unchecked base: BigString, in bounds: Range<Index>) {
      self._base = base
      self._bounds = bounds
    }

    public init(_ base: BigString, in bounds: Range<Index>) {
      self._base = base
      let lower = base._utf8Index(roundingDown: bounds.lowerBound)
      let upper = base._utf8Index(roundingDown: bounds.upperBound)
      self._bounds = Range(uncheckedBounds: (lower, upper))
    }

    internal init(_substring: BigSubstring) {
      self.init(_unchecked: _substring._base, in: _substring._bounds)
    }
  }

  public var utf8: UTF8View {
    UTF8View(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public init?(_ utf8: BigSubstring.UTF8View) {
    guard
      utf8._base.unicodeScalars.index(roundingDown: utf8.startIndex) == utf8.startIndex,
      utf8._base.unicodeScalars.index(roundingDown: utf8.endIndex) == utf8.endIndex
    else {
      return nil
    }
    self.init(_from: utf8._base, in: utf8._bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View {
  public var base: BigString.UTF8View { _base.utf8 }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    BigString.utf8IsEqual(left._base, in: left._bounds, to: right._base, in: right._bounds)
  }

  public func isIdentical(to other: Self) -> Bool {
    guard self._base.isIdentical(to: other._base) else { return false }
    return self._bounds == other._bounds
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View: Hashable {
  public func hash(into hasher: inout Hasher) {
    _base.hashUTF8(into: &hasher, from: _bounds.lowerBound, to: _bounds.upperBound)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View: Sequence {
  public typealias Element = UInt8

  public struct Iterator: IteratorProtocol {
    var _it: BigString.UTF8View.Iterator
    var _end: BigString.Index

    init(_substring: BigSubstring.UTF8View) {
      self._it = .init(_base: _substring._base, from: _substring.startIndex)
      self._end = _substring.endIndex
    }

    public mutating func next() -> UInt8? {
      guard _it._index < _end else { return nil }
      return _it.next()
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View: BidirectionalCollection {
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
    return _base._utf8Index(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    precondition(i > startIndex, "Can't advance below start index")
    return _base._utf8Index(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    let j = _base._utf8Index(i, offsetBy: distance)
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    guard let j = _base._utf8Index(i, offsetBy: distance, limitedBy: limit) else { return nil }
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func distance(from start: Index, to end: Index) -> Int {
    precondition(start >= startIndex && start <= endIndex, "Index out of bounds")
    precondition(end >= startIndex && end <= endIndex, "Index out of bounds")
    return _base._utf8Distance(from: start, to: end)
  }

  public subscript(position: Index) -> UInt8 {
    precondition(position >= startIndex && position < endIndex, "Index out of bounds")
    return _base[_utf8: position]
  }

  public subscript(bounds: Range<Index>) -> Self {
    precondition(
      bounds.lowerBound >= startIndex && bounds.upperBound <= endIndex,
      "Range out of bounds")
    return Self(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF8View {
  public func index(roundingDown i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._utf8Index(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._utf8Index(roundingUp: i)
  }
}
