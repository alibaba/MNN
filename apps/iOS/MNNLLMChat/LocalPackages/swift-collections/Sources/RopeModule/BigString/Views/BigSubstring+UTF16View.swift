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
  public struct UTF16View: Sendable {
    internal var _base: BigString
    internal var _bounds: Range<Index>

    public init(_unchecked base: BigString, in bounds: Range<Index>) {
      self._base = base
      self._bounds = bounds
    }

    public init(_ base: BigString, in bounds: Range<Index>) {
      self._base = base
      let lower = base._utf16Index(roundingDown: bounds.lowerBound)
      let upper = base._utf16Index(roundingDown: bounds.upperBound)
      self._bounds = Range(uncheckedBounds: (lower, upper))
    }

    internal init(_substring: BigSubstring) {
      self.init(_unchecked: _substring._base, in: _substring._bounds)
    }
  }

  public var utf16: UTF16View {
    UTF16View(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public init?(_ utf16: BigSubstring.UTF16View) {
    guard
      !utf16.startIndex._isUTF16TrailingSurrogate,
      !utf16.endIndex._isUTF16TrailingSurrogate
    else {
      return nil
    }
    self.init(_from: utf16._base, in: utf16._bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View {
  public var base: BigString.UTF16View { _base.utf16 }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    var i1 = left._bounds.lowerBound
    var i2 = right._bounds.lowerBound

    var j1 = left._bounds.upperBound
    var j2 = right._bounds.upperBound

    // Compare first code units, if they're trailing surrogates.
    guard i1._isUTF16TrailingSurrogate == i2._isUTF16TrailingSurrogate else { return false }
    if i1._isUTF16TrailingSurrogate {
      guard left[i1] == right[i2] else { return false }
      left.formIndex(after: &i1)
      left.formIndex(after: &i2)
    }
    guard i1 < j1, i2 < j2 else { return i1 == j1 && i2 == j2 }

    // Compare last code units, if they're trailing surrogates.
    guard j1._isUTF16TrailingSurrogate == j2._isUTF16TrailingSurrogate else { return false }
    if j1._isUTF16TrailingSurrogate {
      left.formIndex(before: &j1)
      right.formIndex(before: &j2)
      guard left[j1] == right[j2] else { return false}
    }

    return BigString.utf8IsEqual(left._base, in: i1 ..< j1, to: right._base, in: i2 ..< j2)
  }

  public func isIdentical(to other: Self) -> Bool {
    guard self._base.isIdentical(to: other._base) else { return false }
    return self._bounds == other._bounds
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View: Hashable {
  public func hash(into hasher: inout Hasher) {
    for codeUnit in self {
      hasher.combine(codeUnit)
    }
    hasher.combine(0xFFFF as UInt16)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View: Sequence {
  public typealias Element = UInt16

  public struct Iterator: IteratorProtocol {
    var _it: BigString.UTF16View.Iterator
    var _end: BigString.Index

    init(_substring: BigSubstring.UTF16View) {
      self._it = .init(_base: _substring._base, from: _substring.startIndex)
      self._end = _substring.endIndex
    }

    public mutating func next() -> UInt16? {
      guard _it._index < _end else { return nil }
      return _it.next()
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View: BidirectionalCollection {
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
    return _base._utf16Index(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    precondition(i > startIndex, "Can't advance below start index")
    return _base._utf16Index(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    let j = _base._utf16Index(i, offsetBy: distance)
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    guard let j = _base._utf16Index(i, offsetBy: distance, limitedBy: limit) else { return nil }
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func distance(from start: Index, to end: Index) -> Int {
    precondition(start >= startIndex && start <= endIndex, "Index out of bounds")
    precondition(end >= startIndex && end <= endIndex, "Index out of bounds")
    return _base._utf16Distance(from: start, to: end)
  }

  public subscript(position: Index) -> UInt16 {
    precondition(position >= startIndex && position < endIndex, "Index out of bounds")
    return _base[_utf16: position]
  }

  public subscript(bounds: Range<Index>) -> Self {
    precondition(
      bounds.lowerBound >= startIndex && bounds.upperBound <= endIndex,
      "Range out of bounds")
    return Self(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring.UTF16View {
  public func index(roundingDown i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._utf16Index(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base._utf16Index(roundingUp: i)
  }
}
