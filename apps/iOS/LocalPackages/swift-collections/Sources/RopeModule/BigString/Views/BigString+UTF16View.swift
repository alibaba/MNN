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
extension BigString {
  public struct UTF16View: Sendable {
    var _base: BigString

    @inline(__always)
    init(_base: BigString) {
      self._base = _base
    }
  }

  @inline(__always)
  public var utf16: UTF16View {
    UTF16View(_base: self)
  }

  public init(_ utf16: UTF16View) {
    self = utf16._base
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UTF16View: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    BigString.utf8IsEqual(left._base, to: right._base)
  }

  public func isIdentical(to other: Self) -> Bool {
    self._base.isIdentical(to: other._base)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UTF16View: Hashable {
  public func hash(into hasher: inout Hasher) {
    _base.hashUTF8(into: &hasher)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UTF16View: Sequence {
  public typealias Element = UInt16

  public struct Iterator {
    internal let _base: BigString
    internal var _index: BigString.Index

    internal init(_base: BigString, from start: BigString.Index) {
      self._base = _base
      self._index = _base._utf16Index(roundingDown: start)
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_base: _base, from: startIndex)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UTF16View.Iterator: IteratorProtocol {
  public typealias Element = UInt16

  public mutating func next() -> UInt16? {
    guard _index < _base.endIndex else { return nil }
    let ri = _index._rope!
    var ci = _index._chunkIndex
    let chunk = _base._rope[ri]
    let result = chunk.string.utf16[ci]

    chunk.string.utf16.formIndex(after: &ci)
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
extension BigString.UTF16View: BidirectionalCollection {
  public typealias Index = BigString.Index
  public typealias SubSequence = BigSubstring.UTF16View

  @inline(__always)
  public var startIndex: Index { _base.startIndex }

  @inline(__always)
  public var endIndex: Index { _base.endIndex }

  public var count: Int { _base._utf16Count }

  @inline(__always)
  public func index(after i: Index) -> Index {
    _base._utf16Index(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    _base._utf16Index(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    _base._utf16Index(i, offsetBy: distance)
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    _base._utf16Index(i, offsetBy: distance, limitedBy: limit)
  }

  public func distance(from start: Index, to end: Index) -> Int {
    _base._utf16Distance(from: start, to: end)
  }

  public subscript(position: Index) -> UInt16 {
    _base[_utf16: position]
  }

  public subscript(bounds: Range<Index>) -> BigSubstring.UTF16View {
    BigSubstring.UTF16View(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString.UTF16View {
  public func index(roundingDown i: Index) -> Index {
    _base._utf16Index(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    _base._utf16Index(roundingUp: i)
  }
}
