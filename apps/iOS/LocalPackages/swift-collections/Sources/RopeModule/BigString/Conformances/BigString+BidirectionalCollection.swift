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
extension BigString: BidirectionalCollection {
  public typealias SubSequence = BigSubstring

  public var isEmpty: Bool {
    _rope.summary.isZero
  }

  public var startIndex: Index {
    Index(_utf8Offset: 0)._knownCharacterAligned()
  }

  public var endIndex: Index {
    Index(_utf8Offset: _utf8Count)._knownCharacterAligned()
  }

  public var count: Int { _characterCount }

  @inline(__always)
  public func index(after i: Index) -> Index {
    _characterIndex(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    _characterIndex(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    _characterIndex(i, offsetBy: distance)
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    _characterIndex(i, offsetBy: distance, limitedBy: limit)
  }

  public func distance(from start: Index, to end: Index) -> Int {
    _characterDistance(from: start, to: end)
  }

  public subscript(position: Index) -> Character {
    self[_character: position]
  }

  public subscript(bounds: Range<Index>) -> BigSubstring {
    BigSubstring(self, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  public func index(roundingDown i: Index) -> Index {
    _characterIndex(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    _characterIndex(roundingUp: i)
  }
}
