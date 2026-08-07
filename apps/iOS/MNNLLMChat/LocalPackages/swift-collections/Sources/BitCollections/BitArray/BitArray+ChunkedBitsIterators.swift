//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

internal struct _ChunkedBitsForwardIterator {
  internal typealias _BitPosition = _UnsafeBitSet.Index

  internal let words: UnsafeBufferPointer<_Word>
  internal let end: _BitPosition
  internal var position: _BitPosition
  
  internal init(
    words: UnsafeBufferPointer<_Word>,
    range: Range<Int>
  ) {
    assert(range.lowerBound >= 0)
    assert(range.upperBound <= words.count * _Word.capacity)
    self.words = words
    self.end = _BitPosition(range.upperBound)
    self.position = _BitPosition(range.lowerBound)
  }
  
  mutating func next() -> (bits: _Word, count: UInt)? {
    guard position < end else { return nil }
    let (w, b) = position.split
    if w == end.word {
      position = end
      return (
        bits: words[w]
          .intersection(_Word(upTo: end.bit))
          .shiftedDown(by: b),
        count: end.bit - b)
    }
    let c = UInt(_Word.capacity) - b
    position.value += c
    return (bits: words[w].shiftedDown(by: b), count: c)
  }
}

internal struct _ChunkedBitsBackwardIterator {
  internal typealias _BitPosition = _UnsafeBitSet.Index

  internal let words: UnsafeBufferPointer<_Word>
  internal let start: _BitPosition
  internal var position: _BitPosition
  
  internal init(
    words: UnsafeBufferPointer<_Word>,
    range: Range<Int>
  ) {
    assert(range.lowerBound >= 0)
    assert(range.upperBound <= words.count * _Word.capacity)
    self.words = words
    self.start = _BitPosition(range.lowerBound)
    self.position = _BitPosition(range.upperBound)
  }
  
  internal mutating func next() -> (bits: _Word, count: UInt)? {
    guard position > start else { return nil }
    let (w, b) = position.endSplit
    if w == start.word {
      position = start
      return (
        bits: words[w]
          .intersection(_Word(upTo: b))
          .shiftedDown(by: start.bit),
        count: b - start.bit)
    }
    let c = b
    position.value -= c
    return (bits: words[w].intersection(_Word(upTo: b)), count: c)
  }
}

extension IteratorProtocol where Element == Bool {
  mutating func _nextChunk(
    maximumCount: UInt = UInt(_Word.capacity)
  ) -> (bits: _Word, count: UInt) {
    assert(maximumCount <= _Word.capacity)
    var bits = _Word.empty
    var c: UInt = 0
    while let v = next() {
      if v { bits.insert(c) }
      c += 1
      if c == maximumCount { break }
    }
    return (bits, c)
  }
}
