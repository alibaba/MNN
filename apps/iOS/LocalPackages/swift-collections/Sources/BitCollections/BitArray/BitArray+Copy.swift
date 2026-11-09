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

extension BitArray {
  internal mutating func _copy(
    from range: Range<Int>,
    to target: Int
  ) {
    _update { handle in
      handle.copy(from: range, to: target)
    }
  }

  internal mutating func _copy(
    from range: Range<Int>,
    in source: UnsafeBufferPointer<_Word>,
    to target: Int
  ) {
    _update { handle in
      handle.copy(from: range, in: source, to: target)
    }
  }

  internal mutating func _copy(
    from source: BitArray,
    to target: Int
  ) {
    _copy(from: source[...], to: target)
  }

  internal mutating func _copy(
    from source: BitArray.SubSequence,
    to target: Int
  ) {
    let range = source._bounds
    source.base._storage.withUnsafeBufferPointer { words in
      self._copy(from: range, in: words, to: target)
    }
  }

  internal mutating func _copy<S: Sequence>(
    from source: S,
    to range: Range<Int>
  ) where S.Element == Bool {
    _update { $0.copy(from: source, to: range) }
  }
}

extension BitArray._UnsafeHandle {
  internal mutating func _copy(
    bits: _Word, count: UInt, to target: _BitPosition
  ) {
    assert(count <= _Word.capacity)
    assert(count == _Word.capacity || bits.shiftedDown(by: count).isEmpty)
    assert(target.value + count <= _count)
    let start = target.split
    let end = _BitPosition(target.value + count).endSplit
    let words = _mutableWords
    if start.word == end.word {
      let mask = _Word(from: start.bit, to: end.bit)
      words[start.word].formIntersection(mask.complement())
      words[start.word].formUnion(bits.shiftedUp(by: start.bit))
      return
    }
    assert(start.word + 1 == end.word)
    words[start.word].formIntersection(_Word(upTo: start.bit))
    words[start.word].formUnion(bits.shiftedUp(by: start.bit))
    words[end.word].formIntersection(_Word(upTo: end.bit).complement())
    words[end.word].formUnion(
      bits.shiftedDown(by: UInt(_Word.capacity) &- start.bit))
  }

  internal mutating func copy(
    from range: Range<Int>,
    to target: Int
  ) {
    assert(
      range.lowerBound >= 0 && range.upperBound <= self.count,
      "Source range out of bounds")
    copy(from: range, in: _words, to: target)
  }

  internal mutating func copy(
    from range: Range<Int>,
    in source: UnsafeBufferPointer<_Word>,
    to target: Int
  ) {
    ensureMutable()
    assert(
      range.lowerBound >= 0 && range.upperBound <= source.count * _Word.capacity,
      "Source range out of bounds")
    assert(
      target >= 0 && target + range.count <= count,
      "Target out of bounds")
    guard !range.isEmpty else { return }
    
    func goForward() -> Bool {
      let target = _BitPosition(target).split
      let lowerSource = _BitPosition(range.lowerBound).split
      let upperSource = _BitPosition(range.upperBound).endSplit

      
      let targetPtr = _words._ptr(at: target.word)
      let lowerSourcePtr = source._ptr(at: lowerSource.word)
      let upperSourcePtr = source._ptr(at: upperSource.word)

      if targetPtr < lowerSourcePtr || targetPtr > upperSourcePtr {
        return true
      }
      if targetPtr == lowerSourcePtr, target.bit < lowerSource.bit {
        return true
      }
      if targetPtr == upperSourcePtr, target.bit >= upperSource.bit {
        return true
      }
      return false
    }
    
    if goForward() {
      // Copy forward from a disjoint or following overlapping range.
      var src = _ChunkedBitsForwardIterator(words: source, range: range)
      var dst = _BitPosition(target)
      while let (bits, count) = src.next() {
        _copy(bits: bits, count: count, to: dst)
        dst.value += count
      }
      assert(dst.value == target + range.count)
    } else {
      // Copy backward from a non-following overlapping range.
      var src = _ChunkedBitsBackwardIterator(words: source, range: range)
      var dst = _BitPosition(target + range.count)
      while let (bits, count) = src.next() {
        dst.value -= count
        _copy(bits: bits, count: count, to: dst)
      }
      assert(dst.value == target)
    }
  }
}

extension BitArray._UnsafeHandle {
  internal mutating func copy(
    from source: some Sequence<Bool>,
    to range: Range<Int>
  ) {
    assert(range.lowerBound >= 0 && range.upperBound <= self.count)
    var pos = _BitPosition(range.lowerBound)
    var it = source.makeIterator()
    if pos.bit > 0 {
      let (bits, count) = it._nextChunk(
        maximumCount: UInt(_Word.capacity) - pos.bit)
      _copy(bits: bits, count: count, to: pos)
      pos.value += count
    }
    while true {
      let (bits, count) = it._nextChunk()
      guard count > 0 else { break }
      assert(pos.bit == 0)
      _copy(bits: bits, count: count, to: pos)
      pos.value += count
    }
    precondition(pos.value == range.upperBound)
  }
}
