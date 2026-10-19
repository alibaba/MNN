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

/// An ordered, random-access collection of `Bool` values, implemented as an
/// uncompressed bitmap of as many bits as the count of the array.
///
/// Bit arrays implement `RangeReplaceableCollection` and `MutableCollection`
/// and provide limited support for bitwise operations on same-sized arrays.
///
/// See `BitSet` for an alternative form of the same underlying data
/// structure, treating it as a set of nonnegative integers corresponding to
/// `true` bits.
public struct BitArray {
  @usableFromInline
  internal typealias _BitPosition = _UnsafeBitSet.Index

  @usableFromInline
  internal var _storage: [_Word]

  /// The number of bits in the bit array. This may less than the number of bits
  /// in `_storage` if the last word isn't fully filled.
  @usableFromInline
  internal var _count: UInt

  @usableFromInline
  internal init(_storage: [_Word], count: UInt) {
    assert(count <= _storage.count * _Word.capacity)
    assert(count > (_storage.count - 1) * _Word.capacity)
    self._storage = _storage
    self._count = count
  }

  @inline(__always)
  internal init(_storage: [_Word], count: Int) {
    self.init(_storage: _storage, count: UInt(count))
  }
}

extension BitArray: Sendable {}

extension BitArray {
  @inline(__always)
  internal func _read<R>(
    _ body: (_UnsafeHandle) throws -> R
  ) rethrows -> R {
    try _storage.withUnsafeBufferPointer { words in
      let handle = _UnsafeHandle(
        words: words, count: _count, mutable: false)
      return try body(handle)
    }
  }

  @inline(__always)
  internal mutating func _update<R>(
    _ body: (inout _UnsafeHandle) throws -> R
  ) rethrows -> R {
    defer {
      _checkInvariants()
    }
    return try _storage.withUnsafeMutableBufferPointer { words in
      var handle = _UnsafeHandle(words: words, count: _count, mutable: true)
      return try body(&handle)
    }
  }
  
  internal mutating func _removeLast() {
    assert(_count > 0)
    _count -= 1
    let bit = _BitPosition(_count).bit
    if bit == 0 {
      _storage.removeLast()
    } else {
      _storage[_storage.count - 1].remove(bit)
    }
  }

  internal mutating func _removeLast(_ n: Int) {
    assert(n >= 0 && n <= _count)
    guard n > 0 else { return }
    let wordCount = _Word.wordCount(forBitCount: _count - UInt(n))
    if wordCount < _storage.count {
      _storage.removeLast(_storage.count - wordCount)
    }
    _count -= UInt(n)
    let (word, bit) = _BitPosition(_count).split
    if bit > 0 {
      _storage[word].formIntersection(_Word(upTo: bit))
    }
  }

  internal mutating func _extend(by n: Int, with paddingBit: Bool = false) {
    assert(n >= 0)
    guard n > 0 else { return }
    let newCount = _count + UInt(n)
    let orig = _storage.count
    let new = _Word.wordCount(forBitCount: newCount)
    if paddingBit == false {
      _storage.append(contentsOf: repeatElement(.empty, count: new - orig))
    } else {
      let (w1, b1) = _BitPosition(_count).split
      let (w2, b2) = _BitPosition(newCount).split
      if w1 < _storage.count {
        _storage[w1].formUnion(_Word(upTo: b1).complement())
      }
      _storage.append(contentsOf: repeatElement(.allBits, count: new - orig))
      if w2 < _storage.count {
        _storage[w2].formIntersection(_Word(upTo: b2))
      }
    }
    _count = newCount
  }
}
