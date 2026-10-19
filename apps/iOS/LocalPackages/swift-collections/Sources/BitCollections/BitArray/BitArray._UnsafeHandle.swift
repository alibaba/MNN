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
  /// An unsafe-unowned bitarray view over `UInt` storage, providing bit array
  /// primitives.
  @usableFromInline
  @frozen
  internal struct _UnsafeHandle {
    @usableFromInline
    internal typealias _BitPosition = _UnsafeBitSet.Index

    @usableFromInline
    internal let _words: UnsafeBufferPointer<_Word>

    @usableFromInline
    internal var _count: UInt

#if DEBUG
    /// True when this handle does not support table mutations.
    /// (This is only checked in debug builds.)
    @usableFromInline
    internal let _mutable: Bool
#endif

    @inline(__always)
    internal func ensureMutable() {
#if DEBUG
      assert(_mutable)
#endif
    }

    internal var _mutableWords: UnsafeMutableBufferPointer<_Word> {
      ensureMutable()
      return UnsafeMutableBufferPointer(mutating: _words)
    }

    @inlinable
    @inline(__always)
    internal init(
      words: UnsafeBufferPointer<_Word>,
      count: UInt,
      mutable: Bool
    ) {
      assert(count <= words.count * _Word.capacity)
      assert(count > (words.count - 1) * _Word.capacity)
      self._words = words
      self._count = count
#if DEBUG
      self._mutable = mutable
#endif
    }

    @inlinable
    @inline(__always)
    internal init(
      words: UnsafeMutableBufferPointer<_Word>,
      count: UInt,
      mutable: Bool
    ) {
      self.init(
        words: UnsafeBufferPointer(words),
        count: count,
        mutable: mutable)
    }
  }
}

extension BitArray._UnsafeHandle {
  internal var count: Int {
    Int(_count)
  }

  internal var end: _BitPosition {
    _BitPosition(_count)
  }

  internal func set(at position: Int) {
    ensureMutable()
    assert(position >= 0 && position < _count)
    let (word, bit) = _BitPosition(UInt(position)).split
    _mutableWords[word].insert(bit)
  }

  internal func clear(at position: Int) {
    ensureMutable()
    assert(position >= 0 && position < _count)
    let (word, bit) = _BitPosition(UInt(position)).split
    _mutableWords[word].remove(bit)
  }

  internal subscript(position: Int) -> Bool {
    get {
      assert(position >= 0 && position < _count)
      let (word, bit) = _BitPosition(UInt(position)).split
      return _words[word].contains(bit)
    }
    set {
      ensureMutable()
      assert(position >= 0 && position < _count)
      let (word, bit) = _BitPosition(UInt(position)).split
      if newValue {
        _mutableWords[word].insert(bit)
      } else {
        _mutableWords[word].remove(bit)
      }
    }
  }
}

extension BitArray._UnsafeHandle {
  internal mutating func fill(in range: Range<Int>) {
    ensureMutable()
    precondition(
      range.lowerBound >= 0 && range.upperBound <= count,
      "Range out of bounds")
    guard range.count > 0 else { return }
    let (lw, lb) = _BitPosition(range.lowerBound).split
    let (uw, ub) = _BitPosition(range.upperBound).endSplit
    let words = _mutableWords
    guard lw != uw else {
      words[lw].formUnion(_Word(from: lb, to: ub))
      return
    }
    words[lw].formUnion(_Word(upTo: lb).complement())
    for w in lw + 1 ..< uw {
      words[w] = _Word.allBits
    }
    words[uw].formUnion(_Word(upTo: ub))
  }

  internal mutating func clear(in range: Range<Int>) {
    ensureMutable()
    precondition(
      range.lowerBound >= 0 && range.upperBound <= count,
      "Range out of bounds")
    guard range.count > 0 else { return }
    let (lw, lb) = _BitPosition(range.lowerBound).split
    let (uw, ub) = _BitPosition(range.upperBound).endSplit
    let words = _mutableWords
    guard lw != uw else {
      words[lw].subtract(_Word(from: lb, to: ub))
      return
    }
    words[lw].subtract(_Word(upTo: lb).complement())
    for w in lw + 1 ..< uw {
      words[w] = _Word.empty
    }
    words[uw].subtract(_Word(upTo: ub))
  }
}
