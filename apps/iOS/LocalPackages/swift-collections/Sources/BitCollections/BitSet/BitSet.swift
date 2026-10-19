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

/// A sorted collection of small nonnegative integers, implemented as an
/// uncompressed bitmap of as many bits as the value of the largest member.
///
/// Bit sets implement `SetAlgebra` and provide efficient implementations
/// for set operations based on standard binary logic operations.
///
/// See `BitArray` for an alternative form of the same underlying data
/// structure, treating it as a mutable random-access collection of `Bool`
/// values.
public struct BitSet {
  @usableFromInline
  internal var _storage: [_Word]

  @usableFromInline
  init(_rawStorage storage: [_Word]) {
    self._storage = storage
    _checkInvariants()
  }
}

extension BitSet: Sendable {}

extension BitSet {
  @inline(__always)
  internal func _read<R>(
    _ body: (_UnsafeHandle) throws -> R
  ) rethrows -> R {
    try _storage.withUnsafeBufferPointer { words in
      let handle = _UnsafeHandle(words: words, mutable: false)
      return try body(handle)
    }
  }

  @usableFromInline
  internal var _capacity: UInt {
    UInt(_storage.count) &* UInt(_Word.capacity)
  }

  internal mutating func _ensureCapacity(limit capacity: UInt) {
    let desired = _UnsafeHandle.wordCount(forCapacity: capacity)
    guard _storage.count < desired else { return }
    _storage.append(
      contentsOf: repeatElement(.empty, count: desired - _storage.count))
  }

  internal mutating func _ensureCapacity(forValue value: UInt) {
    let desiredWord = _UnsafeHandle.Index(value).word
    guard desiredWord >= _storage.count else { return }
    _storage.append(
      contentsOf:
        repeatElement(.empty, count: desiredWord - _storage.count + 1))
  }

  internal mutating func _shrink() {
    let suffix = _read { $0._emptySuffix() }
    if suffix > 0 { _storage.removeLast(suffix) }
  }

  @inline(__always)
  internal mutating func _update<R>(
    _ body: (inout _UnsafeHandle) throws -> R
  ) rethrows -> R {
    defer {
      _checkInvariants()
    }
    return try _storage.withUnsafeMutableBufferPointer { words in
      var handle = _UnsafeHandle(words: words, mutable: true)
      return try body(&handle)
    }
  }

  @inline(__always)
  internal mutating func _updateThenShrink<R>(
    _ body: (_ handle: inout _UnsafeHandle, _ shrink: inout Bool) throws -> R
  ) rethrows -> R {
    var shrink = true
    defer {
      if shrink { _shrink() }
      _checkInvariants()
    }
    return try _storage.withUnsafeMutableBufferPointer { words in
      var handle = _UnsafeHandle(words: words, mutable: true)
      return try body(&handle, &shrink)
    }
  }
}
