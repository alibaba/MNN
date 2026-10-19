//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

/// A set of `_Bucket` values, represented by a 32-bit wide bitset.
@usableFromInline
@frozen
internal struct _Bitmap {
  @usableFromInline
  internal typealias Value = UInt32

  @usableFromInline
  internal var _value: Value

  @inlinable @inline(__always)
  init(_value: Value) {
    self._value = _value
  }

  @inlinable @inline(__always)
  init(bitPattern: Int) {
    self._value = Value(bitPattern)
  }

  @inlinable @inline(__always)
  internal init(_ bucket: _Bucket) {
    assert(bucket.value < Self.capacity)
    _value = (1 &<< bucket.value)
  }

  @inlinable @inline(__always)
  internal init(_ bucket1: _Bucket, _ bucket2: _Bucket) {
    assert(bucket1.value < Self.capacity && bucket2.value < Self.capacity)
    assert(bucket1 != bucket2)
    _value = (1 &<< bucket1.value) | (1 &<< bucket2.value)
  }

  @inlinable
  internal init(upTo bucket: _Bucket) {
    assert(bucket.value < Self.capacity)
    _value = (1 &<< bucket.value) &- 1
  }
}

extension _Bitmap: Equatable {
  @inlinable @inline(__always)
  internal static func ==(left: Self, right: Self) -> Bool {
    left._value == right._value
  }
}

extension _Bitmap: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    let b = String(_value, radix: 2)
    let bits = String(repeating: "0", count: _Bitmap.capacity - b.count) + b
    return "\(String(bits.reversed())) (\(_value))"
  }
}

extension _Bitmap {
  @inlinable @inline(__always)
  internal static var empty: Self { .init(_value: 0) }

  @inlinable @inline(__always)
  internal static var capacity: Int { Value.bitWidth }

  @inlinable @inline(__always)
  internal static var bitWidth: Int { capacity.trailingZeroBitCount }

  @inlinable @inline(__always)
  internal var count: Int { _value.nonzeroBitCount }

  @inlinable @inline(__always)
  internal var capacity: Int { Value.bitWidth }

  @inlinable @inline(__always)
  internal var isEmpty: Bool { _value == 0 }

  @inlinable @inline(__always)
  internal var hasExactlyOneMember: Bool {
    _value != 0 && _value & (_value &- 1) == 0
  }

  @inlinable @inline(__always)
  internal var first: _Bucket? {
    guard !isEmpty else { return nil }
    return _Bucket(
      _value: UInt8(truncatingIfNeeded: _value.trailingZeroBitCount))
  }

  @inlinable @inline(__always)
  internal mutating func popFirst() -> _Bucket? {
    guard let bucket = first else { return nil }
    _value &= _value &- 1 // Clear lowest nonzero bit.
    return bucket
  }
}

extension _Bitmap {
  @inlinable @inline(__always)
  internal func contains(_ bucket: _Bucket) -> Bool {
    assert(bucket.value < capacity)
    return _value & (1 &<< bucket.value) != 0
  }

  @inlinable @inline(__always)
  internal mutating func insert(_ bucket: _Bucket) {
    assert(bucket.value < capacity)
    _value |= (1 &<< bucket.value)
  }

  @inlinable @inline(__always)
  internal func inserting(_ bucket: _Bucket) -> _Bitmap {
    assert(bucket.value < capacity)
    return _Bitmap(_value: _value | (1 &<< bucket.value))
  }

  @inlinable @inline(__always)
  internal mutating func remove(_ bucket: _Bucket) {
    assert(bucket.value < capacity)
    _value &= ~(1 &<< bucket.value)
  }

  @inlinable @inline(__always)
  internal func removing(_ bucket: _Bucket) -> _Bitmap {
    assert(bucket.value < capacity)
    return _Bitmap(_value: _value & ~(1 &<< bucket.value))
  }

  @inlinable @inline(__always)
  internal func slot(of bucket: _Bucket) -> _HashSlot {
    _HashSlot(_value._rank(ofBit: bucket.value))
  }

  @inlinable @inline(__always)
  internal func bucket(at slot: _HashSlot) -> _Bucket {
    _Bucket(_value._bit(ranked: slot.value)!)
  }
}

extension _Bitmap {
  @inlinable @inline(__always)
  internal func isSubset(of other: Self) -> Bool {
    _value & ~other._value == 0
  }

  @inlinable @inline(__always)
  internal func isDisjoint(with other: Self) -> Bool {
    _value & other._value == 0
  }

  @inlinable @inline(__always)
  internal func union(_ other: Self) -> Self {
    Self(_value: _value | other._value)
  }

  @inlinable @inline(__always)
  internal func intersection(_ other: Self) -> Self {
    Self(_value: _value & other._value)
  }

  @inlinable @inline(__always)
  internal func symmetricDifference(_ other: Self) -> Self {
    Self(_value: _value & other._value)
  }

  @inlinable @inline(__always)
  internal func subtracting(_ other: Self) -> Self {
    Self(_value: _value & ~other._value)
  }
}

extension _Bitmap: Sequence {
  @usableFromInline
  internal typealias Element = (bucket: _Bucket, slot: _HashSlot)

  @usableFromInline
  @frozen
  internal struct Iterator: IteratorProtocol {
    @usableFromInline
    internal var bitmap: _Bitmap

    @usableFromInline
    internal var slot: _HashSlot

    @inlinable
    internal init(_ bitmap: _Bitmap) {
      self.bitmap = bitmap
      self.slot = .zero
    }

    /// Return the index of the lowest set bit in this word,
    /// and also destructively clear it.
    @inlinable
    internal mutating func next() -> Element? {
      guard let bucket = bitmap.popFirst() else { return nil }
      defer { slot = slot.next() }
      return (bucket, slot)
    }
  }

  @inlinable
  internal var underestimatedCount: Int { count }

  @inlinable
  internal func makeIterator() -> Iterator {
    Iterator(self)
  }
}
