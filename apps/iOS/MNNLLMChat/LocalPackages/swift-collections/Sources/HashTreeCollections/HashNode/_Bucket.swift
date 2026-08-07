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

/// Identifies an entry in the hash table inside a node.
/// (Internally, a number between 0 and 31.)
@usableFromInline
@frozen
internal struct _Bucket {
  @usableFromInline
  internal var _value: UInt8

  @inlinable @inline(__always)
  internal init(_value: UInt8) {
    assert(_value < _Bitmap.capacity || _value == .max)
    self._value = _value
  }
}

extension _Bucket {
  @inlinable @inline(__always)
  internal var value: UInt { UInt(truncatingIfNeeded: _value) }

  @inlinable @inline(__always)
  internal init(_ value: UInt) {
    assert(value < _Bitmap.capacity || value == .max)
    self._value = UInt8(truncatingIfNeeded: value)
  }
}

extension _Bucket {
  @inlinable @inline(__always)
  static var bitWidth: Int { _Bitmap.capacity.trailingZeroBitCount }

  @inlinable @inline(__always)
  static var bitMask: UInt { UInt(bitPattern: _Bitmap.capacity) &- 1 }

  @inlinable @inline(__always)
  static var invalid: _Bucket { _Bucket(_value: .max) }

  @inlinable @inline(__always)
  var isInvalid: Bool { _value == .max }
}

extension _Bucket: Equatable {
  @inlinable @inline(__always)
  internal static func ==(left: Self, right: Self) -> Bool {
    left._value == right._value
  }
}

extension _Bucket: Comparable {
  @inlinable @inline(__always)
  internal static func <(left: Self, right: Self) -> Bool {
    left._value < right._value
  }
}

extension _Bucket: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    String(_value, radix: _Bitmap.capacity)
  }
}
