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

/// Identifies a position within a contiguous storage region within a hash tree
/// node. Hash tree nodes have two storage regions, one for items and one for
/// children; the same `_HashSlot` type is used to refer to positions within both.
///
/// We use the term "slot" to refer to internal storage entries, to avoid
/// confusion with terms that sometimes appear in public API, such as
/// "index", "position" or "offset".
@usableFromInline
@frozen
internal struct _HashSlot {
  @usableFromInline
  internal var _value: UInt32

  @inlinable @inline(__always)
  internal init(_ value: UInt32) {
    self._value = value
  }

  @inlinable @inline(__always)
  internal init(_ value: UInt) {
    assert(value <= UInt32.max)
    self._value = UInt32(truncatingIfNeeded: value)
  }

  @inlinable @inline(__always)
  internal init(_ value: Int) {
    assert(value >= 0 && value <= UInt32.max)
    self._value = UInt32(truncatingIfNeeded: value)
  }
}

extension _HashSlot {
  @inlinable @inline(__always)
  internal static var zero: _HashSlot { _HashSlot(0) }
}

extension _HashSlot {
  @inlinable @inline(__always)
  internal var value: Int {
    Int(truncatingIfNeeded: _value)
  }
}

extension _HashSlot: Equatable {
  @inlinable @inline(__always)
  internal static func ==(left: Self, right: Self) -> Bool {
    left._value == right._value
  }
}

extension _HashSlot: Comparable {
  @inlinable @inline(__always)
  internal static func <(left: Self, right: Self) -> Bool {
    left._value < right._value
  }
}

extension _HashSlot: Hashable {
  @inlinable
  internal func hash(into hasher: inout Hasher) {
    hasher.combine(_value)
  }
}

extension _HashSlot: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    "\(_value)"
  }
}

extension _HashSlot: Strideable {
  @inlinable @inline(__always)
  internal func advanced(by n: Int) -> _HashSlot {
    assert(n >= 0 || value + n >= 0)
    return _HashSlot(_value &+ UInt32(truncatingIfNeeded: n))
  }

  @inlinable @inline(__always)
  internal func distance(to other: _HashSlot) -> Int {
    if self < other {
      return Int(truncatingIfNeeded: other._value - self._value)
    }
    return -Int(truncatingIfNeeded: self._value - other._value)
  }
}

extension _HashSlot {
  @inlinable @inline(__always)
  internal func next() -> _HashSlot {
    assert(_value < .max)
    return _HashSlot(_value &+ 1)
  }

  @inlinable @inline(__always)
  internal func previous() -> _HashSlot {
    assert(_value > 0)
    return _HashSlot(_value &- 1)
  }
}
