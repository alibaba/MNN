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

@usableFromInline
@frozen // Not really! This module isn't ABI stable
internal struct _RopePath<Summary: RopeSummary> {
  // ┌──────────────────────────────────┬────────┐
  // │ b63:b8                           │ b7:b0  │
  // ├──────────────────────────────────┼────────┤
  // │ path                             │ height │
  // └──────────────────────────────────┴────────┘
  @usableFromInline internal var _value: UInt64

  @inlinable @inline(__always)
  internal static var _pathBitWidth: Int { 56 }

  @inlinable
  internal init(_value: UInt64) {
    self._value = _value
  }

  @inlinable
  internal init(height: UInt8) {
    self._value = UInt64(truncatingIfNeeded: height)
    assert((Int(height) + 1) * Summary.nodeSizeBitWidth <= Self._pathBitWidth)
  }
}

extension Rope {
  @usableFromInline internal typealias _Path = _RopePath<Summary>
}

extension _RopePath: Equatable {
  @inlinable
  internal static func ==(left: Self, right: Self) -> Bool {
    left._value == right._value
  }
}
extension _RopePath: Hashable {
  @inlinable
  internal func hash(into hasher: inout Hasher) {
    hasher.combine(_value)
  }
}

extension _RopePath: Comparable {
  @inlinable
  internal static func <(left: Self, right: Self) -> Bool {
    left._value < right._value
  }
}

extension _RopePath: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    var r = "<"
    for h in stride(from: height, through: 0, by: -1) {
      r += "\(self[h])"
      if h > 0 { r += ", " }
    }
    r += ">"
    return r
  }
}

extension _RopePath {
  @inlinable
  internal var height: UInt8 {
    UInt8(truncatingIfNeeded: _value)
  }

  @inlinable
  internal mutating func popRoot() {
    let heightMask: UInt64 = 255
    let h = height
    assert(h > 0 && self[h] == 0)
    _value &= ~heightMask
    _value |= UInt64(truncatingIfNeeded: h - 1) & heightMask
  }

  @inlinable
  internal subscript(height: UInt8) -> Int {
    get {
      assert(height <= self.height)
      let shift = 8 + Int(height) * Summary.nodeSizeBitWidth
      let mask: UInt64 = (1 &<< Summary.nodeSizeBitWidth) &- 1
      return numericCast((_value &>> shift) & mask)
    }
    set {
      assert(height <= self.height)
      assert(newValue >= 0 && newValue <= Summary.maxNodeSize)
      let shift = 8 + Int(height) * Summary.nodeSizeBitWidth
      let mask: UInt64 = (1 &<< Summary.nodeSizeBitWidth) &- 1
      _value &= ~(mask &<< shift)
      _value |= numericCast(newValue) &<< shift
    }
  }

  @inlinable
  internal func isEmpty(below height: UInt8) -> Bool {
    let shift = Int(height) * Summary.nodeSizeBitWidth
    assert(shift + Summary.nodeSizeBitWidth <= Self._pathBitWidth)
    let mask: UInt64 = ((1 &<< shift) - 1) &<< 8
    return (_value & mask) == 0
  }

  @inlinable
  internal mutating func clear(below height: UInt8) {
    let shift = Int(height) * Summary.nodeSizeBitWidth
    assert(shift + Summary.nodeSizeBitWidth <= Self._pathBitWidth)
    let mask: UInt64 = ((1 &<< shift) - 1) &<< 8
    _value &= ~mask
  }
}

