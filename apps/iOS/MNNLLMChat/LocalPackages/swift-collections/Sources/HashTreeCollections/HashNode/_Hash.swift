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

/// An abstract representation of a hash value.
@usableFromInline
@frozen
internal struct _Hash {
  @usableFromInline
  internal var value: UInt

  @inlinable
  internal init(_ key: some Hashable) {
    let hashValue = key._rawHashValue(seed: 0)
    self.value = UInt(bitPattern: hashValue)
  }

  @inlinable
  internal init(_value: UInt) {
    self.value = _value
  }
}

extension _Hash: Equatable {
  @inlinable @inline(__always)
  internal static func ==(left: Self, right: Self) -> Bool {
    left.value == right.value
  }
}

extension _Hash: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    // Print hash values in radix 32 & reversed, so that the path in the hash
    // tree is readily visible.
    let p = String(value, radix: _Bitmap.capacity, uppercase: true)
    let c = _HashLevel.limit
    let path = String(repeating: "0", count: Swift.max(0, c - p.count)) + p
    return String(path.reversed())
  }
}


extension _Hash {
  @inlinable @inline(__always)
  internal static var bitWidth: Int { UInt.bitWidth }
}

extension _Hash {
  @inlinable
  internal subscript(_ level: _HashLevel) -> _Bucket {
    get {
      assert(!level.isAtBottom)
      return _Bucket((value &>> level.shift) & _Bucket.bitMask)
    }
    set {
      let mask = _Bucket.bitMask &<< level.shift
      self.value &= ~mask
      self.value |= newValue.value &<< level.shift
    }
  }
}

extension _Hash {
  @inlinable
  internal static var emptyPath: _Hash {
    _Hash(_value: 0)
  }

  @inlinable
  internal func appending(_ bucket: _Bucket, at level: _HashLevel) -> Self {
    assert(value >> level.shift == 0)
    var copy = self
    copy[level] = bucket
    return copy
  }

  @inlinable
  internal func isEqual(to other: _Hash, upTo level: _HashLevel) -> Bool {
    if level.isAtRoot { return true }
    if level.isAtBottom { return self == other }
    let s = UInt(UInt.bitWidth) - level.shift
    let v1 = self.value &<< s
    let v2 = self.value &<< s
    return v1 == v2
  }
}
