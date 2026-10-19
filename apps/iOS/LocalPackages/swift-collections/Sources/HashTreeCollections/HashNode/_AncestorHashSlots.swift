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

/// A collection of slot values logically addressing a particular node in a
/// hash tree. The collection is (logically) extended with zero slots up to
/// the maximum depth of the tree -- to unambiguously address a single node,
/// this therefore needs to be augmented with a `_HashLevel` value.
///
/// This construct can only be used to identify a particular node in the tree;
/// it does not necessarily have room to include an item offset in the addressed
/// node. (See `_Path` if you need to address a particular item.)
@usableFromInline
@frozen
struct _AncestorHashSlots {
  @usableFromInline
  internal var path: UInt

  @inlinable @inline(__always)
  internal init(_ path: UInt) {
    self.path = path
  }
}

extension _AncestorHashSlots: Equatable {
  @inlinable @inline(__always)
  internal static func ==(left: Self, right: Self) -> Bool {
    left.path == right.path
  }
}

extension _AncestorHashSlots {
  @inlinable @inline(__always)
  internal static var empty: Self { Self(0) }
}

extension _AncestorHashSlots {
  /// Return or set the slot value at the specified level.
  /// If this is used to mutate the collection, then the original value
  /// on the given level must be zero.
  @inlinable @inline(__always)
  internal subscript(_ level: _HashLevel) -> _HashSlot {
    get {
      assert(level.shift < UInt.bitWidth)
      return _HashSlot((path &>> level.shift) & _Bucket.bitMask)
    }
    set {
      assert(newValue._value < _Bitmap.capacity)
      assert(self[level] == .zero)
      path |= (UInt(truncatingIfNeeded: newValue._value) &<< level.shift)
    }
  }

  @inlinable @inline(__always)
  internal func appending(_ slot: _HashSlot, at level: _HashLevel) -> Self {
    var result = self
    result[level] = slot
    return result
  }

  /// Clear the slot at the specified level, by setting it to zero.
  @inlinable
  internal mutating func clear(_ level: _HashLevel) {
    guard level.shift < UInt.bitWidth else { return }
    path &= ~(_Bucket.bitMask &<< level.shift)
  }

  /// Clear all slots at or below the specified level, by setting them to zero.
  @inlinable
  internal mutating func clear(atOrBelow level: _HashLevel) {
    guard level.shift < UInt.bitWidth else { return }
    path &= ~(UInt.max &<< level.shift)
  }

  /// Truncate this path to the specified level.
  /// Slots at or beyond the specified level are cleared.
  @inlinable
  internal func truncating(to level: _HashLevel) -> _AncestorHashSlots {
    assert(level.shift <= UInt.bitWidth)
    guard level.shift < UInt.bitWidth else { return self }
    return _AncestorHashSlots(path & ((1 &<< level.shift) &- 1))
  }

  /// Returns true if this path contains non-zero slots at or beyond the
  /// specified level, otherwise returns false.
  @inlinable
  internal func hasDataBelow(_ level: _HashLevel) -> Bool {
    guard level.shift < UInt.bitWidth else { return false }
    return (path &>> level.shift) != 0
  }

  /// Compares this path to `other` up to but not including the specified level.
  /// Returns true if the path prefixes compare equal, otherwise returns false.
  @inlinable
  internal func isEqual(to other: Self, upTo level: _HashLevel) -> Bool {
    if level.isAtRoot { return true }
    if level.isAtBottom { return self == other }
    let s = UInt(UInt.bitWidth) - level.shift
    let v1 = self.path &<< s
    let v2 = other.path &<< s
    return v1 == v2
  }
}

