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

extension BitSet {
  /// An opaque type that represents a position in a bit set.
  ///
  /// The elements of a bit set can be addressed simply by their value,
  /// so the `Index` type could be defined to be `Int`, the same as `Element`.
  /// However, `BitSet` uses an opaque wrapper instead, to prevent confusion:
  /// it would otherwise be all too easy to accidentally use `i + 1` instead of
  /// calling `index(after: i)`, and ending up with an invalid index.
  @frozen
  public struct Index {
    @usableFromInline
    var _value: UInt

    @inlinable
    internal init(_value: UInt) {
      self._value = _value
    }

    @inlinable
    internal init(_position: _UnsafeHandle.Index) {
      self._value = _position.value
    }

    @inline(__always)
    internal var _position: _UnsafeHandle.Index {
      _UnsafeHandle.Index(_value)
    }
  }
}

extension BitSet.Index: Sendable {}

extension BitSet.Index: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    "\(_value)"
  }
}

extension BitSet.Index: CustomDebugStringConvertible {
  // A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension BitSet.Index: Equatable {
  /// Returns a Boolean value indicating whether two values are equal.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public static func ==(left: Self, right: Self) -> Bool {
    left._value == right._value
  }
}

extension BitSet.Index: Comparable {
  /// Returns a Boolean value indicating whether the first value is ordered
  /// before the second.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public static func < (left: Self, right: Self) -> Bool {
    left._value < right._value
  }
}

extension BitSet.Index: Hashable {
  /// Hashes the essential components of this value by feeding them to the given hasher.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func hash(into hasher: inout Hasher) {
    hasher.combine(_value)
  }
}
