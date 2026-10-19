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

/// A commutative group that is used to augment a tree, enabling quick lookup operations.
public protocol RopeSummary: Equatable, Sendable {
  static var maxNodeSize: Int { get }
  static var nodeSizeBitWidth: Int { get }

  /// The identity element of the group.
  static var zero: Self { get }

  /// Returns a Boolean value that indicates whether `self` is the identity element.
  var isZero: Bool { get }

  /// A commutative and associative operation that combines two instances.
  ///
  /// (As is usually the case, this operation is not necessarily closed over `Self` in practice --
  /// e.g., some results may not be representable.)
  mutating func add(_ other: Self)

  /// A (potentially partial) subtraction function that undoes a previous combination of the given
  /// element to `self`.
  ///
  /// The inverse of any instance can be calculated by subtracting it from the `zero` instance.
  /// (However, conforming types are free to require that `subtract` only be called on values
  /// that "include" the given `other`.)
  mutating func subtract(_ other: Self)
}

extension RopeSummary {
  @inlinable @inline(__always)
  public static var nodeSizeBitWidth: Int {
    Int.bitWidth - maxNodeSize.leadingZeroBitCount
  }

  @inlinable @inline(__always)
  public static var minNodeSize: Int { (maxNodeSize + 1) / 2 }
}

extension RopeSummary {
  @inlinable
  public func adding(_ other: Self) -> Self {
    var c = self
    c.add(other)
    return c
  }

  @inlinable
  public func subtracting(_ other: Self) -> Self {
    var c = self
    c.subtract(other)
    return c
  }
}
