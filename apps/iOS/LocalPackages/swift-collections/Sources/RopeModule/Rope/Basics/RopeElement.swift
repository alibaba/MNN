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

/// The element type of a rope. Rope elements are expected to be container types
/// of their own, with logical positions within them addressed by an `Index`
/// type, similar to `Collection` indices.
///
/// However, rope elements aren't required conform to `Collection`. In fact,
/// they often support multiple different ways to interpret/project their
/// contents, similar to `String`'s views. In some cases, they may just be
/// individual, undivisable items of varying sizes -- although it's usually
/// a better to use a simple fixed-size collection type instead.
///
/// Each such projection may come with a different idea for how large a rope
/// element is -- this is modeled by the `RopeSummary` and `RopeMetric`
/// protocols. The `summary` property returns the size of the element as an
/// additive value, which can be projected to integer sizes using one of the
/// associated rope metrics.
public protocol RopeElement {
  /// The commutative group that is used to augment the tree.
  associatedtype Summary: RopeSummary

  /// A type whose values address a particular pieces of content in this rope
  /// element.
  associatedtype Index: Comparable

  /// Returns the summary of `self`.
  var summary: Summary { get }

  var isEmpty: Bool { get }
  var isUndersized: Bool { get }

  /// Check the consistency of `self`.
  func invariantCheck()

  /// Rebalance contents between `self` and its next neighbor `right`,
  /// eliminating an `isUndersized` condition on one of the inputs, if possible.
  ///
  /// On return, `self` is expected to be non-empty and well-sized.
  ///
  /// - Parameter right: The element immediately following `self` in some rope.
  /// - Precondition: Either `self` or `right` must be undersized.
  /// - Returns: A boolean value indicating whether `right` has become empty.
  mutating func rebalance(nextNeighbor right: inout Self) -> Bool

  /// Rebalance contents between `self` and its previous neighbor `left`,
  /// eliminating an `isUndersized` condition on one of the inputs, if possible.
  ///
  /// On return, `self` is expected to be non-empty and well-sized.
  ///
  /// - Parameter left: The element immediately preceding `self` in some rope.
  /// - Precondition: Either `left` or `self` must be undersized.
  /// - Returns: A boolean value indicating whether `left` has become empty.
  mutating func rebalance(prevNeighbor left: inout Self) -> Bool

  /// Split `self` into two pieces at the specified index, keeping contents
  /// up to `index` in `self`, and moving the rest of it into a new item.
  mutating func split(at index: Index) -> Self
}

extension RopeElement {
  @inlinable
  public mutating func rebalance(prevNeighbor left: inout Self) -> Bool {
    guard left.rebalance(nextNeighbor: &self) else { return false }
    swap(&self, &left)
    return true
  }
}

