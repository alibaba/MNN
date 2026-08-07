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

/// An internal protocol describing a summarizable entity that isn't a full `RopeElement`.
///
/// Used as an implementation detail to increase code reuse across internal nodes and leaf nodes.
/// (Ideally `Rope._Node` would just conform to the full `RopeElement` protocol on its own, but
/// while that's an obvious refactoring idea, it hasn't happened yet.)
@usableFromInline
internal protocol _RopeItem<Summary> {
  associatedtype Summary: RopeSummary

  var summary: Summary { get }
}

extension Sequence where Element: _RopeItem {
  @inlinable
  internal func _sum() -> Element.Summary {
    self.reduce(into: .zero) { $0.add($1.summary) }
  }
}

extension Rope: _RopeItem {
  public typealias Summary = Element.Summary

  @inlinable
  public var summary: Summary {
    guard _root != nil else { return .zero }
    return root.summary
  }
}

extension Rope {
  /// A trivial wrapper around a rope's Element type, giving it `_RopeItem` conformance without
  /// having to make the protocol public.
  @usableFromInline
  @frozen // Not really! This module isn't ABI stable.
  internal struct _Item {
    @usableFromInline internal var value: Element

    @inlinable
    internal init(_ value: Element) { self.value = value }
  }
}

extension Rope._Item: _RopeItem {
  @usableFromInline internal typealias Summary = Rope.Summary

  @inlinable
  internal var summary: Summary { value.summary }
}

extension Rope._Item: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    "\(value)"
  }
}

extension Rope._Item {
  @inlinable
  internal var isEmpty: Bool { value.isEmpty }

  @inlinable
  internal var isUndersized: Bool { value.isUndersized }

  @inlinable
  internal mutating func rebalance(nextNeighbor right: inout Self) -> Bool {
    value.rebalance(nextNeighbor: &right.value)
  }

  @inlinable
  internal mutating func rebalance(prevNeighbor left: inout Self) -> Bool {
    value.rebalance(prevNeighbor: &left.value)
  }

  @usableFromInline internal typealias Index = Element.Index

  @inlinable
  internal mutating func split(at index: Index) -> Self {
    Self(self.value.split(at: index))
  }
}
