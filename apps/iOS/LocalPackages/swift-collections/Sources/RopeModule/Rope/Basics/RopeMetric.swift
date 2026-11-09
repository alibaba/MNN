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

public protocol RopeMetric<Element>: Sendable {
  associatedtype Element: RopeElement

  /// Returns the size of a summarized rope element in this metric.
  func size(of summary: Element.Summary) -> Int

  /// Returns an index addressing the content at the given offset from
  /// the start of the specified rope element.
  ///
  /// - Parameter offset: An integer offset from the start of `element` in this
  ///     metric, not exceeding `size(of: element.summary)`.
  /// - Parameter element: An arbitrary rope element.
  /// - Returns: The index addressing the desired position in the input element.
  func index(at offset: Int, in element: Element) -> Element.Index
}

extension RopeMetric {
  @inlinable @inline(__always)
  internal func _nonnegativeSize(of summary: Element.Summary) -> Int {
    let r = size(of: summary)
    assert(r >= 0)
    return r
  }
}
