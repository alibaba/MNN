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

extension OrderedDictionary {
  /// Accesses the element at the specified index.
  ///
  /// - Parameter offset: The offset of the element to access, measured from
  ///   the start of the collection. `offset` must be greater than or equal to
  ///   `0` and less than `count`.
  ///
  /// - Complexity: O(1)
  @available(*, unavailable, // deprecated in 0.0.6, obsoleted in 1.0.0
    message: "Please use `elements[offset]`")
  @inlinable
  @inline(__always)
  public subscript(offset offset: Int) -> Element {
    fatalError()
  }
}

extension OrderedDictionary {
  @available(*, unavailable, // deprecated in 0.0.6, obsoleted in 1.0.0
    renamed: "updateValue(forKey:default:with:)")
  @inlinable
  public mutating func modifyValue<R>(
    forKey key: Key,
    default defaultValue: @autoclosure () -> Value,
    _ body: (inout Value) throws -> R
  ) rethrows -> R {
    fatalError()
  }

  @available(*, unavailable, // deprecated in 0.0.6, obsoleted in 1.0.0
    renamed: "updateValue(forKey:insertingDefault:at:with:)")
  @inlinable
  public mutating func modifyValue<R>(
    forKey key: Key,
    insertingDefault defaultValue: @autoclosure () -> Value,
    at index: Int,
    _ body: (inout Value) throws -> R
  ) rethrows -> R {
    fatalError()
  }
}
