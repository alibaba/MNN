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

extension BitArray {
  /// Set every bit of this array to `value` (`true` by default).
  ///
  /// - Parameter value: The Boolean value to which to set the array's elements.
  public mutating func fill(with value: Bool = true) {
    fill(in: Range(uncheckedBounds: (0, count)), with: value)
  }

  /// Set every bit of this array within the specified range to `value`
  /// (`true` by default).
  ///
  /// - Parameter range: The range whose elements to overwrite.
  /// - Parameter value: The Boolean value to which to set the array's elements.
  public mutating func fill(in range: Range<Int>, with value: Bool = true) {
    _update { handle in
      if value {
        handle.fill(in: range)
      } else {
        handle.clear(in: range)
      }
    }
  }

  /// Set every bit of this array within the specified range to `value`
  /// (`true` by default).
  ///
  /// - Parameter range: The range whose elements to overwrite.
  /// - Parameter value: The Boolean value to which to set the array's elements.
  public mutating func fill(
    in range: some RangeExpression<Int>,
    with value: Bool = true
  ) {
    fill(in: range.relative(to: self), with: value)
  }
}
