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

extension Range where Bound: FixedWidthInteger {
  @inlinable
  internal func _clampedToUInt() -> Range<UInt> {
    if upperBound <= 0 {
      return Range<UInt>(uncheckedBounds: (0, 0))
    }
    if lowerBound >= UInt.max {
      return Range<UInt>(uncheckedBounds: (UInt.max, UInt.max))
    }
    let lower = lowerBound < 0 ? 0 : UInt(lowerBound)
    let upper = upperBound > UInt.max ? UInt.max : UInt(upperBound)
    return Range<UInt>(uncheckedBounds: (lower, upper))
  }

  @inlinable
  internal func _toUInt() -> Range<UInt>? {
    guard
      let lower = UInt(exactly: lowerBound),
      let upper = UInt(exactly: upperBound)
    else {
      return nil
    }
    return Range<UInt>(uncheckedBounds: (lower: lower, upper: upper))
  }
}
