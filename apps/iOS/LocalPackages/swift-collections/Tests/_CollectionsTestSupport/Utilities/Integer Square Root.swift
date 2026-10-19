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

extension FixedWidthInteger {
  internal func _squareRoot() -> Self {
    // Newton's method
    precondition(self >= 0)
    guard self != 0 else { return 0 }
    var x: Self = 1 &<< ((self.bitWidth + 1) / 2)
    var y: Self = 0
    while true {
      y = (self / x + x) &>> 1
      if x == y || x == y - 1 { break }
      x = y
    }
    return x
  }
}
