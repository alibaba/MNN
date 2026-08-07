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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension UInt {
  /// Returns the position of the `n`th set bit in `self`.
  ///
  /// - Parameter n: The  to retrieve. This value is
  ///    decremented by the number of items found in this `self` towards the
  ///    value we're looking for. (If the function returns non-nil, then `n`
  ///    is set to `0` on return.)
  /// - Returns: If this integer contains enough set bits to satisfy the
  ///    request, then this function returns the position of the bit found.
  ///    Otherwise it returns nil.
  internal func _rank(ofBit n: inout UInt) -> UInt? {
    let c = self.nonzeroBitCount
    guard n < c else {
      n &-= UInt(bitPattern: c)
      return nil
    }
    let m = Int(bitPattern: n)
    n = 0
    return _bit(ranked: m)!
  }
}
