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

extension BitArray {
  /// Shift the bits in this array by the specified number of places to the
  /// left (towards the end of the array), by inserting `amount` false values
  /// at the beginning.
  ///
  /// If `amount` is negative, this is equivalent to shifting `-amount`
  /// places to the right.
  ///
  ///     var bits: BitArray = "1110110"
  ///     bits.maskingShiftLeft(by: 2)
  ///     // bits is now 111011000
  ///     bits.maskingShiftLeft(by: -4)
  ///     // bits is now 11101
  ///     bits.maskingShiftLeft(by: 8)
  ///     // bits is now 111010000000
  public mutating func resizingShiftLeft(by amount: Int) {
    guard amount != 0 else { return }
    if amount > 0 {
      _resizingShiftLeft(by: amount)
    } else {
      _resizingShiftRight(by: -amount)
    }
  }

  /// Shift the bits in this array by the specified number of places to the
  /// right (towards the start of the array), by removing `amount` existing
  /// values from the front of the array.
  ///
  /// If `amount` is negative, then this is equivalent to shifting `-amount`
  /// places to the left. If amount is greater than or equal to `count`,
  /// then the resulting bit array will be empty.
  ///
  ///     var bits: BitArray = "1110110"
  ///     bits.maskingShiftRight(by: 2)
  ///     // bits is now 11101
  ///     bits.maskingShiftRight(by: -4)
  ///     // bits is now 111010000
  ///     bits.maskingShiftRight(by: 10)
  ///     // bits is now empty
  ///
  /// If `amount` is between 0 and `count`, then this has the same effect as
  /// `removeFirst(amount)`.
  public mutating func resizingShiftRight(by amount: Int) {
    guard amount != 0 else { return }
    if amount > 0 {
      _resizingShiftRight(by: amount)
    } else {
      _resizingShiftLeft(by: -amount)
    }
  }

  internal mutating func _resizingShiftLeft(by amount: Int) {
    assert(amount > 0)
    _extend(by: amount, with: false)
    maskingShiftLeft(by: amount)
  }

  internal mutating func _resizingShiftRight(by amount: Int) {
    assert(amount > 0)
    guard amount < count else {
      self = .init()
      return
    }
    self._copy(from: Range(uncheckedBounds: (amount, count)), to: 0)
    self._removeLast(count &- amount)
  }
}

extension BitArray {
  // FIXME: Add maskingShiftRight(by:in:) and maskingShiftLeft(by:in:) for shifting slices.

  /// Shift the bits in this array by the specified number of places to the
  /// left (towards the end of the array), without changing
  /// its count.
  ///
  /// Values that are shifted off the array are discarded. Values that are
  /// shifted in are all set to false.
  ///
  /// If `amount` is negative, this is equivalent to shifting `-amount`
  /// places to the right. If `amount` is greater than or equal to `count`,
  /// then all values are set to false.
  ///
  ///     var bits: BitArray = "1110110"
  ///     bits.maskingShiftLeft(by: 2)
  ///     // bits is now 1011000
  ///     bits.maskingShiftLeft(by: -4)
  ///     // bits is now 0000101
  ///     bits.maskingShiftLeft(by: 8)
  ///     // bits is now 0000000
  public mutating func maskingShiftLeft(by amount: Int) {
    guard amount != 0 else { return }
    _update {
      if amount > 0 {
        $0._maskingShiftLeft(by: amount)
      } else {
        $0._maskingShiftRight(by: -amount)
      }
    }
  }

  /// Shift the bits in this array by the specified number of places to the
  /// right (towards the beginning of the array), without changing
  /// its count.
  ///
  /// Values that are shifted off the array are discarded. Values that are
  /// shifted in are all set to false.
  ///
  /// If `amount` is negative, this is equivalent to shifting `-amount`
  /// places to the left. If `amount` is greater than or equal to `count`,
  /// then all values are set to false.
  ///
  ///     var bits: BitArray = "1110110"
  ///     bits.maskingShiftRight(by: 2)
  ///     // bits is now 0011101
  ///     bits.maskingShiftRight(by: -3)
  ///     // bits is now 1101000
  ///     bits.maskingShiftRight(by: 8)
  ///     // bits is now 0000000
  public mutating func maskingShiftRight(by amount: Int) {
    guard amount != 0 else { return }
    _update {
      if amount > 0 {
        $0._maskingShiftRight(by: amount)
      } else {
        $0._maskingShiftLeft(by: -amount)
      }
    }
  }
}

extension BitArray._UnsafeHandle {
  internal mutating func _maskingShiftLeft(by amount: Int) {
    assert(amount > 0)
    let d = Swift.min(amount, self.count)
    if d == amount {
      let range = Range(uncheckedBounds: (0, self.count &- d))
      self.copy(from: range, to: d)
    }
    self.clear(in: Range(uncheckedBounds: (0, d)))
  }

  internal mutating func _maskingShiftRight(by amount: Int) {
    assert(amount > 0)
    let d = Swift.min(amount, self.count)
    if d == amount {
      let range = Range(uncheckedBounds: (d, self.count))
      self.copy(from: range, to: 0)
    }
    self.clear(in: Range(uncheckedBounds: (self.count &- d, self.count)))
  }
}
