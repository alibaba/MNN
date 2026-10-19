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

extension BitArray: LosslessStringConvertible {
  /// Initializes a new bit array from a string representation.
  ///
  /// The string given is interpreted as if it was the binary representation
  /// of an integer value, consisting of the digits `0` and `1`, with the
  /// highest digits appearing first.
  ///
  ///     BitArray("") // []
  ///     BitArray("001") // [true, false, false]
  ///     BitArray("1110") // [false, true, true, true]
  ///     BitArray("42") // nil
  ///     BitArray("Foo") // nil
  ///
  /// To accept the display format used by `description`, the digits in the
  /// input string are also allowed to be surrounded by a single pair of ASCII
  /// angle brackets:
  ///
  ///     let bits: BitArray = [false, false, true, true]
  ///     let string = bits.description // "<1100>"
  ///     let sameBits = BitArray(string)! // [false, false, true, true]
  ///
  public init?(_ description: String) {
    var digits = description[...]
    if digits.utf8.first == ._asciiLT, digits.utf8.last == ._asciiGT {
      digits = digits.dropFirst().dropLast()
    }
    let bits: BitArray? = digits.utf8.withContiguousStorageIfAvailable { buffer in
      Self(_utf8Digits: buffer)
    } ?? Self(_utf8Digits: description.utf8)
    guard let bits = bits else {
      return nil
    }
    self = bits
  }

  internal init?(_utf8Digits utf8: some Collection<UInt8>) {
    let c = utf8.count
    self.init(repeating: false, count: c)
    var i = c &- 1
    let success = _update { handle in
      for byte in utf8 {
        if byte == ._ascii1 {
          handle.set(at: i)
        } else {
          guard byte == ._ascii0 else { return false }
        }
        i &-= 1
      }
      return true
    }
    guard success else { return nil }
  }
}
