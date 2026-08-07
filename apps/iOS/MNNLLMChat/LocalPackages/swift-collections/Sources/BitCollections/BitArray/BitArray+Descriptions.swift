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

extension UInt8 {
  @inline(__always)
  internal static var _ascii0: Self { 48 }

  @inline(__always)
  internal static var _ascii1: Self { 49 }

  @inline(__always)
  internal static var _asciiLT: Self { 60 }

  @inline(__always)
  internal static var _asciiGT: Self { 62 }
}

extension BitArray: CustomStringConvertible {
  /// A textual representation of this instance.
  /// Bit arrays print themselves as a string of binary bits, with the
  /// highest-indexed elements appearing first, as in the binary representation
  /// of integers. The digits are surrounded by angle brackets, so that the
  /// notation is non-ambigous even for empty bit arrays:
  ///
  ///     let bits: BitArray = [false, false, false, true, true]
  ///     print(bits) // "<11000>"
  ///
  ///     let empty = BitArray()
  ///     print(empty) // "<>"
  public var description: String {
    _bitString
  }
}

extension BitArray: CustomDebugStringConvertible {
  /// A textual representation of this instance.
  /// Bit arrays print themselves as a string of binary bits, with the
  /// highest-indexed elements appearing first, as in the binary representation
  /// of integers:
  ///
  ///     let bits: BitArray = [false, false, false, true, true]
  ///     print(bits) // "<11000>"
  ///
  /// The digits are surrounded by angle brackets to serve as visual delimiters.
  /// (So that empty bit arrays still have a non-empty description.)
  public var debugDescription: String {
    description
  }
}

extension BitArray {
  internal var _bitString: String {
    guard !isEmpty else { return "<>" }
    var result: String
    if #available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *) {
      result = String(unsafeUninitializedCapacity: self.count + 2) { target in
        target.initializeElement(at: count + 1, to: ._asciiGT)
        var i = count
        for v in self {
          target.initializeElement(at: i, to: v ? ._ascii1 : ._ascii0)
          i &-= 1
        }
        assert(i == 0)
        target.initializeElement(at: 0, to: ._asciiLT)
        return count + 2
      }
    } else {
      result = "<"
      result.reserveCapacity(self.count + 2)
      for v in self.reversed() {
        result.append(v ? "1" : "0")
      }
      result.append(">")
    }
    return result
  }
}
