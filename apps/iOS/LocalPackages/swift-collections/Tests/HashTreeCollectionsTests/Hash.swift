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

/// An abstract representation of a hash value.
struct Hash {
  var value: Int

  init(_ value: Int) {
    self.value = value
  }

  init(_ value: UInt) {
    self.value = Int(bitPattern: value)
  }
}


extension Hash {
  static var bucketBitWidth: Int { 5 }
  static var bucketCount: Int { 1 << bucketBitWidth }
  static var bitWidth: Int { UInt.bitWidth }
}

extension Hash: Equatable {
  static func ==(left: Self, right: Self) -> Bool {
    left.value == right.value
  }
}

extension Hash: Hashable {
  func hash(into hasher: inout Hasher) {
    hasher.combine(value)
  }
}

extension Hash: CustomStringConvertible {
  var description: String {
    // Print hash values in radix 32 & reversed, so that the path in the hash
    // tree is readily visible.
    let p = String(
      UInt(bitPattern: value),
      radix: Self.bucketCount,
      uppercase: true)
    #if false // The zeroes look overwhelmingly long in this context
    let c = (Self.bitWidth + Self.bucketBitWidth - 1) / Self.bucketBitWidth
    let path = String(repeating: "0", count: Swift.max(0, c - p.count)) + p
    return String(path.reversed())
    #else
    return String(p.reversed())
    #endif
  }
}

extension Hash: LosslessStringConvertible {
  init?(_ description: String) {
    let s = String(description.reversed())
    guard let hash = UInt(s, radix: 32) else { return nil }
    self.init(Int(bitPattern: hash))
  }
}

extension Hash: ExpressibleByIntegerLiteral {
  init(integerLiteral value: UInt) {
    self.init(value)
  }
}

extension Hash: ExpressibleByStringLiteral {
  init(stringLiteral value: String) {
    self.init(value)!
  }
}
