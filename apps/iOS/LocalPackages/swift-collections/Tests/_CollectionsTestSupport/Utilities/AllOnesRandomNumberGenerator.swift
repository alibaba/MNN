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

/// A terrible random number generator that always returns values with all
/// bits set to true.
public struct AllOnesRandomNumberGenerator: RandomNumberGenerator {

  public init() {}

  public mutating func next() -> UInt64 {
    UInt64.max
  }
}
