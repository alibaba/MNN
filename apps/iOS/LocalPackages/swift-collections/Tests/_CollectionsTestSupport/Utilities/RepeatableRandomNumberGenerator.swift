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

public struct RepeatableRandomNumberGenerator: RandomNumberGenerator {
  public static var globalSeed: Int {
#if COLLECTIONS_RANDOMIZED_TESTING
    42.hashValue
#else
    0
#endif
  }

  // This uses the same linear congruential generator as rand48.
  // FIXME: Replace with something better.
  internal static let _m: UInt64 = 1 << 48
  internal static let _a: UInt64 = 25214903917
  internal static let _c: UInt64 = 11

  internal var _state: UInt64

  public init(seed: Int) {
    self.init(seed: UInt64(truncatingIfNeeded: seed))
  }

  public init(seed: UInt64) {
    // Perturb the seed a little so that the sequence doesn't start with a
    // zero value in the common case of seed == 0. (Using a zero seed is a
    // rather silly thing to do, but it's the easy thing.)
    let global = UInt64(truncatingIfNeeded: Self.globalSeed)
    _state = seed ^ global ^ 0x536f52616e646f6d // "SoRandom"
  }

  private mutating func _next() -> UInt64 {
    _state = (Self._a &* _state &+ Self._c) & (Self._m - 1)
    return _state &>> 16
  }

  public mutating func next() -> UInt64 {
    return (_next() &<< 32) | _next()
  }
}
