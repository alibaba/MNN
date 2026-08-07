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
  /// Create and return a new bit array consisting of `count` random bits,
  /// using the system random number generator.
  ///
  /// - Parameter count: The number of random bits to generate.
  public static func randomBits(count: Int) -> BitArray {
    var rng = SystemRandomNumberGenerator()
    return randomBits(count: count, using: &rng)
  }

  /// Create and return a new bit array consisting of `count` random bits,
  /// using the given random number generator.
  ///
  /// - Parameter count: The number of random bits to generate.
  /// - Parameter rng: The random number generator to use.
  public static func randomBits(
    count: Int,
    using rng: inout some RandomNumberGenerator
  ) -> BitArray {
    precondition(count >= 0, "Invalid count")
    guard count > 0 else { return BitArray() }
    let (w, b) = _BitPosition(count).endSplit
    var words = (0 ... w).map { _ in _Word(rng.next() as UInt) }
    words[w].formIntersection(_Word(upTo: b))
    return BitArray(_storage: words, count: count)
  }
}
