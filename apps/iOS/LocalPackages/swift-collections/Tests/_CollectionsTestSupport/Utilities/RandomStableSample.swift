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

// Note: This comes from swift-algorithms.

extension Collection {
  /// Randomly selects the specified number of elements from this collection,
  /// maintaining their relative order.
  ///
  /// - Parameters:
  ///   - k: The number of elements to randomly select.
  ///   - rng: The random number generator to use for the sampling.
  /// - Returns: An array of `k` random elements. If `k` is greater than this
  ///   collection's count, then this method returns the full collection.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  @inlinable
  public func randomStableSample<G: RandomNumberGenerator>(
    count k: Int, using rng: inout G
  ) -> [Element] {
    guard k > 0 else { return [] }
    
    var remainingCount = count
    guard k < remainingCount else { return Array(self) }
    
    var result: [Element] = []
    result.reserveCapacity(k)
    
    var i = startIndex
    var countToSelect = k
    while countToSelect > 0 {
      let r = Int.random(in: 0..<remainingCount, using: &rng)
      if r < countToSelect {
        result.append(self[i])
        countToSelect -= 1
      }

      formIndex(after: &i)
      remainingCount -= 1
    }
    
    return result
  }
  
  /// Randomly selects the specified number of elements from this collection,
  /// maintaining their relative order.
  ///
  /// This method is equivalent to calling `randomStableSample(k:using:)`,
  /// passing in the system's default random generator.
  ///
  /// - Parameter k: The number of elements to randomly select.
  /// - Returns: An array of `k` random elements. If `k` is greater than this
  ///   collection's count, then this method returns the full collection.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  @inlinable
  public func randomStableSample(count k: Int) -> [Element] {
    var g = SystemRandomNumberGenerator()
    return randomStableSample(count: k, using: &g)
  }
}
