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

import XCTest
#if !COLLECTIONS_SINGLE_MODULE
import _CollectionsTestSupport
#endif

class CombinatoricsTests: CollectionTestCase {
  func testEverySubset_smoke() {
    func collectSubsets(of set: [Int]) -> Set<[Int]> {
      var result: Set<[Int]> = []
      withEverySubset("subset", of: set) { subset in
        let r = result.insert(subset)
        expectTrue(r.inserted)
      }
      return result
    }

    expectEqual(collectSubsets(of: []), [[]])
    expectEqual(collectSubsets(of: [0]), [[], [0]])
    expectEqual(collectSubsets(of: [0, 1]), [[], [0], [1], [0, 1]])
    expectEqual(
      collectSubsets(of: [0, 1, 2]),
      [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
  }

  func testEveryPermutation_smoke() {
    func collectPermutations(of items: [Int]) -> Set<[Int]> {
      var result: Set<[Int]> = []
      withEveryPermutation("permutation", of: items) { permutation in
        let r = result.insert(permutation)
        expectTrue(r.inserted)
      }
      return result
    }

    expectEqual(collectPermutations(of: []), [[]])
    expectEqual(collectPermutations(of: [0]), [[0]])
    expectEqual(collectPermutations(of: [0, 1]), [[0, 1], [1, 0]])
    expectEqual(
      collectPermutations(of: [0, 1, 2]),
      [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
  }
}
