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

import XCTest
#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
@_spi(Testing) import OrderedCollections
import _CollectionsTestSupport
#endif

// Note: This cannot really work unless `UnorderedView` becomes a Collection.
// extension OrderedSet.UnorderedView: SetAPIChecker {}

class OrderedSetUnorderedViewTests: CollectionTestCase {
  func test_unordered_insert() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("dupes", in: 1 ... 3) { dupes in
        withLifetimeTracking { tracker in
          let input = (0 ..< dupes * count)
            .map { tracker.instance(for: $0 / dupes) }
            .shuffled()
          let reference: [Int: LifetimeTracked<Int>] =
            .init(input.lazy.map { ($0.payload, $0) },
                  uniquingKeysWith: { a, b in a })
          var set = OrderedSet<LifetimeTracked<Int>>()
          withEvery("offset", in: input.indices) { offset in
            let item = input[offset]
            let ref = reference[item.payload]
            let (inserted, member) = set.unordered.insert(item)
            expectEqual(inserted, ref === item)
            expectEqual(member, item)
            expectIdentical(member, ref)
            expectTrue(set.contains(item))
          }
        }
      }
    }
    // Check CoW copying behavior
    do {
      var set = OrderedSet<Int>(0 ..< 30)
      let copy = set
      expectTrue(set.unordered.insert(30).inserted)
      expectTrue(set.contains(30))
      expectFalse(copy.contains(30))
    }
  }

  func test_unordered_update() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("dupes", in: 1 ... 3) { dupes in
        withLifetimeTracking { tracker in
          let input = (0 ..< dupes * count)
            .map { tracker.instance(for: $0 / dupes) }
            .shuffled()
          var reference: [Int: LifetimeTracked<Int>] = [:]
          var set = OrderedSet<LifetimeTracked<Int>>()
          withEvery("offset", in: input.indices) { offset in
            let item = input[offset]
            let old = set.unordered.update(with: item)
            expectIdentical(old, reference[item.payload])
            reference[item.payload] = item
            expectTrue(set.contains(item))
          }
        }
      }
    }
    // Check CoW copying behavior
    do {
      var set = OrderedSet<Int>(0 ..< 30)
      let copy = set
      expectNil(set.unordered.update(with: 30))
      expectTrue(set.contains(30))
      expectFalse(copy.contains(30))
    }
  }
}
