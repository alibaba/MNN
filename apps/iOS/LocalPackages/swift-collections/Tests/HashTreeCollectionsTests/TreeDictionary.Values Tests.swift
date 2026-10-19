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
#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import _CollectionsTestSupport
import HashTreeCollections
#endif

class TreeDictionaryValuesTests: CollectionTestCase {
  func test_BidirectionalCollection_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (d, ref) = tracker.shareableDictionary(for: fixture)
        let v = ref.map { $0.value }
        checkCollection(d.values, expectedContents: v, by: ==)
        _checkBidirectionalCollection_indexOffsetBy(
          d.values, expectedContents: v, by: ==)
      }
    }
  }

  func test_descriptions() {
    let d: TreeDictionary = [
      "a": 1,
      "b": 2
    ]

    if d.first!.key == "a" {
      expectEqual(d.values.description, "[1, 2]")
    } else {
      expectEqual(d.values.description, "[2, 1]")
    }
  }
}
