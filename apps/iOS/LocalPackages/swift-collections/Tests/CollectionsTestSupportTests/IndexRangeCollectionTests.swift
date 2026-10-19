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
#if !COLLECTIONS_SINGLE_MODULE
import _CollectionsTestSupport
#endif

final class IndexRangeCollectionTests: CollectionTestCase {
  func testCollection() {
    withEvery("b", in: [0, 1]) { b in
      withEvery("c", in: 0 ..< 3) { c in
        let expected = (b ... b + c).flatMap { end in
          (b ... end).lazy.map { start in start ..< end }
        }
        let actual = IndexRangeCollection(bounds: b ..< b + c)
        checkBidirectionalCollection(actual, expectedContents: expected)
      }
    }
  }
}
