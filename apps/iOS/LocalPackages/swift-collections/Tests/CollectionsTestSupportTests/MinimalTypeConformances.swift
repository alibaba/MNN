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

final class MinimalTypeTests: CollectionTestCase {
  func testMinimalSequence() {
    withEvery(
      "behavior",
      in: [
        UnderestimatedCountBehavior.precise,
        UnderestimatedCountBehavior.half,
        UnderestimatedCountBehavior.value(0)
      ]
    ) { behavior in
      withEvery("isContiguous", in: [false, true]) { isContiguous in
        func make() -> MinimalSequence<Int> {
          MinimalSequence(
            elements: 0 ..< 50,
            underestimatedCount: behavior,
            isContiguous: isContiguous)
        }
        checkSequence(make, expectedContents: 0 ..< 50)
      }
    }
  }

  func testMinimalCollection() {
    checkCollection(MinimalCollection(0 ..< 50), expectedContents: 0 ..< 50)
  }

  func testMinimalBidirectionalCollection() {
    checkBidirectionalCollection(MinimalBidirectionalCollection(0 ..< 50), expectedContents: 0 ..< 50)
  }
}
