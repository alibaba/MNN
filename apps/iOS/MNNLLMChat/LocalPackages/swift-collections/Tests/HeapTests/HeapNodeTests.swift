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

#if DEBUG // These unit tests need access to HeapModule internals
import XCTest
#if COLLECTIONS_SINGLE_MODULE
@testable import Collections
#else
@testable import HeapModule
#endif

class HeapNodeTests: XCTestCase {
  func test_levelCalculation() {
    // Check alternating min and max levels in the heap
    var isMin = true
    for exp in 0...12 {
      // Check [2^exp, 2^(exp + 1))
      for offset in Int(pow(2, Double(exp)) - 1)..<Int(pow(2, Double(exp + 1)) - 1) {
        let node = _HeapNode(offset: offset)
        XCTAssertEqual(node.isMinLevel, isMin)
      }
      isMin.toggle()
    }
  }
}
#endif // DEBUG
