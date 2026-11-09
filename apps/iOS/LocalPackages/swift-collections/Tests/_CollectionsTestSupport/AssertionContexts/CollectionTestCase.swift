//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

import XCTest

open class CollectionTestCase: XCTestCase {
  internal var _context: TestContext?

  public var context: TestContext { _context! }

  open var isAvailable: Bool { true }

  #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
  open override func invokeTest() {
    guard isAvailable else {
      print("\(Self.self) unavailable; skipping")
      return
    }
    return super.invokeTest()
  }
  #endif

  open override func setUp() {
    super.setUp()
    _context = TestContext.pushNew()
  }

  open override func tearDown() {
    if let context = _context {
      TestContext.pop(context)
      _context = nil
    }
    super.tearDown()
  }
}
