//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import XCTest
@testable import ArgumentParser

final class StringEditDistanceTests: XCTestCase {}

extension StringEditDistanceTests {
  func testStringEditDistance() {
    XCTAssertEqual("".editDistance(to: ""), 0)
    XCTAssertEqual("".editDistance(to: "foo"), 3)
    XCTAssertEqual("foo".editDistance(to: ""), 3)
    XCTAssertEqual("foo".editDistance(to: "bar"), 3)
    XCTAssertEqual("bar".editDistance(to: "foo"), 3)
    XCTAssertEqual("bar".editDistance(to: "baz"), 1)
    XCTAssertEqual("baz".editDistance(to: "bar"), 1)
    XCTAssertEqual("friend".editDistance(to: "fresh"), 3)
    XCTAssertEqual("friend".editDistance(to: "friend"), 0)
    XCTAssertEqual("friend".editDistance(to: "fried"), 1)
    XCTAssertEqual("friend".editDistance(to: "friendly"), 2)
    XCTAssertEqual("friendly".editDistance(to: "friend"), 2)
  }
}
