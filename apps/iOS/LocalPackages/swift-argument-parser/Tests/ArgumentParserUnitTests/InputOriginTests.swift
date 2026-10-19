//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import XCTest
@testable import ArgumentParser

final class InputOriginTests: XCTestCase {}

extension InputOriginTests {
  func testIsDefaultValue() {
    func Assert(elements: [InputOrigin.Element], expectedIsDefaultValue: Bool) {
      let inputOrigin = InputOrigin(elements: elements)
      if expectedIsDefaultValue {
        XCTAssertTrue(inputOrigin.isDefaultValue)
      } else {
        XCTAssertFalse(inputOrigin.isDefaultValue)
      }
    }
    
    Assert(elements: [], expectedIsDefaultValue: false)
    Assert(elements: [.defaultValue], expectedIsDefaultValue: true)
    Assert(elements: [.argumentIndex(SplitArguments.Index(inputIndex: 1))], expectedIsDefaultValue: false)
    Assert(elements: [.defaultValue, .argumentIndex(SplitArguments.Index(inputIndex: 1))], expectedIsDefaultValue: false)
  }
}
