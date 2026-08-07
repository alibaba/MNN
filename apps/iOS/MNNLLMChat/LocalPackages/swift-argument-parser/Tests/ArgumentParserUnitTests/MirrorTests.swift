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

final class MirrorTests: XCTestCase {}

extension MirrorTests {
  private struct Foo {
    let foo: String?
    let bar: String
    let baz: String!
  }
  func testRealValue() {
    func checkChildValue(_ child: Mirror.Child, expectedString: String?) {
      if let expectedString = expectedString {
        guard let stringValue = child.value as? String else {
          XCTFail("child.value is not a String type")
          return
        }
        XCTAssertEqual(stringValue, expectedString)
      } else {
        XCTAssertNil(nilOrValue(child.value))
        // This is why we use `unwrapedOptionalValue` for optionality checks
        // Even though the `value` is `nil` this returns `false`
        XCTAssertFalse(child.value as Any? == nil)
      }
    }
    func performTest(foo: String?, baz: String!) {
      let fooChild = Foo(foo: foo, bar: "foobar", baz: baz)
      Mirror(reflecting: fooChild).children.forEach { child in
        switch child.label {
        case "foo":
          checkChildValue(child, expectedString: foo)
        case "bar":
          checkChildValue(child, expectedString: "foobar")
        case "baz":
          checkChildValue(child, expectedString: baz)
        default:
          XCTFail("Unexpected child")
        }
      }
    }
    
    performTest(foo: "foo", baz: "baz")
    performTest(foo: "foo", baz: nil)
    performTest(foo: nil, baz: "baz")
    performTest(foo: nil, baz: nil)
  }
}
