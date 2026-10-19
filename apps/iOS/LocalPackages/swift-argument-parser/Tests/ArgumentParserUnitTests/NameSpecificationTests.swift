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

final class NameSpecificationTests: XCTestCase {
}

extension NameSpecificationTests {
  func testFlagNames_withNoPrefix() {
    let key = InputKey(name: "index", parent: nil)
    
    XCTAssertEqual(FlagInversion.prefixedNo.enableDisableNamePair(for: key, name: .customLong("foo")).1, [.long("no-foo")])
    XCTAssertEqual(FlagInversion.prefixedNo.enableDisableNamePair(for: key, name: .customLong("foo-bar-baz")).1, [.long("no-foo-bar-baz")])
    XCTAssertEqual(FlagInversion.prefixedNo.enableDisableNamePair(for: key, name: .customLong("foo_bar_baz")).1, [.long("no_foo_bar_baz")])
    XCTAssertEqual(FlagInversion.prefixedNo.enableDisableNamePair(for: key, name: .customLong("fooBarBaz")).1, [.long("noFooBarBaz")])
  }
  
  func testFlagNames_withEnableDisablePrefix() {
    let key = InputKey(name: "index", parent: nil)
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .long).0, [.long("enable-index")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .long).1, [.long("disable-index")])
    
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo")).0, [.long("enable-foo")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo")).1, [.long("disable-foo")])
    
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo-bar-baz")).0, [.long("enable-foo-bar-baz")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo-bar-baz")).1, [.long("disable-foo-bar-baz")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo_bar_baz")).0, [.long("enable_foo_bar_baz")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("foo_bar_baz")).1, [.long("disable_foo_bar_baz")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("fooBarBaz")).0, [.long("enableFooBarBaz")])
    XCTAssertEqual(FlagInversion.prefixedEnableDisable.enableDisableNamePair(for: key, name: .customLong("fooBarBaz")).1, [.long("disableFooBarBaz")])
  }
}

fileprivate func Assert(nameSpecification: NameSpecification, key: String, parent: InputKey? = nil, makeNames expected: [Name], file: StaticString = #file, line: UInt = #line) {
  let names = nameSpecification.makeNames(InputKey(name: key, parent: parent))
  Assert(names: names, expected: expected, file: file, line: line)
}

fileprivate func Assert<N>(names: [N], expected: [N], file: StaticString = #file, line: UInt = #line) where N: Equatable {
  names.forEach {
    XCTAssert(expected.contains($0), "Unexpected name '\($0)'.", file: (file), line: line)
  }
  expected.forEach {
    XCTAssert(names.contains($0), "Missing name '\($0)'.", file: (file), line: line)
  }
}

extension NameSpecificationTests {
  func testMakeNames_short() {
    Assert(nameSpecification: .short, key: "foo", makeNames: [.short("f")])
  }
  
  func testMakeNames_Long() {
    Assert(nameSpecification: .long, key: "fooBarBaz", makeNames: [.long("foo-bar-baz")])
    Assert(nameSpecification: .long, key: "fooURLForBarBaz", makeNames: [.long("foo-url-for-bar-baz")])
  }
  
  func testMakeNames_customLong() {
    Assert(nameSpecification: .customLong("bar"), key: "foo", makeNames: [.long("bar")])
  }
  
  func testMakeNames_customShort() {
    Assert(nameSpecification: .customShort("v"), key: "foo", makeNames: [.short("v")])
  }
  
  func testMakeNames_customLongWithSingleDash() {
    Assert(nameSpecification: .customLong("baz", withSingleDash: true), key: "foo", makeNames: [.longWithSingleDash("baz")])
  }
}
