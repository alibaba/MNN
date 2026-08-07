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
import ArgumentParserTestHelpers

extension ArgumentParser.SplitArguments.InputIndex: Swift.ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Int) {
    self.init(rawValue: value)
  }
}

private func AssertIndexEqual(_ sut: SplitArguments, at index: Int, inputIndex: Int, subIndex: SplitArguments.SubIndex, file: StaticString = #file, line: UInt = #line) {
  guard index < sut.elements.endIndex else {
    XCTFail("Element index \(index) is out of range. sur only has \(sut.elements.count) elements.", file: (file), line: line)
    return
  }
  let splitIndex = sut.elements[index].index
  let expected = SplitArguments.Index(inputIndex: SplitArguments.InputIndex(rawValue: inputIndex), subIndex: subIndex)
  if splitIndex.inputIndex != expected.inputIndex {
    XCTFail("inputIndex does not match: \(splitIndex.inputIndex.rawValue) != \(expected.inputIndex.rawValue)", file: (file), line: line)
  }
  if splitIndex.subIndex != expected.subIndex {
    XCTFail("inputIndex does not match: \(splitIndex.subIndex) != \(expected.subIndex)", file: (file), line: line)
  }
}

private func AssertElementEqual(_ sut: SplitArguments, at index: Int, _ element: SplitArguments.Element.Value, file: StaticString = #file, line: UInt = #line) {
  guard index < sut.elements.endIndex else {
    XCTFail("Element index \(index) is out of range. sur only has \(sut.elements.count) elements.", file: (file), line: line)
    return
  }
  XCTAssertEqual(sut.elements[index].value, element, file: (file), line: line)
}

final class SplitArgumentTests: XCTestCase {
  func testEmpty() throws {
    let sut = try SplitArguments(arguments: [])
    XCTAssertEqual(sut.elements.count, 0)
    XCTAssertEqual(sut.originalInput.count, 0)
  }
  
  func testSingleValue() throws {
    let sut = try SplitArguments(arguments: ["abc"])
    
    XCTAssertEqual(sut.elements.count, 1)
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .value("abc"))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["abc"])
  }
  
  func testSingleLongOption() throws {
    let sut = try SplitArguments(arguments: ["--abc"])
    
    XCTAssertEqual(sut.elements.count, 1)
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("abc"))))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["--abc"])
  }
  
  func testSingleShortOption() throws {
    let sut = try SplitArguments(arguments: ["-a"])
    
    XCTAssertEqual(sut.elements.count, 1)
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.short("a"))))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["-a"])
  }
  
  func testSingleLongOptionWithValue() throws {
    let sut = try SplitArguments(arguments: ["--abc=def"])
    
    XCTAssertEqual(sut.elements.count, 1)
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.nameWithValue(.long("abc"), "def")))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["--abc=def"])
  }
  
  func testMultipleShortOptionsCombined() throws {
    let sut = try SplitArguments(arguments: ["-abc"])
    
    XCTAssertEqual(sut.elements.count, 4)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.longWithSingleDash("abc"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 0, subIndex: .sub(0))
    AssertElementEqual(sut, at: 1, .option(.name(.short("a"))))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 0, subIndex: .sub(1))
    AssertElementEqual(sut, at: 2, .option(.name(.short("b"))))
    
    AssertIndexEqual(sut, at: 3, inputIndex: 0, subIndex: .sub(2))
    AssertElementEqual(sut, at: 3, .option(.name(.short("c"))))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["-abc"])
  }
  
  func testSingleLongOptionWithValueAndSingleDash() throws {
    let sut = try SplitArguments(arguments: ["-abc=def"])
    
    XCTAssertEqual(sut.elements.count, 1)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.nameWithValue(.longWithSingleDash("abc"), "def")))
    
    XCTAssertEqual(sut.originalInput.count, 1)
    XCTAssertEqual(sut.originalInput, ["-abc=def"])
  }
}

extension SplitArgumentTests {
  func testMultipleValues() throws {
    let sut = try SplitArguments(arguments: ["abc", "x", "1234"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .value("abc"))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .value("x"))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .value("1234"))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["abc", "x", "1234"])
  }
  
  func testMultipleLongOptions() throws {
    let sut = try SplitArguments(arguments: ["--d", "--1", "--abc-def"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("d"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .option(.name(.long("1"))))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .option(.name(.long("abc-def"))))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["--d", "--1", "--abc-def"])
  }
  
  func testMultipleShortOptions() throws {
    let sut = try SplitArguments(arguments: ["-x", "-y", "-z"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.short("x"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .option(.name(.short("y"))))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .option(.name(.short("z"))))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["-x", "-y", "-z"])
  }
  
  func testMultipleShortOptionsCombined_2() throws {
    let sut = try SplitArguments(arguments: ["-bc", "-fv", "-a"])
    
    XCTAssertEqual(sut.elements.count, 7)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.longWithSingleDash("bc"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 0, subIndex: .sub(0))
    AssertElementEqual(sut, at: 1, .option(.name(.short("b"))))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 0, subIndex: .sub(1))
    AssertElementEqual(sut, at: 2, .option(.name(.short("c"))))
    
    AssertIndexEqual(sut, at: 3, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 3, .option(.name(.longWithSingleDash("fv"))))
    
    AssertIndexEqual(sut, at: 4, inputIndex: 1, subIndex: .sub(0))
    AssertElementEqual(sut, at: 4, .option(.name(.short("f"))))
    
    AssertIndexEqual(sut, at: 5, inputIndex: 1, subIndex: .sub(1))
    AssertElementEqual(sut, at: 5, .option(.name(.short("v"))))
    
    AssertIndexEqual(sut, at: 6, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 6, .option(.name(.short("a"))))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["-bc", "-fv", "-a"])
  }
}

extension SplitArgumentTests {
  func testMixed_1() throws {
    let sut = try SplitArguments(arguments: ["-x", "abc", "--foo", "1234", "-zz"])
    
    XCTAssertEqual(sut.elements.count, 7)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.short("x"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .value("abc"))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .option(.name(.long("foo"))))
    
    AssertIndexEqual(sut, at: 3, inputIndex: 3, subIndex: .complete)
    AssertElementEqual(sut, at: 3, .value("1234"))
    
    AssertIndexEqual(sut, at: 4, inputIndex: 4, subIndex: .complete)
    AssertElementEqual(sut, at: 4, .option(.name(.longWithSingleDash("zz"))))
    
    AssertIndexEqual(sut, at: 5, inputIndex: 4, subIndex: .sub(0))
    AssertElementEqual(sut, at: 5, .option(.name(.short("z"))))
    
    AssertIndexEqual(sut, at: 6, inputIndex: 4, subIndex: .sub(1))
    AssertElementEqual(sut, at: 6, .option(.name(.short("z"))))
    
    XCTAssertEqual(sut.originalInput.count, 5)
    XCTAssertEqual(sut.originalInput, ["-x", "abc", "--foo", "1234", "-zz"])
  }
  
  func testMixed_2() throws {
    let sut = try SplitArguments(arguments: ["1234", "-zz", "abc", "-x", "--foo"])
    
    XCTAssertEqual(sut.elements.count, 7)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .value("1234"))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .option(.name(.longWithSingleDash("zz"))))
    
    AssertIndexEqual(sut, at: 2, inputIndex: 1, subIndex: .sub(0))
    AssertElementEqual(sut, at: 2, .option(.name(.short("z"))))
    
    AssertIndexEqual(sut, at: 3, inputIndex: 1, subIndex: .sub(1))
    AssertElementEqual(sut, at: 3, .option(.name(.short("z"))))
    
    AssertIndexEqual(sut, at: 4, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 4, .value("abc"))
    
    AssertIndexEqual(sut, at: 5, inputIndex: 3, subIndex: .complete)
    AssertElementEqual(sut, at: 5, .option(.name(.short("x"))))
    
    AssertIndexEqual(sut, at: 6, inputIndex: 4, subIndex: .complete)
    AssertElementEqual(sut, at: 6, .option(.name(.long("foo"))))
    
    XCTAssertEqual(sut.originalInput.count, 5)
    XCTAssertEqual(sut.originalInput, ["1234", "-zz", "abc", "-x", "--foo"])
  }
  
  func testTerminator_1() throws {
    let sut = try SplitArguments(arguments: ["--foo", "--", "--bar"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("foo"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .terminator)
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .value("--bar"))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["--foo", "--", "--bar"])
  }
  
  func testTerminator_2() throws {
    let sut = try SplitArguments(arguments: ["--foo", "--", "bar"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("foo"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .terminator)
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .value("bar"))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["--foo", "--", "bar"])
  }
  
  func testTerminator_3() throws {
    let sut = try SplitArguments(arguments: ["--foo", "--", "--bar=baz"])
    
    XCTAssertEqual(sut.elements.count, 3)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("foo"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .terminator)
    
    AssertIndexEqual(sut, at: 2, inputIndex: 2, subIndex: .complete)
    AssertElementEqual(sut, at: 2, .value("--bar=baz"))
    
    XCTAssertEqual(sut.originalInput.count, 3)
    XCTAssertEqual(sut.originalInput, ["--foo", "--", "--bar=baz"])
  }
  
  func testTerminatorAtTheEnd() throws {
    let sut = try SplitArguments(arguments: ["--foo", "--"])
    
    XCTAssertEqual(sut.elements.count, 2)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.long("foo"))))
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .terminator)
    
    XCTAssertEqual(sut.originalInput.count, 2)
    XCTAssertEqual(sut.originalInput, ["--foo", "--"])
  }
  
  func testTerminatorAtTheBeginning() throws {
    let sut = try SplitArguments(arguments: ["--", "--foo"])
    
    XCTAssertEqual(sut.elements.count, 2)
    
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .terminator)
    
    AssertIndexEqual(sut, at: 1, inputIndex: 1, subIndex: .complete)
    AssertElementEqual(sut, at: 1, .value("--foo"))
    
    XCTAssertEqual(sut.originalInput.count, 2)
    XCTAssertEqual(sut.originalInput, ["--", "--foo"])
  }
}

// MARK: - Removing Entries

extension SplitArgumentTests {
  func testRemovingValuesForLongNames() throws {
    var sut = try SplitArguments(arguments: ["--foo", "--bar"])
    XCTAssertEqual(sut.elements.count, 2)
    sut.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 1)
    sut.remove(at: SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 0)
  }
  
  func testRemovingValuesForLongNamesWithValue() throws {
    var sut = try SplitArguments(arguments: ["--foo=A", "--bar=B"])
    XCTAssertEqual(sut.elements.count, 2)
    sut.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 1)
    sut.remove(at: SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 0)
  }
  
  func testRemovingValuesForShortNames() throws {
    var sut = try SplitArguments(arguments: ["-f", "-b"])
    XCTAssertEqual(sut.elements.count, 2)
    sut.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 1)
    sut.remove(at: SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(sut.elements.count, 0)
  }
  
  func testRemovingValuesForCombinedShortNames() throws {
    let sut = try SplitArguments(arguments: ["-fb"])
    
    XCTAssertEqual(sut.elements.count, 3)
    AssertIndexEqual(sut, at: 0, inputIndex: 0, subIndex: .complete)
    AssertElementEqual(sut, at: 0, .option(.name(.longWithSingleDash("fb"))))
    AssertIndexEqual(sut, at: 1, inputIndex: 0, subIndex: .sub(0))
    AssertElementEqual(sut, at: 1, .option(.name(.short("f"))))
    AssertIndexEqual(sut, at: 2, inputIndex: 0, subIndex: .sub(1))
    AssertElementEqual(sut, at: 2, .option(.name(.short("b"))))
    
    do {
      var sutB = sut
      sutB.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .complete))
      
      XCTAssertEqual(sutB.elements.count, 0)
    }
    do {
      var sutB = sut
      sutB.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .sub(0)))
      
      XCTAssertEqual(sutB.elements.count, 1)
      AssertIndexEqual(sutB, at: 2, inputIndex: 0, subIndex: .sub(1))
      AssertElementEqual(sutB, at: 2, .option(.name(.short("b"))))
    }
    do {
      var sutB = sut
      sutB.remove(at: SplitArguments.Index(inputIndex: 0, subIndex: .sub(1)))
      
      XCTAssertEqual(sutB.elements.count, 1)
      AssertIndexEqual(sutB, at: 2, inputIndex: 0, subIndex: .sub(0))
      AssertElementEqual(sutB, at: 2, .option(.name(.short("f"))))
    }
  }
}

// MARK: - Pop & Peek

extension SplitArgumentTests {
  func testPopNext() throws {
    var sut = try SplitArguments(arguments: ["--foo", "bar"])
    
    let a = try XCTUnwrap(sut.popNext())
    XCTAssertEqual(a.0, .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete)))
    XCTAssertEqual(a.1.value, .option(.name(.long("foo"))))
    
    let b = try XCTUnwrap(sut.popNext())
    XCTAssertEqual(b.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(b.1.value, .value("bar"))
    
    XCTAssertNil(sut.popNext())
  }
  
  func testPeekNext() throws {
    let sut = try SplitArguments(arguments: ["--foo", "bar"])
    
    let a = try XCTUnwrap(sut.peekNext())
    XCTAssertEqual(a.0, .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete)))
    XCTAssertEqual(a.1.value, .option(.name(.long("foo"))))
    
    let b = try XCTUnwrap(sut.peekNext())
    XCTAssertEqual(b.0, .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete)))
    XCTAssertEqual(b.1.value, .option(.name(.long("foo"))))
  }
  
  func testPeekNextWhenEmpty() throws {
    let sut = try SplitArguments(arguments: [])
    XCTAssertNil(sut.peekNext())
  }
  
  func testPopNextElementIfValueAfter_1() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let value = try XCTUnwrap(sut.popNextElementIfValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(value.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(value.1, "bar")
  }
  
  func testPopNextElementIfValueAfter_2() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let value = try XCTUnwrap(sut.popNextElementIfValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 2, subIndex: .complete))))
    XCTAssertEqual(value.0, .argumentIndex(SplitArguments.Index(inputIndex: 3, subIndex: .complete)))
    XCTAssertEqual(value.1, "foo")
  }
  
  func testPopNextElementIfValueAfter_3() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    XCTAssertNil(sut.popNextElementIfValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete))))
  }
  
  func testPopNextValueAfter_1() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let valueA = try XCTUnwrap(sut.popNextValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(valueA.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(valueA.1, "bar")
    
    let valueB = try XCTUnwrap(sut.popNextValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(valueB.0, .argumentIndex(SplitArguments.Index(inputIndex: 3, subIndex: .complete)))
    XCTAssertEqual(valueB.1, "foo")
  }
  
  func testPopNextValueAfter_2() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let value = try XCTUnwrap(sut.popNextValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 2, subIndex: .complete))))
    XCTAssertEqual(value.0, .argumentIndex(SplitArguments.Index(inputIndex: 3, subIndex: .complete)))
    XCTAssertEqual(value.1, "foo")
    
    XCTAssertNil(sut.popNextValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 2, subIndex: .complete))))
  }
  
  func testPopNextValueAfter_3() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    XCTAssertNil(sut.popNextValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 3, subIndex: .complete))))
  }
  
  func testPopNextElementAsValueAfter_1() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let valueA = try XCTUnwrap(sut.popNextElementAsValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(valueA.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(valueA.1, "bar")
    
    let valueB = try XCTUnwrap(sut.popNextElementAsValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(valueB.0, .argumentIndex(SplitArguments.Index(inputIndex: 2, subIndex: .complete)))
    XCTAssertEqual(valueB.1, "--foo")
  }
  
  func testPopNextElementAsValueAfter_2() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    XCTAssertNil(sut.popNextElementAsValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 3, subIndex: .complete))))
  }
  
  func testPopNextElementAsValueAfter_3() throws {
    var sut = try SplitArguments(arguments: ["--bar", "-bar"])
    
    let value = try XCTUnwrap(sut.popNextElementAsValue(after: .argumentIndex(SplitArguments.Index(inputIndex: 0, subIndex: .complete))))
    XCTAssertEqual(value.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(value.1, "-bar")
  }
  
  func testPopNextElementIfValue() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    _ = try XCTUnwrap(sut.popNext())
    
    let value = try XCTUnwrap(sut.popNextElementIfValue())
    XCTAssertEqual(value.0, .argumentIndex(SplitArguments.Index(inputIndex: 1, subIndex: .complete)))
    XCTAssertEqual(value.1, "bar")
    
    XCTAssertNil(sut.popNextElementIfValue())
  }
  
  func testPopNextValue() throws {
    var sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let valueA = try XCTUnwrap(sut.popNextValue())
    XCTAssertEqual(valueA.0, SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(valueA.1, "bar")
    
    let valueB = try XCTUnwrap(sut.popNextValue())
    XCTAssertEqual(valueB.0, SplitArguments.Index(inputIndex: 3, subIndex: .complete))
    XCTAssertEqual(valueB.1, "foo")
    
    XCTAssertNil(sut.popNextElementIfValue())
  }
  
  func testPeekNextValue() throws {
    let sut = try SplitArguments(arguments: ["--bar", "bar", "--foo", "foo"])
    
    let valueA = try XCTUnwrap(sut.peekNextValue())
    XCTAssertEqual(valueA.0, SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(valueA.1, "bar")
    
    let valueB = try XCTUnwrap(sut.peekNextValue())
    XCTAssertEqual(valueB.0, SplitArguments.Index(inputIndex: 1, subIndex: .complete))
    XCTAssertEqual(valueB.1, "bar")
  }
}
