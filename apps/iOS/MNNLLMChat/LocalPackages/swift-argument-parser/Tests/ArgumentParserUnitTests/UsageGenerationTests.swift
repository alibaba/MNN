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

final class UsageGenerationTests: XCTestCase {
}

func _testSynopsis<T: ParsableArguments>(
  _ type: T.Type,
  visibility: ArgumentVisibility = .default,
  expected: String,
  file: StaticString = #file,
  line: UInt = #line
) {
  let help = UsageGenerator(toolName: "example", parsable: T(), visibility: visibility, parent: nil)
  XCTAssertEqual(help.synopsis, expected, file: file, line: line)
}

// MARK: -

extension UsageGenerationTests {
  func testNameSynopsis() {
    XCTAssertEqual(Name.long("foo").synopsisString, "--foo")
    XCTAssertEqual(Name.short("f").synopsisString, "-f")
    XCTAssertEqual(Name.longWithSingleDash("foo").synopsisString, "-foo")
  }
}

extension UsageGenerationTests {
  struct A: ParsableArguments {
    @Option() var firstName: String
    @Option() var title: String
  }

  func testSynopsis() {
    _testSynopsis(A.self, expected: "example --first-name <first-name> --title <title>")
  }

  struct B: ParsableArguments {
    @Option() var firstName: String?
    @Option() var title: String?
  }

  func testSynopsisWithOptional() {
    _testSynopsis(B.self, expected: "example [--first-name <first-name>] [--title <title>]")
  }

  struct C: ParsableArguments {
    @Flag var log: Bool = false
    @Flag() var verbose: Int
  }

  func testFlagSynopsis() {
    _testSynopsis(C.self, expected: "example [--log] [--verbose ...]")
  }

  struct D: ParsableArguments {
    @Argument() var firstName: String
    @Argument() var title: String?
  }

  func testPositionalSynopsis() {
    _testSynopsis(D.self, expected: "example <first-name> [<title>]")
  }

  struct E: ParsableArguments {
    @Option
    var name: String = "no-name"

    @Option
    var count: Int = 0

    @Argument
    var arg: String = "no-arg"
  }

  func testSynopsisWithDefaults() {
    _testSynopsis(E.self, expected: "example [--name <name>] [--count <count>] [<arg>]")
  }

  struct F: ParsableArguments {
    @Option() var name: [String] = []
    @Argument() var nameCounts: [Int] = []
  }

  func testSynopsisWithRepeats() {
    _testSynopsis(F.self, expected: "example [--name <name> ...] [<name-counts> ...]")
  }

  struct G: ParsableArguments {
    @Option(help: ArgumentHelp(valueName: "path"))
    var filePath: String?

    @Argument(help: ArgumentHelp(valueName: "user-home-path"))
    var homePath: String
  }

  func testSynopsisWithCustomization() {
    _testSynopsis(G.self, expected: "example [--file-path <path>] <user-home-path>")
  }

  struct H: ParsableArguments {
    @Option(help: .hidden) var firstName: String?
    @Argument(help: .hidden) var title: String?
  }

  func testSynopsisWithHidden() {
    _testSynopsis(H.self, expected: "example")
    _testSynopsis(H.self, visibility: .hidden, expected: "example [--first-name <first-name>] [<title>]")
  }

  struct I: ParsableArguments {
    enum Color {
        case red, blue
        
        static func transform(_ string: String) throws -> Color {
          switch string {
          case "red":
            return .red
          case "blue":
            return .blue
          default:
            throw ValidationError("Not a valid string for 'Color'")
          }
        }
    }

    @Option(transform: Color.transform)
    var color: Color = .red
  }

  func testSynopsisWithDefaultValueAndTransform() {
    _testSynopsis(I.self, expected: "example [--color <color>]")
  }

  struct J: ParsableArguments {
    struct Foo {}
    @Option(transform: { _ in Foo() }) var req: Foo
    @Option(transform: { _ in Foo() }) var opt: Foo?
  }

  func testSynopsisWithTransform() {
    _testSynopsis(J.self, expected: "example --req <req> [--opt <opt>]")
  }

  struct K: ParsableArguments {
    @Option(
      name: [.short, .customLong("remote"),  .customLong("when"), .customLong("there")],
      help: "Help Message")
    var time: String?
  }

  func testSynopsisWithMultipleCustomNames() {
    _testSynopsis(K.self, expected: "example [--remote <remote>]")
  }

  struct L: ParsableArguments {
    @Option(
      name: [.short, .short, .customLong("remote", withSingleDash: true), .short, .customLong("remote", withSingleDash: true)],
      help: "Help Message")
    var time: String?
  }

  func testSynopsisWithSingleDashLongNameFirst() {
    _testSynopsis(L.self, expected: "example [-remote <remote>]")
  }

  struct M: ParsableArguments {
    enum Color: String, EnumerableFlag {
      case green, blue, yellow
    }
    
    @Flag var a: Bool = false
    @Flag var b: Bool = false
    @Flag var c: Bool = false
    @Flag var d: Bool = false
    @Flag var e: Bool = false
    @Flag var f: Bool = false
    @Flag var g: Bool = false
    @Flag var h: Bool = false
    @Flag var i: Bool = false
    @Flag var j: Bool = false
    @Flag var k: Bool = false
    @Flag var l: Bool = false
    
    @Flag(inversion: .prefixedEnableDisable)
    var optionalBool: Bool?
    
    @Flag var optionalColor: Color?
    
    @Option var option: Bool
    @Argument var input: String
    @Argument var output: String?
  }

  func testSynopsisWithTooManyOptions() {
    _testSynopsis(M.self, expected: "example [<options>] --option <option> <input> [<output>]")
  }
  
  struct N: ParsableArguments {
    @Flag var a: Bool = false
    @Flag var b: Bool = false
    var title = "defaulted value"
    var decode = false
  }
  
  func testNonwrappedValues() {
    _testSynopsis(N.self, expected: "example [--a] [--b]")
    _testSynopsis(N.self, visibility: .hidden, expected: "example [--a] [--b]")
  }
  
  struct O: ParsableArguments {
    @Argument var a: String
    @Argument(parsing: .postTerminator) var b: [String] = []
  }
  
  func testSynopsisWithPostTerminatorParsingStrategy() {
    _testSynopsis(O.self, expected: "example <a> -- [<b> ...]")
  }
}
