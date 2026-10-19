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
import ArgumentParserTestHelpers
import ArgumentParser

/// The goal of this test class is to validate source compatibility. By running
/// this class's tests, all property wrapper initializers should be called.
final class SourceCompatEndToEndTests: XCTestCase {}

// MARK: - Property Wrapper Initializers

fileprivate struct AlmostAllArguments: ParsableArguments {
  @Argument(help: "") var a_newDefaultSyntax: Int = 0
  @Argument() var a0: Int
  @Argument(help: "") var a1: Int
  @Argument var a2_newDefaultSyntax: Int = 0

  @Argument(help: "", transform: { _ in 0 }) var b_newDefaultSyntax: Int = 0
  @Argument var b1_newDefaultSyntax: Int = 0
  @Argument(help: "") var b2: Int
  @Argument(transform: { _ in 0 }) var b3: Int
  @Argument(help: "", transform: { _ in 0 }) var b4: Int
  @Argument(transform: { _ in 0 }) var b5_newDefaultSyntax: Int = 0
  @Argument(help: "") var b6_newDefaultSyntax: Int = 0

  @Argument() var c0: Int?
  @Argument(help: "") var c1: Int?

  @Argument(help: "") var d2: Int?
  @Argument(transform: { _ in 0 }) var d3: Int?
  @Argument(help: "", transform: { _ in 0 }) var d4: Int?

  @Argument(parsing: .remaining, help: "") var e: [Int] = [1, 2]
  @Argument(parsing: .remaining, help: "") var e1: [Int]
  @Argument(parsing: .remaining) var e2: [Int] = [1, 2]
  @Argument(help: "") var e3: [Int] = [1, 2]
  @Argument() var e4: [Int]
  @Argument(help: "") var e5: [Int]
  @Argument(parsing: .remaining) var e6: [Int]
  @Argument() var e7: [Int] = [1, 2]
  @Argument(parsing: .remaining, help: "", transform: { _ in 0 }) var e8: [Int] = [1, 2]
  @Argument(parsing: .remaining, help: "", transform: { _ in 0 }) var e9: [Int]
  @Argument(parsing: .remaining, transform: { _ in 0 }) var e10: [Int] = [1, 2]
  @Argument(help: "", transform: { _ in 0 }) var e11: [Int] = [1, 2]
  @Argument(transform: { _ in 0 }) var e12: [Int]
  @Argument(help: "", transform: { _ in 0 }) var e13: [Int]
  @Argument(parsing: .remaining, transform: { _ in 0 }) var e14: [Int]
  @Argument(transform: { _ in 0 }) var e15: [Int] = [1, 2]
}

fileprivate struct AllOptions: ParsableArguments {
  @Option(name: .long, parsing: .next, help: "") var a_newDefaultSyntax: Int = 0
  @Option(parsing: .next, help: "") var a1_newDefaultSyntax: Int = 0
  @Option(name: .long, parsing: .next, help: "") var a2: Int
  @Option(name: .long, help: "") var a3_newDefaultSyntax: Int = 0
  @Option(parsing: .next, help: "") var a4: Int
  @Option(help: "") var a5_newDefaultSyntax: Int = 0
  @Option(parsing: .next) var a6_newDefaultSyntax: Int = 0
  @Option(name: .long, help: "") var a7: Int
  @Option(name: .long, parsing: .next) var a8: Int
  @Option(name: .long) var a9_newDefaultSyntax: Int = 0
  @Option(name: .long) var a10: Int
  @Option var a11_newDefaultSyntax: Int = 0
  @Option(parsing: .next) var a12: Int
  @Option(help: "") var a13: Int

  @Option(name: .long, parsing: .next, help: "") var b2: Int?
  @Option(parsing: .next, help: "") var b4: Int?
  @Option(name: .long, help: "") var b7: Int?
  @Option(name: .long, parsing: .next) var b8: Int?
  @Option(name: .long) var b10: Int?
  @Option(parsing: .next) var b12: Int?
  @Option(help: "") var b13: Int?

  @Option(name: .long, parsing: .next, help: "", transform: { _ in 0 }) var c_newDefaultSyntax: Int = 0
  @Option(parsing: .next, help: "", transform: { _ in 0 }) var c1_newDefaultSyntax: Int = 0
  @Option(name: .long, parsing: .next, help: "", transform: { _ in 0 }) var c2: Int
  @Option(name: .long, help: "", transform: { _ in 0 }) var c3_newDefaultSyntax: Int = 0
  @Option(parsing: .next, help: "", transform: { _ in 0 }) var c4: Int
  @Option(help: "", transform: { _ in 0 }) var c5_newDefaultSyntax: Int = 0
  @Option(parsing: .next, transform: { _ in 0 }) var c6_newDefaultSyntax: Int = 0
  @Option(name: .long, help: "", transform: { _ in 0 }) var c7: Int
  @Option(name: .long, parsing: .next, transform: { _ in 0 }) var c8: Int
  @Option(name: .long, transform: { _ in 0 }) var c9_newDefaultSyntax: Int = 0
  @Option(name: .long, transform: { _ in 0 }) var c10: Int
  @Option(transform: { _ in 0 }) var c11_newDefaultSyntax: Int = 0
  @Option(parsing: .next, transform: { _ in 0 }) var c12: Int
  @Option(help: "", transform: { _ in 0 }) var c13: Int

  @Option(name: .long, parsing: .next, help: "", transform: { _ in 0 }) var d2: Int?
  @Option(parsing: .next, help: "", transform: { _ in 0 }) var d4: Int?
  @Option(name: .long, help: "", transform: { _ in 0 }) var d7: Int?
  @Option(name: .long, parsing: .next, transform: { _ in 0 }) var d8: Int?
  @Option(name: .long, transform: { _ in 0 }) var d10: Int?
  @Option(parsing: .next, transform: { _ in 0 }) var d12: Int?
  @Option(help: "", transform: { _ in 0 }) var d13: Int?

  @Option(name: .long, parsing: .singleValue, help: "") var e: [Int] = [1, 2]
  @Option(parsing: .singleValue, help: "") var e1: [Int] = [1, 2]
  @Option(name: .long, parsing: .singleValue, help: "") var e2: [Int]
  @Option(name: .long, help: "") var e3: [Int] = [1, 2]
  @Option(parsing: .singleValue, help: "") var e4: [Int]
  @Option(help: "") var e5: [Int] = [1, 2]
  @Option(parsing: .singleValue) var e6: [Int] = [1, 2]
  @Option(name: .long, help: "") var e7: [Int]
  @Option(name: .long, parsing: .singleValue) var e8: [Int]
  @Option(name: .long) var e9: [Int] = [1, 2]
  @Option(name: .long) var e10: [Int]
  @Option() var e11: [Int] = [1, 2]
  @Option(parsing: .singleValue) var e12: [Int]
  @Option(help: "") var e13: [Int]

  @Option(name: .long, parsing: .singleValue, help: "", transform: { _ in 0 }) var f: [Int] = [1, 2]
  @Option(parsing: .singleValue, help: "", transform: { _ in 0 }) var f1: [Int] = [1, 2]
  @Option(name: .long, parsing: .singleValue, help: "", transform: { _ in 0 }) var f2: [Int]
  @Option(name: .long, help: "", transform: { _ in 0 }) var f3: [Int] = [1, 2]
  @Option(parsing: .singleValue, help: "", transform: { _ in 0 }) var f4: [Int]
  @Option(help: "", transform: { _ in 0 }) var f5: [Int] = [1, 2]
  @Option(parsing: .singleValue, transform: { _ in 0 }) var f6: [Int] = [1, 2]
  @Option(name: .long, help: "", transform: { _ in 0 }) var f7: [Int]
  @Option(name: .long, parsing: .singleValue, transform: { _ in 0 }) var f8: [Int]
  @Option(name: .long, transform: { _ in 0 }) var f9: [Int] = [1, 2]
  @Option(name: .long, transform: { _ in 0 }) var f10: [Int]
  @Option(transform: { _ in 0 }) var f11: [Int] = [1, 2]
  @Option(parsing: .singleValue, transform: { _ in 0 }) var f12: [Int]
  @Option(help: "", transform: { _ in 0 }) var f13: [Int]
}

struct AllFlags: ParsableArguments {
  enum E: String, EnumerableFlag {
    case one, two, three
  }

  @Flag(name: .long, help: "") var a_explicitFalse: Bool = false
  @Flag() var a0_explicitFalse: Bool = false
  @Flag(name: .long) var a1_explicitFalse: Bool = false
  @Flag(help: "") var a2_explicitFalse: Bool = false

  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var b: Bool
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var b1: Bool
  @Flag(name: .long, inversion: .prefixedNo, help: "") var b2: Bool
  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast) var b3: Bool
  @Flag(inversion: .prefixedNo, help: "") var b4: Bool
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast) var b5: Bool
  @Flag(name: .long, inversion: .prefixedNo) var b6: Bool
  @Flag(inversion: .prefixedNo) var b7: Bool

  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var c_newDefaultSyntax: Bool = false
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var c1_newDefaultSyntax: Bool = false
  @Flag(name: .long, inversion: .prefixedNo, help: "") var c2_newDefaultSyntax: Bool = false
  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast) var c3_newDefaultSyntax: Bool = false
  @Flag(inversion: .prefixedNo, help: "") var c4_newDefaultSyntax: Bool = false
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast) var c5_newDefaultSyntax: Bool = false
  @Flag(name: .long, inversion: .prefixedNo) var c6_newDefaultSyntax: Bool = false
  @Flag(inversion: .prefixedNo) var c7_newDefaultSyntax: Bool = false

  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var d_implicitNil: Bool
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast, help: "") var d1_implicitNil: Bool
  @Flag(name: .long, inversion: .prefixedNo, help: "") var d2_implicitNil: Bool
  @Flag(name: .long, inversion: .prefixedNo, exclusivity: .chooseLast) var d3_implicitNil: Bool
  @Flag(inversion: .prefixedNo, help: "") var d4_implicitNil: Bool
  @Flag(inversion: .prefixedNo, exclusivity: .chooseLast) var d5_implicitNil: Bool
  @Flag(name: .long, inversion: .prefixedNo) var d6_implicitNil: Bool
  @Flag(inversion: .prefixedNo) var d7_implicitNil: Bool

  @Flag(name: .long, help: "") var e: Int
  @Flag() var e0: Int
  @Flag(name: .long) var e1: Int
  @Flag(help: "") var e2: Int

  @Flag(exclusivity: .chooseLast, help: "") var f_newDefaultSyntax: E = .one
  @Flag() var f1: E
  @Flag(exclusivity: .chooseLast, help: "") var f2: E
  @Flag(help: "") var f3_newDefaultSyntax: E = .one
  @Flag(exclusivity: .chooseLast) var f4_newDefaultSyntax: E = .one
  @Flag(help: "") var f5: E
  @Flag(exclusivity: .chooseLast) var f6: E
  @Flag var f7_newDefaultSyntax: E = .one

  @Flag(exclusivity: .chooseLast, help: "") var g: E?
  @Flag() var g1: E?
  @Flag(help: "") var g2: E?
  @Flag(exclusivity: .chooseLast) var g3: E?

  @Flag(help: "") var h: [E] = []
  @Flag() var h1: [E] = []
  @Flag(help: "") var h2: [E]
  @Flag() var h3: [E]
}

extension SourceCompatEndToEndTests {
  func testParsingAll() throws {
    // This is just checking building the argument definitions, not the actual
    // validation or usage of these definitions, which would fail.
    _ = AlmostAllArguments()
    _ = AllOptions()
    _ = AllFlags()
  }
}

