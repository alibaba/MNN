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

final class DefaultsEndToEndTests: XCTestCase {
}

// MARK: -

fileprivate struct Foo: ParsableArguments {
  struct Name: RawRepresentable, ExpressibleByArgument {
    var rawValue: String
  }
  @Option
  var name: Name = Name(rawValue: "A")
  @Option
  var max: Int = 3
}

extension DefaultsEndToEndTests {
  func testParsing_Defaults() throws {
    AssertParse(Foo.self, []) { foo in
      XCTAssertEqual(foo.name.rawValue, "A")
      XCTAssertEqual(foo.max, 3)
    }

    AssertParse(Foo.self, ["--name", "B"]) { foo in
      XCTAssertEqual(foo.name.rawValue, "B")
      XCTAssertEqual(foo.max, 3)
    }

    AssertParse(Foo.self, ["--max", "5"]) { foo in
      XCTAssertEqual(foo.name.rawValue, "A")
      XCTAssertEqual(foo.max, 5)
    }

    AssertParse(Foo.self, ["--max", "5", "--name", "B"]) { foo in
      XCTAssertEqual(foo.name.rawValue, "B")
      XCTAssertEqual(foo.max, 5)
    }
  }
}

// MARK: -

fileprivate struct Bar: ParsableArguments {
  enum Format: String, ExpressibleByArgument {
    case A
    case B
    case C
  }
  @Option
  var name: String = "N"
  @Option
  var format: Format = .A
  @Option()
  var foo: String
  @Argument()
  var bar: String?
}

extension DefaultsEndToEndTests {
  func testParsing_Optional_WithAllValues_1() {
    AssertParse(Bar.self, ["--name", "A", "--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithAllValues_2() {
    AssertParse(Bar.self, ["D", "--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithAllValues_3() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithMissingValues_1() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithMissingValues_2() {
    AssertParse(Bar.self, ["D", "--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithMissingValues_3() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "D"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, "D")
    }
  }

  func testParsing_Optional_WithMissingValues_4() {
    AssertParse(Bar.self, ["--name", "A", "--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_5() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format,.B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_6() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C", "--name", "A"]) { bar in
      XCTAssertEqual(bar.name, "A")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_7() {
    AssertParse(Bar.self, ["--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .A)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_8() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_9() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithMissingValues_10() {
    AssertParse(Bar.self, ["--format", "B", "--foo", "C"]) { bar in
      XCTAssertEqual(bar.name, "N")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "C")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_Fails() throws {
    XCTAssertThrowsError(try Bar.parse([]))
    XCTAssertThrowsError(try Bar.parse(["--fooz", "C"]))
    XCTAssertThrowsError(try Bar.parse(["--nam", "A", "--foo", "C"]))
    XCTAssertThrowsError(try Bar.parse(["--name"]))
    XCTAssertThrowsError(try Bar.parse(["A"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "D"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "--foo"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "--format", "B"]))
    XCTAssertThrowsError(try Bar.parse(["--name", "A", "-f"]))
    XCTAssertThrowsError(try Bar.parse(["D", "--name", "A"]))
    XCTAssertThrowsError(try Bar.parse(["-f", "--name", "A"]))
    XCTAssertThrowsError(try Bar.parse(["--foo", "--name", "A"]))
    XCTAssertThrowsError(try Bar.parse(["--foo", "--name", "AA", "BB"]))
  }
}

fileprivate struct Bar_NextInput: ParsableArguments {
  enum Format: String, ExpressibleByArgument {
    case A
    case B
    case C
    case D = "-d"
  }
  @Option(parsing: .unconditional)
  var name: String = "N"
  @Option(parsing: .unconditional)
  var format: Format = .A
  @Option(parsing: .unconditional)
  var foo: String
  @Argument()
  var bar: String?
}

extension DefaultsEndToEndTests {
  func testParsing_Optional_WithOverlappingValues_1() {
    AssertParse(Bar_NextInput.self, ["--format", "B", "--name", "--foo", "--foo", "--name"]) { bar in
      XCTAssertEqual(bar.name, "--foo")
      XCTAssertEqual(bar.format, .B)
      XCTAssertEqual(bar.foo, "--name")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithOverlappingValues_2() {
    AssertParse(Bar_NextInput.self, ["--format", "-d", "--foo", "--name", "--name", "--foo"]) { bar in
      XCTAssertEqual(bar.name, "--foo")
      XCTAssertEqual(bar.format, .D)
      XCTAssertEqual(bar.foo, "--name")
      XCTAssertEqual(bar.bar, nil)
    }
  }

  func testParsing_Optional_WithOverlappingValues_3() {
    AssertParse(Bar_NextInput.self, ["--format", "-d", "--name", "--foo", "--foo", "--name", "bar"]) { bar in
      XCTAssertEqual(bar.name, "--foo")
      XCTAssertEqual(bar.format, .D)
      XCTAssertEqual(bar.foo, "--name")
      XCTAssertEqual(bar.bar, "bar")
    }
  }
}

// MARK: -

fileprivate struct Baz: ParsableArguments {
  @Option(parsing: .unconditional) var int: Int = 0
  @Option(parsing: .unconditional) var int8: Int8 = 0
  @Option(parsing: .unconditional) var int16: Int16 = 0
  @Option(parsing: .unconditional) var int32: Int32 = 0
  @Option(parsing: .unconditional) var int64: Int64 = 0
  @Option var uint: UInt = 0
  @Option var uint8: UInt8 = 0
  @Option var uint16: UInt16 = 0
  @Option var uint32: UInt32 = 0
  @Option var uint64: UInt64 = 0

  @Option(parsing: .unconditional) var float: Float = 0
  @Option(parsing: .unconditional) var double: Double = 0

  @Option var bool: Bool = false
}

extension DefaultsEndToEndTests {
  func testParsing_AllTypes_1() {
    AssertParse(Baz.self, []) { baz in
      XCTAssertEqual(baz.int, 0)
      XCTAssertEqual(baz.int8, 0)
      XCTAssertEqual(baz.int16, 0)
      XCTAssertEqual(baz.int32, 0)
      XCTAssertEqual(baz.int64, 0)
      XCTAssertEqual(baz.uint, 0)
      XCTAssertEqual(baz.uint8, 0)
      XCTAssertEqual(baz.uint16, 0)
      XCTAssertEqual(baz.uint32, 0)
      XCTAssertEqual(baz.uint64, 0)
      XCTAssertEqual(baz.float, 0)
      XCTAssertEqual(baz.double, 0)
      XCTAssertEqual(baz.bool, false)
    }
  }

  func testParsing_AllTypes_2() {
    AssertParse(Baz.self, [
      "--int", "-1", "--int8", "-2", "--int16", "-3", "--int32", "-4", "--int64", "-5",
      "--uint", "1", "--uint8", "2", "--uint16", "3", "--uint32", "4", "--uint64", "5",
      "--float", "1.25", "--double", "2.5", "--bool", "true"
    ]) { baz in
      XCTAssertEqual(baz.int, -1)
      XCTAssertEqual(baz.int8, -2)
      XCTAssertEqual(baz.int16, -3)
      XCTAssertEqual(baz.int32, -4)
      XCTAssertEqual(baz.int64, -5)
      XCTAssertEqual(baz.uint, 1)
      XCTAssertEqual(baz.uint8, 2)
      XCTAssertEqual(baz.uint16, 3)
      XCTAssertEqual(baz.uint32, 4)
      XCTAssertEqual(baz.uint64, 5)
      XCTAssertEqual(baz.float, 1.25)
      XCTAssertEqual(baz.double, 2.5)
      XCTAssertEqual(baz.bool, true)
    }
  }

  func testParsing_AllTypes_Fails() throws {
    XCTAssertThrowsError(try Baz.parse(["--int8", "256"]))
    XCTAssertThrowsError(try Baz.parse(["--int16", "32768"]))
    XCTAssertThrowsError(try Baz.parse(["--int32", "2147483648"]))
    XCTAssertThrowsError(try Baz.parse(["--int64", "9223372036854775808"]))
    XCTAssertThrowsError(try Baz.parse(["--int", "9223372036854775808"]))

    XCTAssertThrowsError(try Baz.parse(["--uint8", "512"]))
    XCTAssertThrowsError(try Baz.parse(["--uint16", "65536"]))
    XCTAssertThrowsError(try Baz.parse(["--uint32", "4294967296"]))
    XCTAssertThrowsError(try Baz.parse(["--uint64", "18446744073709551616"]))
    XCTAssertThrowsError(try Baz.parse(["--uint", "18446744073709551616"]))

    XCTAssertThrowsError(try Baz.parse(["--uint8", "-1"]))
    XCTAssertThrowsError(try Baz.parse(["--uint16", "-1"]))
    XCTAssertThrowsError(try Baz.parse(["--uint32", "-1"]))
    XCTAssertThrowsError(try Baz.parse(["--uint64", "-1"]))
    XCTAssertThrowsError(try Baz.parse(["--uint", "-1"]))

    XCTAssertThrowsError(try Baz.parse(["--float", "zzz"]))
    XCTAssertThrowsError(try Baz.parse(["--double", "zzz"]))
    XCTAssertThrowsError(try Baz.parse(["--bool", "truthy"]))
  }
}

fileprivate struct Qux: ParsableArguments {
  @Argument
  var name: String = "quux"
}

extension DefaultsEndToEndTests {
  func testParsing_ArgumentDefaults() throws {
    AssertParse(Qux.self, []) { qux in
      XCTAssertEqual(qux.name, "quux")
    }
    AssertParse(Qux.self, ["Bar"]) { qux in
      XCTAssertEqual(qux.name, "Bar")
    }
    AssertParse(Qux.self, ["Bar-"]) { qux in
      XCTAssertEqual(qux.name, "Bar-")
    }
    AssertParse(Qux.self, ["Bar--"]) { qux in
      XCTAssertEqual(qux.name, "Bar--")
    }
    AssertParse(Qux.self, ["--", "-Bar"]) { qux in
      XCTAssertEqual(qux.name, "-Bar")
    }
    AssertParse(Qux.self, ["--", "--Bar"]) { qux in
      XCTAssertEqual(qux.name, "--Bar")
    }
    AssertParse(Qux.self, ["--", "--"]) { qux in
      XCTAssertEqual(qux.name, "--")
    }
  }

  func testParsing_ArgumentDefaults_Fails() throws {
    XCTAssertThrowsError(try Qux.parse(["--name"]))
    XCTAssertThrowsError(try Qux.parse(["Foo", "Bar"]))
  }
}

fileprivate func exclaim(_ input: String) throws -> String {
  return input + "!"
}

fileprivate struct OptionPropertyInitArguments_Default: ParsableArguments {
  @Option
  var data: String = "test"

  @Option(transform: exclaim)
  var transformedData: String = "test"
}

fileprivate struct OptionPropertyInitArguments_NoDefault_NoTransform: ParsableArguments {
  @Option()
  var data: String
}

fileprivate struct OptionPropertyInitArguments_NoDefault_Transform: ParsableArguments {
  @Option(transform: exclaim)
  var transformedData: String
}

extension DefaultsEndToEndTests {
  /// Tests that using default property initialization syntax parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_OptionPropertyInit_Default_NoTransform_UseDefault() throws {
    AssertParse(OptionPropertyInitArguments_Default.self, []) { arguments in
      XCTAssertEqual(arguments.data, "test")
    }
  }

  /// Tests that using default property initialization syntax parses the command-line-provided value for the argument when provided.
  func testParsing_OptionPropertyInit_Default_NoTransform_OverrideDefault() throws {
    AssertParse(OptionPropertyInitArguments_Default.self, ["--data", "test2"]) { arguments in
      XCTAssertEqual(arguments.data, "test2")
    }
  }

  /// Tests that *not* providing a default value still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_OptionPropertyInit_NoDefault_NoTransform() throws {
    AssertParse(OptionPropertyInitArguments_NoDefault_NoTransform.self, ["--data", "test"]) { arguments in
      XCTAssertEqual(arguments.data, "test")
    }
  }

  /// Tests that using default property initialization syntax on a property with a `transform` function provided parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_OptionPropertyInit_Default_Transform_UseDefault() throws {
    AssertParse(OptionPropertyInitArguments_Default.self, []) { arguments in
      XCTAssertEqual(arguments.transformedData, "test")
    }
  }

  /// Tests that using default property initialization syntax on a property with a `transform` function provided parses and transforms the command-line-provided value for the argument when provided.
  func testParsing_OptionPropertyInit_Default_Transform_OverrideDefault() throws {
    AssertParse(OptionPropertyInitArguments_Default.self, ["--transformed-data", "test2"]) { arguments in
      XCTAssertEqual(arguments.transformedData, "test2!")
    }
  }

  /// Tests that *not* providing a default value for a property with a `transform` function still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_OptionPropertyInit_NoDefault_Transform() throws {
    AssertParse(OptionPropertyInitArguments_NoDefault_Transform.self, ["--transformed-data", "test"]) { arguments in
      XCTAssertEqual(arguments.transformedData, "test!")
    }
  }
}


fileprivate struct ArgumentPropertyInitArguments_Default_NoTransform: ParsableArguments {
  @Argument
  var data: String = "test"
}

fileprivate struct ArgumentPropertyInitArguments_NoDefault_NoTransform: ParsableArguments {
  @Argument()
  var data: String
}

fileprivate struct ArgumentPropertyInitArguments_Default_Transform: ParsableArguments {
  @Argument(transform: exclaim)
    var transformedData: String = "test"
}

fileprivate struct ArgumentPropertyInitArguments_NoDefault_Transform: ParsableArguments {
  @Argument(transform: exclaim)
  var transformedData: String
}

extension DefaultsEndToEndTests {
  /// Tests that using default property initialization syntax parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_ArgumentPropertyInit_Default_NoTransform_UseDefault() throws {
    AssertParse(ArgumentPropertyInitArguments_Default_NoTransform.self, []) { arguments in
      XCTAssertEqual(arguments.data, "test")
    }
  }

  /// Tests that using default property initialization syntax parses the command-line-provided value for the argument when provided.
  func testParsing_ArgumentPropertyInit_Default_NoTransform_OverrideDefault() throws {
    AssertParse(ArgumentPropertyInitArguments_Default_NoTransform.self, ["test2"]) { arguments in
      XCTAssertEqual(arguments.data, "test2")
    }
  }

  /// Tests that *not* providing a default value still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_ArgumentPropertyInit_NoDefault_NoTransform() throws {
    AssertParse(ArgumentPropertyInitArguments_NoDefault_NoTransform.self, ["test"]) { arguments in
      XCTAssertEqual(arguments.data, "test")
    }
  }

  /// Tests that using default property initialization syntax on a property with a `transform` function provided parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_ArgumentPropertyInit_Default_Transform_UseDefault() throws {
    AssertParse(ArgumentPropertyInitArguments_Default_Transform.self, []) { arguments in
      XCTAssertEqual(arguments.transformedData, "test")
    }
  }

  /// Tests that using default property initialization syntax on a property with a `transform` function provided parses and transforms the command-line-provided value for the argument when provided.
  func testParsing_ArgumentPropertyInit_Default_Transform_OverrideDefault() throws {
    AssertParse(ArgumentPropertyInitArguments_Default_Transform.self, ["test2"]) { arguments in
      XCTAssertEqual(arguments.transformedData, "test2!")
    }
  }

  /// Tests that *not* providing a default value for a property with a `transform` function still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_ArgumentPropertyInit_NoDefault_Transform() throws {
    AssertParse(ArgumentPropertyInitArguments_NoDefault_Transform.self, ["test"]) { arguments in
      XCTAssertEqual(arguments.transformedData, "test!")
    }
  }
}

fileprivate struct Quux: ParsableArguments {
  @Option(parsing: .upToNextOption)
  var letters: [String] = ["A", "B"]
  
  @Argument()
  var numbers: [Int] = [1, 2]
}

extension DefaultsEndToEndTests {
  func testParsing_ArrayDefaults() throws {
    AssertParse(Quux.self, []) { qux in
      XCTAssertEqual(qux.letters, ["A", "B"])
      XCTAssertEqual(qux.numbers, [1, 2])
    }
    AssertParse(Quux.self, ["--letters", "C", "D"]) { qux in
      XCTAssertEqual(qux.letters, ["C", "D"])
      XCTAssertEqual(qux.numbers, [1, 2])
    }
    AssertParse(Quux.self, ["3", "4"]) { qux in
      XCTAssertEqual(qux.letters, ["A", "B"])
      XCTAssertEqual(qux.numbers, [3, 4])
    }
    AssertParse(Quux.self, ["3", "4", "--letters", "C", "D"]) { qux in
      XCTAssertEqual(qux.letters, ["C", "D"])
      XCTAssertEqual(qux.numbers, [3, 4])
    }
  }
}

fileprivate struct FlagPropertyInitArguments_Bool_Default: ParsableArguments {
  @Flag(inversion: .prefixedNo)
  var data: Bool = false
}

fileprivate struct FlagPropertyInitArguments_Bool_NoDefault: ParsableArguments {
  @Flag(inversion: .prefixedNo)
  var data: Bool
}

extension DefaultsEndToEndTests {
  /// Tests that using default property initialization syntax parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_FlagPropertyInit_Bool_Default_UseDefault() throws {
    AssertParse(FlagPropertyInitArguments_Bool_Default.self, []) { arguments in
      XCTAssertEqual(arguments.data, false)
    }
  }

  /// Tests that using default property initialization syntax parses the command-line-provided value for the argument when provided.
  func testParsing_FlagPropertyInit_Bool_Default_OverrideDefault() throws {
    AssertParse(FlagPropertyInitArguments_Bool_Default.self, ["--data"]) { arguments in
      XCTAssertEqual(arguments.data, true)
    }
  }

  /// Tests that *not* providing a default value still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_FlagPropertyInit_Bool_NoDefault() throws {
    AssertParse(FlagPropertyInitArguments_Bool_NoDefault.self, ["--data"]) { arguments in
      XCTAssertEqual(arguments.data, true)
    }
  }
}


fileprivate enum HasData: EnumerableFlag {
  case noData
  case data
}

fileprivate struct FlagPropertyInitArguments_EnumerableFlag_Default: ParsableArguments {
  @Flag
  var data: HasData = .noData
}

fileprivate struct FlagPropertyInitArguments_EnumerableFlag_NoDefault: ParsableArguments {
  @Flag()
  var data: HasData
}


extension DefaultsEndToEndTests {
  /// Tests that using default property initialization syntax parses the default value for the argument when nothing is provided from the command-line.
  func testParsing_FlagPropertyInit_EnumerableFlag_Default_UseDefault() throws {
    AssertParse(FlagPropertyInitArguments_EnumerableFlag_Default.self, []) { arguments in
      XCTAssertEqual(arguments.data, .noData)
    }
  }

  /// Tests that using default property initialization syntax parses the command-line-provided value for the argument when provided.
  func testParsing_FlagPropertyInit_EnumerableFlag_Default_OverrideDefault() throws {
    AssertParse(FlagPropertyInitArguments_EnumerableFlag_Default.self, ["--data"]) { arguments in
      XCTAssertEqual(arguments.data, .data)
    }
  }

  /// Tests that *not* providing a default value still parses the argument correctly from the command-line.
  /// This test is almost certainly duplicated by others in the repository, but allows for quick use of test filtering during development on the initialization functionality.
  func testParsing_FlagPropertyInit_EnumerableFlag_NoDefault() throws {
    AssertParse(FlagPropertyInitArguments_EnumerableFlag_NoDefault.self, ["--data"]) { arguments in
      XCTAssertEqual(arguments.data, .data)
    }
  }
}

fileprivate struct Main: ParsableCommand {
  static let configuration = CommandConfiguration(
    subcommands: [Sub.self],
    defaultSubcommand: Sub.self
  )
  
  struct Options: ParsableArguments {
    @Option(parsing: .upToNextOption)
    var letters: [String] = ["A", "B"]
  }
  
  struct Sub: ParsableCommand {
    @Argument()
    var numbers: [Int] = [1, 2]
    
    @OptionGroup()
    var options: Main.Options
  }
}

extension DefaultsEndToEndTests {
  func testParsing_ArrayDefaults_Subcommands() {
    AssertParseCommand(Main.self, Main.Sub.self, []) { sub in
      XCTAssertEqual(sub.options.letters, ["A", "B"])
      XCTAssertEqual(sub.numbers, [1, 2])
    }
    AssertParseCommand(Main.self, Main.Sub.self, ["--letters", "C", "D"]) { sub in
      XCTAssertEqual(sub.options.letters, ["C", "D"])
      XCTAssertEqual(sub.numbers, [1, 2])
    }
    AssertParseCommand(Main.self, Main.Sub.self, ["3", "4"]) { sub in
      XCTAssertEqual(sub.options.letters, ["A", "B"])
      XCTAssertEqual(sub.numbers, [3, 4])
    }
    AssertParseCommand(Main.self, Main.Sub.self, ["3", "4", "--letters", "C", "D"]) { sub in
      XCTAssertEqual(sub.options.letters, ["C", "D"])
      XCTAssertEqual(sub.numbers, [3, 4])
    }
  }
}


fileprivate struct RequiredArray_Option_NoTransform: ParsableArguments {
  @Option(parsing: .remaining)
  var array: [String]
}

fileprivate struct RequiredArray_Option_Transform: ParsableArguments {
  @Option(parsing: .remaining, transform: exclaim)
  var array: [String]
}

fileprivate struct RequiredArray_Argument_NoTransform: ParsableArguments {
  @Argument()
  var array: [String]
}

fileprivate struct RequiredArray_Argument_Transform: ParsableArguments {
  @Argument(transform: exclaim)
  var array: [String]
}

fileprivate struct RequiredArray_Flag: ParsableArguments {
  @Flag
  var array: [HasData]
}

extension DefaultsEndToEndTests {
  /// Tests that not providing an argument for a required array option produces an error.
  func testParsing_RequiredArray_Option_NoTransform_NoInput() {
    XCTAssertThrowsError(try RequiredArray_Option_NoTransform.parse([]))
  }

  /// Tests that providing a single argument for a required array option parses that value correctly.
  func testParsing_RequiredArray_Option_NoTransform_SingleInput() {
    AssertParse(RequiredArray_Option_NoTransform.self, ["--array", "1"]) { arguments in
      XCTAssertEqual(arguments.array, ["1"])
    }
  }

  /// Tests that providing multiple arguments for a required array option parses those values correctly.
  func testParsing_RequiredArray_Option_NoTransform_MultipleInput() {
    AssertParse(RequiredArray_Option_NoTransform.self, ["--array", "2", "3"]) { arguments in
      XCTAssertEqual(arguments.array, ["2", "3"])
    }
  }

  /// Tests that not providing an argument for a required array option with a transform produces an error.
  func testParsing_RequiredArray_Option_Transform_NoInput() {
    XCTAssertThrowsError(try RequiredArray_Option_Transform.parse([]))
  }

  /// Tests that providing a single argument for a required array option with a transform parses that value correctly.
  func testParsing_RequiredArray_Option_Transform_SingleInput() {
    AssertParse(RequiredArray_Option_Transform.self, ["--array", "1"]) { arguments in
      XCTAssertEqual(arguments.array, ["1!"])
    }
  }

  /// Tests that providing multiple arguments for a required array option with a transform parses those values correctly.
  func testParsing_RequiredArray_Option_Transform_MultipleInput() {
    AssertParse(RequiredArray_Option_Transform.self, ["--array", "2", "3"]) { arguments in
      XCTAssertEqual(arguments.array, ["2!", "3!"])
    }
  }


  /// Tests that not providing an argument for a required array argument produces an error.
  func testParsing_RequiredArray_Argument_NoTransform_NoInput() {
    XCTAssertThrowsError(try RequiredArray_Argument_NoTransform.parse([]))
  }

  /// Tests that providing a single argument for a required array argument parses that value correctly.
  func testParsing_RequiredArray_Argument_NoTransform_SingleInput() {
    AssertParse(RequiredArray_Argument_NoTransform.self, ["1"]) { arguments in
      XCTAssertEqual(arguments.array, ["1"])
    }
  }

  /// Tests that providing multiple arguments for a required array argument parses those values correctly.
  func testParsing_RequiredArray_Argument_NoTransform_MultipleInput() {
    AssertParse(RequiredArray_Argument_NoTransform.self, ["2", "3"]) { arguments in
      XCTAssertEqual(arguments.array, ["2", "3"])
    }
  }

  /// Tests that not providing an argument for a required array argument with a transform produces an error.
  func testParsing_RequiredArray_Argument_Transform_NoInput() {
    XCTAssertThrowsError(try RequiredArray_Argument_Transform.parse([]))
  }

  /// Tests that providing a single argument for a required array argument with a transform parses that value correctly.
  func testParsing_RequiredArray_Argument_Transform_SingleInput() {
    AssertParse(RequiredArray_Argument_Transform.self, ["1"]) { arguments in
      XCTAssertEqual(arguments.array, ["1!"])
    }
  }

  /// Tests that providing multiple arguments for a required array argument with a transform parses those values correctly.
  func testParsing_RequiredArray_Argument_Transform_MultipleInput() {
    AssertParse(RequiredArray_Argument_Transform.self, ["2", "3"]) { arguments in
      XCTAssertEqual(arguments.array, ["2!", "3!"])
    }
  }


  /// Tests that not providing an argument for a required array flag produces an error.
  func testParsing_RequiredArray_Flag_NoInput() {
    XCTAssertThrowsError(try RequiredArray_Flag.parse([]))
  }

  /// Tests that providing a single argument for a required array flag parses that value correctly.
  func testParsing_RequiredArray_Flag_SingleInput() {
    AssertParse(RequiredArray_Flag.self, ["--data"]) { arguments in
      XCTAssertEqual(arguments.array, [.data])
    }
  }

  /// Tests that providing multiple arguments for a required array flag parses those values correctly.
  func testParsing_RequiredArray_Flag_MultipleInput() {
    AssertParse(RequiredArray_Flag.self, ["--data", "--no-data"]) { arguments in
      XCTAssertEqual(arguments.array, [.data, .noData])
    }
  }
}

@available(*, deprecated)
fileprivate struct OptionPropertyDeprecatedInit_NoDefault: ParsableArguments {
  @Option(completion: .file(), help: "")
  var data: String = "test"
}

extension DefaultsEndToEndTests {
  /// Tests that instances created using deprecated initializer with completion and help arguments swapped are constructed and parsed correctly.
  @available(*, deprecated)
  func testParsing_OptionPropertyDeprecatedInit_NoDefault() {
    AssertParse(OptionPropertyDeprecatedInit_NoDefault.self, []) { arguments in
      XCTAssertEqual(arguments.data, "test")
    }
  }
}

// MARK: Overload selection

extension DefaultsEndToEndTests {
  private struct AbsolutePath: ExpressibleByArgument {
    init(_ value: String) {}
    init?(argument: String) {}
  }
  
  private struct TwoPaths: ParsableCommand {
    @Argument(help: .init("The path"))
    var path1 = AbsolutePath("abc")

    @Argument(help: "The path")
    var path2 = AbsolutePath("abc")

    @Option(help: .init("The path"))
    var path3 = AbsolutePath("abc")

    @Option(help: "The path")
    var path4 = AbsolutePath("abc")
  }
  
  /// Tests that a non-optional `Value` type is inferred, regardless of how the
  /// initializer parameters are spelled. Previously, string literals and
  /// `.init` calls for the help parameter inferred different generic types.
  func testHelpInitInferredType() throws {
    AssertParse(TwoPaths.self, []) { cmd in
      XCTAssert(type(of: cmd.path1) == AbsolutePath.self)
      XCTAssert(type(of: cmd.path2) == AbsolutePath.self)
      XCTAssert(type(of: cmd.path3) == AbsolutePath.self)
      XCTAssert(type(of: cmd.path4) == AbsolutePath.self)
    }
  }
}

extension DefaultsEndToEndTests {
  private struct UnderscoredOptional: ParsableCommand {
    @Option(name: .customLong("arg"))
    var _arg: String?
  }

  private struct UnderscoredArray: ParsableCommand {
    @Option(name: .customLong("columns"), parsing: .upToNextOption)
    var _columns: [String] = []
  }

  func testUnderscoredOptional() throws {
    AssertParse(UnderscoredOptional.self, []) { parsed in
      XCTAssertNil(parsed._arg)
    }
    AssertParse(UnderscoredOptional.self, ["--arg", "foo"]) { parsed in
      XCTAssertEqual(parsed._arg, "foo")
    }
  }

  func testUnderscoredArray() throws {
    AssertParse(UnderscoredArray.self, []) { parsed in
      XCTAssertEqual(parsed._columns, [])
    }
    AssertParse(UnderscoredArray.self, ["--columns", "foo", "bar", "baz"]) { parsed in
      XCTAssertEqual(parsed._columns, ["foo", "bar", "baz"])
    }
  }
}
