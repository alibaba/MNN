//
//  FilterTests.swift
//  Jinja
//
//  Created by Anthony DePasquale on 07.01.2025.
//

// Adapted from https://github.com/pallets/jinja/blob/main/tests/test_filters.py

import XCTest
import OrderedCollections

@testable import Jinja

final class FilterTests: XCTestCase {
    func testFilters() throws {
        // Helper function to run tests for a filter
        func runTest(
            filterName: String,
            input: Any,
            args: [Any?] = [],
            expected: Any,
            file: StaticString = #file,
            line: UInt = #line
        ) throws {
            let env = Environment()

            // Convert input to RuntimeValue
            guard let input = try? env.convertToRuntimeValues(input: input) else {
                XCTFail(
                    "Failed to convert input \(input) to RuntimeValue in test for \(filterName)",
                    file: file,
                    line: line
                )
                return
            }

            // Set the input value in the environment
            try env.set(name: "input", value: input)

            // Set filter arguments in the environment
            for (index, arg) in args.enumerated() {
                if let arg {
                    try env.set(name: "arg\(index)", value: arg)
                }
            }

            // Construct the filter arguments for direct call
            var filterArgs: [any RuntimeValue] = [input]
            for (index, _) in args.enumerated() {
                filterArgs.append(env.lookupVariable(name: "arg\(index)"))
            }

            // Get the filter function from the environment
            guard let filter = env.filters[filterName] else {
                XCTFail("Filter not found: \(filterName)", file: file, line: line)
                return
            }

            // Call the filter function directly with the input and arguments
            let result = try filter(filterArgs, env)

            // Perform assertions based on the expected type
            if let expectedString = expected as? String {
                XCTAssertEqual(
                    (result as? StringValue)?.value,
                    expectedString,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedInt = expected as? Int {
                XCTAssertEqual(
                    (result as? NumericValue)?.value as? Int,
                    expectedInt,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedDouble = expected as? Double {
                XCTAssertEqual(
                    (result as? NumericValue)?.value as? Double,
                    expectedDouble,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedBool = expected as? Bool {
                XCTAssertEqual(
                    (result as? BooleanValue)?.value,
                    expectedBool,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if expected is UndefinedValue {
                XCTAssertTrue(
                    result is UndefinedValue,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedArray = expected as? [String] {
                guard let resultArray = (result as? ArrayValue)?.value else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [String], got \(type(of: result)), value: \(result)",
                        file: file,
                        line: line
                    )
                    return
                }
                let resultStrings = resultArray.compactMap { value -> String? in
                    if let stringValue = value as? StringValue {
                        return stringValue.value
                    } else if value is NullValue {
                        return "None"
                    }
                    return nil
                }
                XCTAssertEqual(
                    resultStrings,
                    expectedArray,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedArray = expected as? [Int] {
                guard let resultArray = (result as? ArrayValue)?.value else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [Int], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }
                let resultInts = resultArray.compactMap { ($0 as? NumericValue)?.value as? Int }
                XCTAssertEqual(
                    resultInts,
                    expectedArray,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedArray = expected as? [[String]] {
                guard let resultArray = (result as? ArrayValue)?.value else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [[String]], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }
                let resultArrays = resultArray.compactMap { value -> [String]? in
                    if let arrayValue = value as? ArrayValue {
                        return arrayValue.value.compactMap { ($0 as? StringValue)?.value }
                    } else if let stringValue = value as? StringValue {
                        return [stringValue.value]
                    }
                    return nil
                }
                XCTAssertEqual(
                    resultArrays,
                    expectedArray,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedArray = expected as? [[Int]] {
                guard let resultArray = (result as? ArrayValue)?.value as? [ArrayValue] else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [[Int]], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }
                let resultInts = resultArray.map { $0.value.compactMap { ($0 as? NumericValue)?.value as? Int } }
                XCTAssertEqual(
                    resultInts,
                    expectedArray,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedDict = expected as? [String: Any] {
                guard let resultDict = (result as? ObjectValue)?.value else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [String: Any], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }
                XCTAssertEqual(
                    resultDict.count,
                    expectedDict.count,
                    "\(filterName) filter failed: Dictionary count mismatch",
                    file: file,
                    line: line
                )
                for (key, expectedValue) in expectedDict {
                    guard let resultValue = resultDict[key] else {
                        XCTFail(
                            "\(filterName) filter failed: Missing key '\(key)' in result",
                            file: file,
                            line: line
                        )
                        return
                    }
                    if let expectedString = expectedValue as? String {
                        XCTAssertEqual(
                            (resultValue as? StringValue)?.value,
                            expectedString,
                            "\(filterName) filter failed for key '\(key)'",
                            file: file,
                            line: line
                        )
                    } else if let expectedInt = expectedValue as? Int {
                        XCTAssertEqual(
                            ((resultValue as? NumericValue)?.value as? Int),
                            expectedInt,
                            "\(filterName) filter failed for key '\(key)'",
                            file: file,
                            line: line
                        )
                    } else if let expectedDouble = expectedValue as? Double {
                        XCTAssertEqual(
                            ((resultValue as? NumericValue)?.value as? Double),
                            expectedDouble,
                            "\(filterName) filter failed for key '\(key)'",
                            file: file,
                            line: line
                        )
                    } else if let expectedBool = expectedValue as? Bool {
                        XCTAssertEqual(
                            (resultValue as? BooleanValue)?.value,
                            expectedBool,
                            "\(filterName) filter failed for key '\(key)'",
                            file: file,
                            line: line
                        )
                    } else if expectedValue is UndefinedValue {
                        XCTAssertTrue(
                            resultValue is UndefinedValue,
                            "\(filterName) filter failed for key '\(key)'",
                            file: file,
                            line: line
                        )
                    } else {
                        XCTFail(
                            "\(filterName) filter failed: Unsupported type for key '\(key)'",
                            file: file,
                            line: line
                        )
                    }
                }
            } else if let expectedArray = expected as? [(String, Any)] {
                guard let resultArray = (result as? ArrayValue)?.value as? [ArrayValue] else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [(String, Any)], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }

                XCTAssertEqual(
                    resultArray.count,
                    expectedArray.count,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )

                for (index, expectedTuple) in expectedArray.enumerated() {
                    let resultTuple = resultArray[index].value

                    guard resultTuple.count == 2 else {
                        XCTFail(
                            "\(filterName) filter failed at index \(index): Result tuple does not have 2 elements",
                            file: file,
                            line: line
                        )
                        return
                    }

                    XCTAssertEqual(
                        (resultTuple[0] as? StringValue)?.value,
                        expectedTuple.0,
                        "\(filterName) filter failed at index \(index)",
                        file: file,
                        line: line
                    )

                    if let expectedInt = expectedTuple.1 as? Int {
                        XCTAssertEqual(
                            ((resultTuple[1] as? NumericValue)?.value as? Int),
                            expectedInt,
                            "\(filterName) filter failed at index \(index)",
                            file: file,
                            line: line
                        )
                    } else if let expectedString = expectedTuple.1 as? String {
                        XCTAssertEqual(
                            (resultTuple[1] as? StringValue)?.value,
                            expectedString,
                            "\(filterName) filter failed at index \(index)",
                            file: file,
                            line: line
                        )
                    } else {
                        XCTFail(
                            "\(filterName) filter failed: Unsupported type for tuple element at index \(index)",
                            file: file,
                            line: line
                        )
                    }
                }
            } else if let expectedMixedArray = expected as? [Any] {
                guard let resultArray = (result as? ArrayValue)?.value else {
                    XCTFail(
                        "\(filterName) filter failed: Expected [Any], got \(type(of: result))",
                        file: file,
                        line: line
                    )
                    return
                }

                // Convert both arrays to strings for comparison since they may contain mixed types
                let resultStrings = resultArray.map { value -> String in
                    if let arrayValue = value as? ArrayValue {
                        return "["
                            + arrayValue.value.map {
                                if let strValue = $0 as? StringValue {
                                    return strValue.value
                                }
                                return String(describing: $0)
                            }.joined(separator: ", ") + "]"
                    } else if let stringValue = value as? StringValue {
                        return stringValue.value
                    } else {
                        return String(describing: value)
                    }
                }

                let expectedStrings = expectedMixedArray.map { value -> String in
                    if let array = value as? [String] {
                        return "[" + array.joined(separator: ", ") + "]"
                    } else {
                        return String(describing: value)
                    }
                }

                XCTAssertEqual(
                    resultStrings,
                    expectedStrings,
                    "\(filterName) filter failed",
                    file: file,
                    line: line
                )
            } else if let expectedGroups = expected as? [[String: Any]] {
                // For "groupby" filter
                // Convert both expected and actual results to JSON strings for comparison
                let expectedJSON = try toJSON(try env.convertToRuntimeValues(input: expectedGroups))
                let actualJSON = try toJSON(result)

                XCTAssertEqual(
                    expectedJSON,
                    actualJSON,
                    "\(filterName) filter failed: Expected \(expectedJSON) but got \(actualJSON)",
                    file: file,
                    line: line
                )
            } else {
                XCTFail(
                    "\(filterName) filter failed: Unsupported expected type \(type(of: expected))",
                    file: file,
                    line: line
                )
            }
        }

        // Test abs
        try runTest(filterName: "abs", input: -1, expected: 1)
        try runTest(filterName: "abs", input: 1, expected: 1)
        try runTest(filterName: "abs", input: -3.14, expected: 3.14)
        try runTest(filterName: "abs", input: 3.14, expected: 3.14)

        // Test attr
        try runTest(
            filterName: "attr",
            input: ["name": "John"],
            args: ["name"],
            expected: "John"
        )
        try runTest(
            filterName: "attr",
            input: ["age": 30],
            args: ["age"],
            expected: 30
        )
        try runTest(
            filterName: "attr",
            input: ["name": "John"],
            args: ["age"],
            expected: UndefinedValue()
        )

        // Test batch
        try runTest(
            filterName: "batch",
            input: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            args: [3],
            expected: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )
        try runTest(
            filterName: "batch",
            input: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            args: [3, 0],
            expected: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )
        try runTest(
            filterName: "batch",
            input: [1, 2, 3, 4, 5, 6, 7, 8],
            args: [3, 0],
            expected: [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        )

        // Test capitalize
        try runTest(filterName: "capitalize", input: "foo bar", expected: "Foo bar")
        try runTest(filterName: "capitalize", input: "FOO BAR", expected: "Foo bar")

        // Test center
        try runTest(
            filterName: "center",
            input: "foo",
            expected: String(repeating: " ", count: 38) + "foo" + String(repeating: " ", count: 39)
        )  // Default width 80
        try runTest(filterName: "center", input: "foo", args: [NumericValue(value: 11)], expected: "    foo    ")
        try runTest(filterName: "center", input: "foo", args: [NumericValue(value: 5)], expected: " foo ")
        try runTest(filterName: "center", input: "foo", args: [NumericValue(value: 4)], expected: "foo ")
        try runTest(filterName: "center", input: "foo", args: [NumericValue(value: 3)], expected: "foo")
        try runTest(filterName: "center", input: "foo", args: [NumericValue(value: 2)], expected: "foo")

        // Test count
        try runTest(filterName: "count", input: "Hello", expected: 5)
        try runTest(filterName: "count", input: [1, 2, 3].map { NumericValue(value: $0) }, expected: 3)
        try runTest(
            filterName: "count",
            input: ObjectValue(value: ["name": StringValue(value: "John"), "age": NumericValue(value: 30)]),
            expected: 2
        )

        // Test default
        try runTest(filterName: "default", input: UndefinedValue(), expected: "")
        try runTest(filterName: "default", input: UndefinedValue(), args: ["foo"], expected: "foo")
        try runTest(filterName: "default", input: false, args: ["foo", true], expected: "foo")
        try runTest(filterName: "default", input: true, args: ["foo", true], expected: "true")
        try runTest(filterName: "default", input: "bar", args: ["foo"], expected: "bar")

        // Test dictsort
        try runTest(
            filterName: "dictsort",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)),
            expected: [("a", 1), ("b", 4), ("c", 3), ("d", 2), ("f", 5)]
        )
        try runTest(
            filterName: "dictsort",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)),
            args: [true],
            expected: [("a", 1), ("b", 4), ("c", 3), ("d", 2), ("f", 5)]
        )
        try runTest(
            filterName: "dictsort",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)),
            args: [false, "value"],
            expected: [("a", 1), ("d", 2), ("c", 3), ("b", 4), ("f", 5)]
        )
        try runTest(
            filterName: "dictsort",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)),
            args: [false, "key", true],
            expected: [("f", 5), ("d", 2), ("c", 3), ("b", 4), ("a", 1)]
        )
        try runTest(
            filterName: "dictsort",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)),
            args: [false, "value", true],
            expected: [("f", 5), ("b", 4), ("c", 3), ("d", 2), ("a", 1)]
        )

        // Test escape
        try runTest(filterName: "escape", input: "<foo>", expected: "&lt;foo&gt;")
        try runTest(filterName: "escape", input: "foo & bar", expected: "foo &amp; bar")

        // Test filesizeformat
        try runTest(filterName: "filesizeformat", input: 100, expected: "100 Bytes")
        try runTest(filterName: "filesizeformat", input: 1000, expected: "1.0 kB")
        try runTest(filterName: "filesizeformat", input: 1_000_000, expected: "1.0 MB")
        try runTest(filterName: "filesizeformat", input: 1_000_000_000, expected: "1.0 GB")
        try runTest(
            filterName: "filesizeformat",
            input: 1_000_000_000_000,
            expected: "1.0 TB"
        )
        try runTest(filterName: "filesizeformat", input: 300, expected: "300 Bytes")
        try runTest(filterName: "filesizeformat", input: 3000, expected: "3.0 kB")
        try runTest(filterName: "filesizeformat", input: 3_000_000, expected: "3.0 MB")
        try runTest(filterName: "filesizeformat", input: 3_000_000_000, expected: "3.0 GB")
        try runTest(
            filterName: "filesizeformat",
            input: 3_000_000_000_000,
            expected: "3.0 TB"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 100,
            args: [true],
            expected: "100 Bytes"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 1000,
            args: [true],
            expected: "1000 Bytes"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 1_000_000,
            args: [true],
            expected: "976.6 KiB"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 1_000_000_000,
            args: [true],
            expected: "953.7 MiB"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 1_000_000_000_000,
            args: [true],
            expected: "931.3 GiB"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 300,
            args: [true],
            expected: "300 Bytes"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 3000,
            args: [true],
            expected: "2.9 KiB"
        )
        try runTest(
            filterName: "filesizeformat",
            input: 3_000_000,
            args: [true],
            expected: "2.9 MiB"
        )

        // Test first
        try runTest(filterName: "first", input: [1, 2, 3], expected: 1)
        try runTest(filterName: "first", input: ["a", "b", "c"], expected: "a")
        try runTest(filterName: "first", input: [], expected: UndefinedValue())

        // Test float
        try runTest(filterName: "float", input: 42, expected: 42.0)
        try runTest(filterName: "float", input: 42.5, expected: 42.5)
        try runTest(filterName: "float", input: "42", expected: 0.0)
        try runTest(filterName: "float", input: "42.5", expected: 0.0)

        // Test forceescape
        try runTest(filterName: "forceescape", input: "<foo>", expected: "&lt;foo&gt;")
        try runTest(filterName: "forceescape", input: "foo & bar", expected: "foo &amp; bar")

        // Test format
        try runTest(filterName: "format", input: "%s %s", args: ["Hello", "World"], expected: "Hello World")
        try runTest(filterName: "format", input: "%d", args: [123], expected: "123")

        // TODO: Test groupby

        // Test indent
        try runTest(
            filterName: "indent",
            input: "Hello\nWorld",
            expected: "Hello\n    World"
        )  // Default: width=4, first=false, blank=false
        try runTest(
            filterName: "indent",
            input: "Hello\nWorld",
            args: [2],
            expected: "Hello\n  World"
        )  // width=2
        try runTest(
            filterName: "indent",
            input: "Hello\nWorld",
            args: [4, true],
            expected: "    Hello\n    World"
        )  // first=true
        try runTest(
            filterName: "indent",
            input: "\nfoo bar\n\"baz\"\n",
            args: [2, false, false],
            expected: "\n  foo bar\n  \"baz\"\n"
        )  // blank=false
        try runTest(
            filterName: "indent",
            input: "\nfoo bar\n\"baz\"\n",
            args: [2, false, true],
            expected: "\n  foo bar\n  \"baz\"\n  "
        )  // blank=true
        try runTest(
            filterName: "indent",
            input: "\nfoo bar\n\"baz\"\n",
            args: [2, true, false],
            expected: "  \n  foo bar\n  \"baz\"\n"
        )  // first=true, blank=false
        try runTest(
            filterName: "indent",
            input: "\nfoo bar\n\"baz\"\n",
            args: [2, true, true],
            expected: "  \n  foo bar\n  \"baz\"\n  "
        )  // first=true, blank=true
        try runTest(
            filterName: "indent",
            input: "jinja",
            expected: "jinja"
        )  // Single line, no indent
        try runTest(
            filterName: "indent",
            input: "jinja",
            args: [4, true],
            expected: "    jinja"
        )  // Single line, first=true
        try runTest(
            filterName: "indent",
            input: "jinja",
            args: [4, false, true],
            expected: "jinja"
        )  // Single line, blank=true (no effect)
        try runTest(
            filterName: "indent",
            input: "jinja\nflask",
            args: [">>> ", true],
            expected: ">>> jinja\n>>> flask"
        )  // String width, first=true

        // Test int
        try runTest(filterName: "int", input: 42.0, expected: 42)
        try runTest(filterName: "int", input: 42.5, expected: 42)
        try runTest(filterName: "int", input: "42", expected: 42)
        try runTest(filterName: "int", input: "42.5", expected: 42)

        // Test items
        // Test with dictionary
        try runTest(
            filterName: "items",
            input: OrderedDictionary<String, Any>(
                dictionaryLiteral: ("0", "a"),
                ("1", "b"),
                ("2", "c")
            ),
            expected: [
                ("0", "a"),
                ("1", "b"),
                ("2", "c"),
            ]
        )
        // Test with undefined value
        try runTest(
            filterName: "items",
            input: UndefinedValue(),
            expected: []
        )
        // Test with invalid input (should throw error)
        XCTAssertThrowsError(
            try runTest(
                filterName: "items",
                input: [1, 2, 3],  // Array instead of mapping
                expected: []
            )
        ) { error in
            XCTAssertEqual(
                error as? JinjaError,
                .runtime("Can only get item pairs from a mapping.")
            )
        }

        // Test join
        try runTest(filterName: "join", input: [1, 2, 3], expected: "123")
        try runTest(filterName: "join", input: [1, 2, 3], args: [","], expected: "1,2,3")
        try runTest(filterName: "join", input: ["a", "b", "c"], args: ["-"], expected: "a-b-c")

        // Test last
        try runTest(filterName: "last", input: [1, 2, 3], expected: 3)
        try runTest(filterName: "last", input: ["a", "b", "c"], expected: "c")
        try runTest(filterName: "last", input: [], expected: UndefinedValue())

        // Test length
        try runTest(filterName: "length", input: "Hello", expected: 5)
        try runTest(filterName: "length", input: [1, 2, 3], expected: 3)
        try runTest(filterName: "length", input: ["name": "John", "age": 30], expected: 2)

        // Test list
        try runTest(filterName: "list", input: [1, 2, 3], expected: [1, 2, 3])
        try runTest(filterName: "list", input: ["a", "b", "c"], expected: ["a", "b", "c"])

        // Test lower
        try runTest(filterName: "lower", input: "FOO", expected: "foo")
        try runTest(filterName: "lower", input: "Foo", expected: "foo")

        // Test map
        // Test simple map with int conversion
        try runTest(
            filterName: "map",
            input: ["1", "2", "3"],
            args: [StringValue(value: "int")],
            expected: [1, 2, 3]
        )

        // TODO: Test `map` with `sum` (currently failing, may require changes to `map` or `sum`)
        //      try runFilterTest(
        //        filterName: "map",
        //        input: [[1, 2], [3], [4, 5, 6]],
        //        args: [StringValue(value: "sum")],
        //        expected: [3, 3, 15]
        //      )

        // Test attribute map
        try runTest(
            filterName: "map",
            input: [
                ["username": "john"],
                ["username": "jane"],
                ["username": "mike"],
            ],
            args: [
                ObjectValue(value: [
                    "attribute": StringValue(value: "username")
                ])
            ],
            expected: ["john", "jane", "mike"]
        )

        // Test map with default value
        try runTest(
            filterName: "map",
            input: [
                ["firstname": "john", "lastname": "lennon"],
                ["firstname": "jane", "lastname": "edwards"],
                ["firstname": "jon", "lastname": UndefinedValue()],
                ["firstname": "mike"],
            ],
            args: [
                ObjectValue(value: [
                    "attribute": StringValue(value: "lastname"),
                    "default": StringValue(value: "smith"),
                ])
            ],
            expected: ["lennon", "edwards", "None", "smith"]
        )

        // Test map with list default value
        try runTest(
            filterName: "map",
            input: [
                ["firstname": "john", "lastname": "lennon"],
                ["firstname": "jane", "lastname": "edwards"],
                ["firstname": "jon", "lastname": UndefinedValue()],
                ["firstname": "mike"],
            ],
            args: [
                ObjectValue(value: [
                    "attribute": StringValue(value: "lastname"),
                    "default": ArrayValue(value: [
                        StringValue(value: "smith"),
                        StringValue(value: "x"),
                    ]),
                ])
            ],
            expected: ["lennon", "edwards", "None", ["smith", "x"]]
        )

        // Test map with empty string default value
        try runTest(
            filterName: "map",
            input: [
                ["firstname": "john", "lastname": "lennon"],
                ["firstname": "jane", "lastname": "edwards"],
                ["firstname": "jon", "lastname": UndefinedValue()],
                ["firstname": "mike"],
            ],
            args: [
                ObjectValue(value: [
                    "attribute": StringValue(value: "lastname"),
                    "default": StringValue(value: ""),
                ])
            ],
            expected: ["lennon", "edwards", "None", ""]
        )

        // Test min
        try runTest(filterName: "min", input: [3, 1, 4, 2], expected: 1)
        try runTest(filterName: "min", input: ["b", "a", "d", "c"], expected: "a")
        try runTest(filterName: "min", input: [], expected: UndefinedValue())

        // Test max
        try runTest(filterName: "max", input: [3, 1, 4, 2], expected: 4)
        try runTest(filterName: "max", input: ["b", "a", "d", "c"], expected: "d")
        try runTest(filterName: "max", input: [], expected: UndefinedValue())

        // TODO: Figure out how to test "pprint", given that Swift 5.10 doesn't preserve the key order in dictionaries

        // TODO: Figure out how to test "random" filter

        // Test reject
        try runTest(
            filterName: "reject",
            input: [1, 2, 3, 4, 5],
            args: ["even"],
            expected: [1, 3, 5]
        )

        // TODO: Test rejectattr
        //        try runFilterTest(
        //            filterName: "rejectattr",
        //            input: [
        //                ["name": "John", "admin": true],
        //                ["name": "Jane", "admin": false],
        //            ],
        //            args: ["admin"],
        //            expected: [
        //                ["admin": false, "name": "Jane"]
        //            ]
        //        )

        // Test replace
        try runTest(
            filterName: "replace",
            input: "Hello World",
            args: ["World", "Jinja"],
            expected: "Hello Jinja"
        )
        try runTest(
            filterName: "replace",
            input: "aaaa",
            args: ["a", "b", 2],
            expected: "bbbb"
        )

        // Test reverse
        try runTest(filterName: "reverse", input: [1, 2, 3], expected: [3, 2, 1])
        try runTest(filterName: "reverse", input: ["a", "b", "c"], expected: ["c", "b", "a"])

        // Test round
        try runTest(filterName: "round", input: 42.55, expected: 43.0)
        try runTest(filterName: "round", input: 42.55, args: [1], expected: 42.6)
        try runTest(filterName: "round", input: 42.55, args: [1, "floor"], expected: 42.5)
        try runTest(filterName: "round", input: 42.55, args: [1, "ceil"], expected: 42.6)

        // Test safe
        try runTest(filterName: "safe", input: "<foo>", expected: "<foo>")
        try runTest(filterName: "safe", input: "foo & bar", expected: "foo & bar")

        // Test select
        try runTest(
            filterName: "select",
            input: [1, 2, 3, 4, 5],
            args: ["even"],
            expected: [2, 4]
        )
        // TODO: Make this test pass
        //        try runFilterTest(
        //            filterName: "select",
        //            input: [
        //                ["name": "John", "age": 30],
        //                ["name": "Jane", "age": 25],
        //            ],
        //            args: ["even"],
        //            expected: [["name": "John", "age": 30]]
        //        )

        // TODO: Test selectattr
        //        try runFilterTest(
        //            filterName: "selectattr",
        //            input: [
        //                ["name": "John", "admin": true],
        //                ["name": "Jane", "admin": false],
        //            ],
        //            args: ["admin"],
        //            expected: [["name": "John", "admin": true]]
        //        )
        //        try runFilterTest(
        //            filterName: "selectattr",
        //            input: [
        //                ["name": "John", "age": 30],
        //                ["name": "Jane", "age": 25],
        //            ],
        //            args: ["age", "equalto", 25],
        //            expected: [["name": "Jane", "age": 25]]
        //        )

        // Test slice
        try runTest(
            filterName: "slice",
            input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            args: [3],
            expected: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
        )
        try runTest(
            filterName: "slice",
            input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            args: [3, 0],
            expected: [[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 10, 0]]
        )

        // Test sort
        try runTest(filterName: "sort", input: [3, 1, 4, 2], expected: [1, 2, 3, 4])
        try runTest(filterName: "sort", input: [3, 1, 4, 2], args: [true], expected: [4, 3, 2, 1])
        try runTest(
            filterName: "sort",
            input: ["b", "A", "d", "c"],
            expected: ["A", "b", "c", "d"]
        )
        try runTest(
            filterName: "sort",
            input: ["b", "A", "d", "c"],
            args: [false, true],
            expected: ["A", "b", "c", "d"]
        )
        // TODO: Make these tests pass
        //        try runFilterTest(
        //            filterName: "sort",
        //            input: [
        //                ["name": "John", "age": 30],
        //                ["name": "Jane", "age": 25],
        //            ],
        //            args: [false, false, "name"],
        //            expected: [
        //                ["name": "Jane", "age": 25],
        //                ["name": "John", "age": 30],
        //            ]
        //        )
        //        try runFilterTest(
        //            filterName: "sort",
        //            input: [
        //                ["name": "John", "age": 30],
        //                ["name": "Jane", "age": 25],
        //            ],
        //            args: [false, false, "age"],
        //            expected: [
        //                ["name": "Jane", "age": 25],
        //                ["name": "John", "age": 30],
        //            ]
        //        )

        // Test string
        try runTest(filterName: "string", input: 123, expected: "123")
        try runTest(filterName: "string", input: true, expected: "true")
        try runTest(filterName: "string", input: [1, 2, 3], expected: "[1, 2, 3]")
        try runTest(
            filterName: "string",
            input: OrderedDictionary<String, Any>(dictionaryLiteral: ("a", 1), ("b", 2)),
            expected: "{\"a\": 1, \"b\": 2}"
        )

        // Test striptags
        try runTest(
            filterName: "striptags",
            input: "<p>Hello, <b>World</b>!</p>",
            expected: "Hello, World!"
        )
        try runTest(
            filterName: "striptags",
            input: "<a href=\"#\">Link</a>",
            expected: "Link"
        )

        // Test sum
        try runTest(filterName: "sum", input: [1, 2, 3, 4, 5], expected: 15)
        try runTest(
            filterName: "sum",
            input: [
                ["value": 1],
                ["value": 2],
                ["value": 3],
            ],
            args: ["value"],
            expected: 6
        )
        try runTest(filterName: "sum", input: [1, 2, 3, 4, 5], args: [], expected: 15)
        // TODO: Make this test pass
        //        try runFilterTest(filterName: "sum", input: [1, 2, 3, 4, 5], args: ["", 10], expected: 25)

        // Test title
        try runTest(filterName: "title", input: "hello world", expected: "Hello World")
        try runTest(filterName: "title", input: "HELLO WORLD", expected: "Hello World")

        // Test trim
        try runTest(filterName: "trim", input: "  hello   ", expected: "hello")
        try runTest(filterName: "trim", input: "\t  hello \n  ", expected: "hello")

        // Test truncate
        try runTest(filterName: "truncate", input: "Hello World", expected: "Hello World")
        try runTest(filterName: "truncate", input: "Hello World", args: [5], expected: "He...")
        try runTest(filterName: "truncate", input: "Hello World", args: [5, true], expected: "He...")
        try runTest(filterName: "truncate", input: "Hello World", args: [5, false], expected: "He...")
        try runTest(filterName: "truncate", input: "Hello World", args: [5, false, "---"], expected: "He---")
        try runTest(filterName: "truncate", input: "Hello Big World", args: [10, false], expected: "Hello...")

        // Test unique
        try runTest(filterName: "unique", input: [2, 1, 2, 3, 4, 4], expected: [2, 1, 3, 4])
        try runTest(filterName: "unique", input: ["Foo", "foo", "bar"], expected: ["Foo", "bar"])
        try runTest(
            filterName: "unique",
            input: ["Foo", "foo", "bar"],
            args: [true],
            expected: ["Foo", "foo", "bar"]
        )
        // TODO: Make these tests pass
        //        try runFilterTest(
        //            filterName: "unique",
        //            input: [
        //                ["name": "foo", "id": 1],
        //                ["name": "foo", "id": 2],
        //                ["name": "bar", "id": 3],
        //            ],
        //            args: [false, "name"],
        //            expected: [["name": "foo", "id": 1], ["name": "bar", "id": 3]]
        //        )
        //        try runFilterTest(
        //            filterName: "unique",
        //            input: [
        //                ["name": "foo", "id": 1],
        //                ["name": "foo", "id": 2],
        //                ["name": "bar", "id": 3],
        //            ],
        //            args: [false, "id"],
        //            expected: [["name": "foo", "id": 1], ["name": "bar", "id": 3]]
        //        )
        try runTest(
            filterName: "unique",
            input: "abcba",
            expected: ["a", "b", "c"]
        )
        try runTest(  //XCTAssertEqual failed: ("["a"]") is not equal to ("["a", "b", "c"]") - unique filter failed
            filterName: "unique",
            input: "abcba",
            args: [false, 0],
            expected: ["a", "b", "c"]
        )

        // Test upper
        try runTest(filterName: "upper", input: "foo", expected: "FOO")
        try runTest(filterName: "upper", input: "Foo", expected: "FOO")

        // TODO: Test urlencode

        // Test urlize
        try runTest(
            filterName: "urlize",
            input: "http://www.example.com/",
            expected: "<a href=\"http://www.example.com/\">http://www.example.com/</a>"
        )
        try runTest(
            filterName: "urlize",
            input: "www.example.com",
            expected: "<a href=\"www.example.com\">www.example.com</a>"
        )
        try runTest(
            filterName: "urlize",
            input: "http://www.example.com/",
            args: [10],
            expected: "<a href=\"http://www.example.com/\">http://www...</a>"
        )
        try runTest(
            filterName: "urlize",
            input: "http://www.example.com/",
            args: [10, true],
            expected: "<a href=\"http://www.example.com/\" rel=\"nofollow\">http://www...</a>"
        )
        // TODO: Make this test pass
        //        try runFilterTest(
        //            filterName: "urlize",
        //            input: "http://www.example.com/",
        //            args: [10, true, "_blank"],
        //            expected: "<a href=\"http://www.example.com/\" target=\"_blank\" rel=\"nofollow\">http://www...</a>"
        //        )

        // Test wordcount
        try runTest(filterName: "wordcount", input: "foo bar baz", expected: 3)
        try runTest(filterName: "wordcount", input: "foo  bar baz", expected: 3)

        // TODO: Test wordwrap

        // TODO: Test xmlattr

        // TODO: Fix key order in results using OrderedDictionary as input
        // Test tojson
        //        try runFilterTest(
        //            filterName: "tojson",
        //            input: ["foo": 42, "bar": 23],
        //            expected: "{\n  \"foo\": 42,\n  \"bar\": 23\n}"
        //        )
        //        try runFilterTest(
        //            filterName: "tojson",
        //            input: ["foo": 42, "bar": 23],
        //            args: [["indent": 4]],
        //            expected: "{\n    \"foo\": 42,\n    \"bar\": 23\n}"
        //        )
    }
}
