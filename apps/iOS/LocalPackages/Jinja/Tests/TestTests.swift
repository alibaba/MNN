//
//  TestTests.swift
//  Jinja
//
//  Created by Anthony DePasquale on 07.01.2025.
//

// Adapted from https://github.com/pallets/jinja/blob/main/tests/test_tests.py

import XCTest
@testable import Jinja

final class TestTests: XCTestCase {
    func testTests() throws {
        // Helper function to run tests
        func runTest(
            testName: String,
            input: Any,
            args: [Any?] = [],
            expected: Bool,
            file: StaticString = #file,
            line: UInt = #line
        ) throws {
            let env = Environment()

            // Convert input to RuntimeValue
            guard let input = try? env.convertToRuntimeValues(input: input) else {
                XCTFail(
                    "Failed to convert input \(input) to RuntimeValue in test for \(testName)",
                    file: file,
                    line: line
                )
                return
            }

            // Convert args to RuntimeValues
            let runtimeArgs = try args.map { arg -> any RuntimeValue in
                if let arg = arg {
                    return try env.convertToRuntimeValues(input: arg)
                }
                return UndefinedValue()
            }

            // Get the test function from the environment
            guard let test = env.tests[testName] else {
                XCTFail("Test not found: \(testName)", file: file, line: line)
                return
            }

            // Call the test function with input and arguments
            let result = try test([input] + runtimeArgs)

            XCTAssertEqual(result, expected, "\(testName) test failed", file: file, line: line)
        }

        // Test defined
        try runTest(testName: "defined", input: UndefinedValue(), expected: false)
        try runTest(testName: "defined", input: true, expected: true)

        // Test even/odd
        try runTest(testName: "even", input: 1, expected: false)
        try runTest(testName: "even", input: 2, expected: true)
        try runTest(testName: "odd", input: 1, expected: true)
        try runTest(testName: "odd", input: 2, expected: false)

        // Test lower/upper
        try runTest(testName: "lower", input: "foo", expected: true)
        try runTest(testName: "lower", input: "FOO", expected: false)
        try runTest(testName: "upper", input: "FOO", expected: true)
        try runTest(testName: "upper", input: "foo", expected: false)

        // Test type checks
        try runTest(testName: "none", input: NullValue(), expected: true)
        try runTest(testName: "none", input: false, expected: false)
        try runTest(testName: "none", input: true, expected: false)
        try runTest(testName: "none", input: 42, expected: false)

        try runTest(testName: "boolean", input: false, expected: true)
        try runTest(testName: "boolean", input: true, expected: true)
        try runTest(testName: "boolean", input: 0, expected: false)
        try runTest(testName: "boolean", input: 1, expected: false)

        try runTest(testName: "false", input: false, expected: true)
        try runTest(testName: "false", input: true, expected: false)
        try runTest(testName: "true", input: true, expected: true)
        try runTest(testName: "true", input: false, expected: false)

        try runTest(testName: "integer", input: 42, expected: true)
        try runTest(testName: "integer", input: 3.14159, expected: false)
        try runTest(testName: "float", input: 3.14159, expected: true)
        try runTest(testName: "float", input: 42, expected: false)

        try runTest(testName: "string", input: "foo", expected: true)
        try runTest(testName: "string", input: 42, expected: false)

        try runTest(testName: "sequence", input: [1, 2, 3], expected: true)
        try runTest(testName: "sequence", input: "foo", expected: true)
        try runTest(testName: "sequence", input: 42, expected: false)

        try runTest(testName: "mapping", input: ["foo": "bar"], expected: true)
        try runTest(testName: "mapping", input: [1, 2, 3], expected: false)

        try runTest(testName: "number", input: 42, expected: true)
        try runTest(testName: "number", input: 3.14159, expected: true)
        try runTest(testName: "number", input: "foo", expected: false)

        // Test equalto/eq
        try runTest(testName: "eq", input: 12, args: [12], expected: true)
        try runTest(testName: "eq", input: 12, args: [0], expected: false)
        try runTest(testName: "eq", input: "baz", args: ["baz"], expected: true)
        try runTest(testName: "eq", input: "baz", args: ["zab"], expected: false)

        // Test comparison aliases
        try runTest(testName: "ne", input: 2, args: [3], expected: true)
        try runTest(testName: "ne", input: 2, args: [2], expected: false)
        try runTest(testName: "lt", input: 2, args: [3], expected: true)
        try runTest(testName: "lt", input: 2, args: [2], expected: false)
        try runTest(testName: "le", input: 2, args: [2], expected: true)
        try runTest(testName: "le", input: 2, args: [1], expected: false)
        try runTest(testName: "gt", input: 2, args: [1], expected: true)
        try runTest(testName: "gt", input: 2, args: [2], expected: false)
        try runTest(testName: "ge", input: 2, args: [2], expected: true)
        try runTest(testName: "ge", input: 2, args: [3], expected: false)

        // Test in
        try runTest(testName: "in", input: "o", args: [["f", "o", "o"]], expected: true)
        try runTest(testName: "in", input: "foo", args: [["foo"]], expected: true)
        try runTest(testName: "in", input: "b", args: [["f", "o", "o"]], expected: false)
        try runTest(testName: "in", input: 1, args: [[1, 2]], expected: true)
        try runTest(testName: "in", input: 3, args: [[1, 2]], expected: false)

        // Test filter/test existence
        try runTest(testName: "filter", input: "title", expected: true)
        try runTest(testName: "filter", input: "bad-name", expected: false)
        try runTest(testName: "test", input: "number", expected: true)
        try runTest(testName: "test", input: "bad-name", expected: false)
    }
}
