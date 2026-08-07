import XCTest
@testable import SQLite

class RowTests: XCTestCase {

    public func test_get_value() throws {
        let row = Row(["\"foo\"": 0], ["value"])
        let result = try row.get(Expression<String>("foo"))

        XCTAssertEqual("value", result)
    }

    public func test_get_value_subscript() {
        let row = Row(["\"foo\"": 0], ["value"])
        let result = row[Expression<String>("foo")]

        XCTAssertEqual("value", result)
    }

    public func test_get_value_optional() throws {
        let row = Row(["\"foo\"": 0], ["value"])
        let result = try row.get(Expression<String?>("foo"))

        XCTAssertEqual("value", result)
    }

    public func test_get_value_optional_subscript() {
        let row = Row(["\"foo\"": 0], ["value"])
        let result = row[Expression<String?>("foo")]

        XCTAssertEqual("value", result)
    }

    public func test_get_value_optional_nil() throws {
        let row = Row(["\"foo\"": 0], [nil])
        let result = try row.get(Expression<String?>("foo"))

        XCTAssertNil(result)
    }

    public func test_get_value_optional_nil_subscript() {
        let row = Row(["\"foo\"": 0], [nil])
        let result = row[Expression<String?>("foo")]

        XCTAssertNil(result)
    }

    public func test_get_type_mismatch_throws_unexpected_null_value() {
        let row = Row(["\"foo\"": 0], ["value"])
        XCTAssertThrowsError(try row.get(Expression<Int>("foo"))) { error in
            if case QueryError.unexpectedNullValue(let name) = error {
                XCTAssertEqual("\"foo\"", name)
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }

    public func test_get_type_mismatch_optional_returns_nil() throws {
        let row = Row(["\"foo\"": 0], ["value"])
        let result = try row.get(Expression<Int?>("foo"))
        XCTAssertNil(result)
    }

    public func test_get_non_existent_column_throws_no_such_column() {
        let row = Row(["\"foo\"": 0], ["value"])
        XCTAssertThrowsError(try row.get(Expression<Int>("bar"))) { error in
            if case QueryError.noSuchColumn(let name, let columns) = error {
                XCTAssertEqual("\"bar\"", name)
                XCTAssertEqual(["\"foo\""], columns)
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }

    public func test_get_ambiguous_column_throws() {
        let row = Row(["table1.\"foo\"": 0, "table2.\"foo\"": 1], ["value"])
        XCTAssertThrowsError(try row.get(Expression<Int>("foo"))) { error in
            if case QueryError.ambiguousColumn(let name, let columns) = error {
                XCTAssertEqual("\"foo\"", name)
                XCTAssertEqual(["table1.\"foo\"", "table2.\"foo\""], columns.sorted())
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }

    public func test_get_datatype_throws() {
        // swiftlint:disable nesting
        struct MyType: Value {
            enum MyError: Error {
                case failed
            }

            public static var declaredDatatype: String {
                Blob.declaredDatatype
            }

            public static func fromDatatypeValue(_ dataValue: Blob) throws -> Data {
                throw MyError.failed
            }

            public var datatypeValue: Blob {
                return Blob(bytes: [])
            }
        }

        let row = Row(["\"foo\"": 0], [Blob(bytes: [])])
        XCTAssertThrowsError(try row.get(Expression<MyType>("foo"))) { error in
            if case MyType.MyError.failed = error {
                XCTAssertTrue(true)
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }
}
