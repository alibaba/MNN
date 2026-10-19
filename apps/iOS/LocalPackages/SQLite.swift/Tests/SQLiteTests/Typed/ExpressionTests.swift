import XCTest
@testable import SQLite

class ExpressionTests: XCTestCase {

    func test_asSQL_expression_bindings() {
        let expression = Expression<String>("foo ? bar", ["baz"])
        XCTAssertEqual(expression.asSQL(), "foo 'baz' bar")
    }

    func test_asSQL_expression_bindings_quoting() {
        let expression = Expression<String>("foo ? bar", ["'baz'"])
        XCTAssertEqual(expression.asSQL(), "foo '''baz''' bar")
    }

    func test_expression_custom_string_convertible() {
        let expression = Expression<String>("foo ? bar", ["baz"])
        XCTAssertEqual(expression.asSQL(), expression.description)
    }

    func test_builtin_unambiguously_custom_string_convertible() {
        let integer: Int = 45
        XCTAssertEqual(integer.description, "45")
    }

    func test_init_literal() {
        let expression = Expression<String>(literal: "literal")
        XCTAssertEqual(expression.template, "literal")
    }

    func test_init_identifier() {
        let expression = Expression<String>("identifier")
        XCTAssertEqual(expression.template, "\"identifier\"")
    }
}
