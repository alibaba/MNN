/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class StatementTests: SyntaxHighlighterTestCase {
    func testImportStatement() {
        let components = highlighter.highlight("import UIKit")

        XCTAssertEqual(components, [
            .token("import", .keyword),
            .whitespace(" "),
            .plainText("UIKit")
        ])
    }

    func testImportStatementWithSubmodule() {
        let components = highlighter.highlight("import os.log")

        XCTAssertEqual(components, [
            .token("import", .keyword),
            .whitespace(" "),
            .plainText("os.log")
        ])
    }

    func testChainedIfElseStatements() {
        let components = highlighter.highlight("if condition { } else if call() { } else { \"string\" }")

        XCTAssertEqual(components, [
            .token("if", .keyword),
            .whitespace(" "),
            .plainText("condition"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace(" "),
            .token("else", .keyword),
            .whitespace(" "),
            .token("if", .keyword),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace(" "),
            .token("else", .keyword),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("\"string\"", .string),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testIfLetStatementWithKeywordSymbolName() {
        let components = highlighter.highlight("if let override = optional {}")

        XCTAssertEqual(components, [
            .token("if", .keyword),
            .whitespace(" "),
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("override"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("optional"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testGuardStatementUnwrappingWeakSelf() {
        let components = highlighter.highlight("guard let self = self else {}")

        XCTAssertEqual(components, [
            .token("guard", .keyword),
            .whitespace(" "),
            .token("let", .keyword),
            .whitespace(" "),
            .token("self", .keyword),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("self", .keyword),
            .whitespace(" "),
            .token("else", .keyword),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testSwitchStatement() {
        let components = highlighter.highlight("""
        switch variable {
        case .one: break
        case .two: callA()
        default:
            callB()
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("variable"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("."),
            .token("one", .dotAccess),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("."),
            .token("two", .dotAccess),
            .plainText(":"),
            .whitespace(" "),
            .token("callA", .call),
            .plainText("()"),
            .whitespace("\n"),
            .token("default", .keyword),
            .plainText(":"),
            .whitespace("\n    "),
            .token("callB", .call),
            .plainText("()"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithSingleAssociatedValue() {
        let components = highlighter.highlight("""
        switch value {
        case .one(let a): break
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("."),
            .token("one", .dotAccess),
            .plainText("("),
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("a):"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithMultipleAssociatedValues() {
        let components = highlighter.highlight("""
        switch value {
        case .one(let a), .two(let b): break
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("."),
            .token("one", .dotAccess),
            .plainText("("),
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("a),"),
            .whitespace(" "),
            .plainText("."),
            .token("two", .dotAccess),
            .plainText("("),
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("b):"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithFallthrough() {
        let components = highlighter.highlight("""
        switch variable {
        case .one: fallthrough
        default:
            callB()
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("variable"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("."),
            .token("one", .dotAccess),
            .plainText(":"),
            .whitespace(" "),
            .token("fallthrough", .keyword),
            .whitespace("\n"),
            .token("default", .keyword),
            .plainText(":"),
            .whitespace("\n    "),
            .token("callB", .call),
            .plainText("()"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithTypePatternMatching() {
        let components = highlighter.highlight("""
        switch variable {
        case is MyType: break
        default: break
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("variable"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .token("is", .keyword),
            .whitespace(" "),
            .token("MyType", .type),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .token("default", .keyword),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithOptional() {
        let components = highlighter.highlight("""
        switch anOptional {
        case nil: break
        case "value"?: break
        default: break
        }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("anOptional"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .token("nil", .keyword),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .token("case", .keyword),
            .whitespace(" "),
            .token("\"value\"", .string),
            .plainText("?:"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .token("default", .keyword),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSwitchStatementWithProperty() {
        let components = highlighter.highlight("""
        switch object.value { default: break }
        """)

        XCTAssertEqual(components, [
            .token("switch", .keyword),
            .whitespace(" "),
            .plainText("object."),
            .token("value", .property),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("default", .keyword),
            .plainText(":"),
            .whitespace(" "),
            .token("break", .keyword),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testForStatementWithStaticProperty() {
        let components = highlighter.highlight("for value in Enum.allCases { }")

        XCTAssertEqual(components, [
            .token("for", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .token("in", .keyword),
            .whitespace(" "),
            .token("Enum", .type),
            .plainText("."),
            .token("allCases", .property),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testForStatementWithContinue() {
        let components = highlighter.highlight("for value in Enum.allCases { continue }")

        XCTAssertEqual(components, [
            .token("for", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .token("in", .keyword),
            .whitespace(" "),
            .token("Enum", .type),
            .plainText("."),
            .token("allCases", .property),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("continue", .keyword),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testRepeatWhileStatement() {
        let components = highlighter.highlight("""
        var x = 5
        repeat {
            print(x)
            x = x - 1
        } while x > 1
        """)

        XCTAssertEqual(components, [
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("x"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("5", .number),
            .whitespace("\n"),
            .token("repeat", .keyword),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("print", .call),
            .plainText("(x)"),
            .whitespace("\n    "),
            .plainText("x"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("x"),
            .whitespace(" "),
            .plainText("-"),
            .whitespace(" "),
            .token("1", .number),
            .whitespace("\n"),
            .plainText("}"),
            .whitespace(" "),
            .token("while", .keyword),
            .whitespace(" "),
            .plainText("x"),
            .whitespace(" "),
            .plainText(">"),
            .whitespace(" "),
            .token("1", .number)
        ])
    }

    func testInitializingTypeWithLeadingUnderscore() {
        let components = highlighter.highlight("_MyType()")

        XCTAssertEqual(components, [
            .token("_MyType", .type),
            .plainText("()")
        ])
    }

    func testCallingFunctionWithLeadingUnderscore() {
        let components = highlighter.highlight("_myFunction()")

        XCTAssertEqual(components, [
            .token("_myFunction", .call),
            .plainText("()")
        ])
    }

    func testTernaryOperationContainingNil() {
        let components = highlighter.highlight("""
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        """)

        XCTAssertEqual(components, [
            .plainText("components."),
            .token("queryItems", .property),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("queryItems."),
            .token("isEmpty", .property),
            .whitespace(" "),
            .plainText("?"),
            .whitespace(" "),
            .token("nil", .keyword),
            .whitespace(" "),
            .plainText(":"),
            .whitespace(" "),
            .plainText("queryItems")
        ])
    }

    func testAwaitingFunctionCall() {
        let components = highlighter.highlight("let result = await call()")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("result"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("await", .keyword),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testAwaitingVariable() {
        let components = highlighter.highlight("let result = await value")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("result"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("await", .keyword),
            .whitespace(" "),
            .plainText("value")
        ])
    }

    func testAwaitingAsyncSequenceElement() {
        let components = highlighter.highlight("for await value in sequence {}")

        XCTAssertEqual(components, [
            .token("for", .keyword),
            .whitespace(" "),
            .token("await", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .token("in", .keyword),
            .whitespace(" "),
            .plainText("sequence"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testAwaitingThrowingAsyncSequenceElement() {
        let components = highlighter.highlight("for try await value in sequence {}")

        XCTAssertEqual(components, [
            .token("for", .keyword),
            .whitespace(" "),
            .token("try", .keyword),
            .whitespace(" "),
            .token("await", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .token("in", .keyword),
            .whitespace(" "),
            .plainText("sequence"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testAsyncLetExpression() {
        let components = highlighter.highlight("async let result = call()")

        XCTAssertEqual(components, [
            .token("async", .keyword),
            .whitespace(" "),
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("result"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()")
        ])
    }
}
