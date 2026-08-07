/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class FunctionCallTests: SyntaxHighlighterTestCase {
    func testFunctionCallWithIntegers() {
        let components = highlighter.highlight("add(1, 2)")

        XCTAssertEqual(components, [
            .token("add", .call),
            .plainText("("),
            .token("1", .number),
            .plainText(","),
            .whitespace(" "),
            .token("2", .number),
            .plainText(")")
        ])
    }

    func testFunctionCallWithNil() {
        let components = highlighter.highlight("handler(nil)")

        XCTAssertEqual(components, [
            .token("handler", .call),
            .plainText("("),
            .token("nil", .keyword),
            .plainText(")")
        ])
    }

    func testImplicitInitializerCall() {
        let components = highlighter.highlight("let string = String()")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("string"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("String", .type),
            .plainText("()")
        ])
    }

    func testExplicitInitializerCall() {
        let components = highlighter.highlight("let string = String.init()")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("string"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("String", .type),
            .plainText("."),
            .token("init", .keyword),
            .plainText("()")
        ])
    }

    func testExplicitInitializerCallUsingTrailingClosureSyntax() {
        let components = highlighter.highlight("let task = Task.init {}")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("task"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("Task", .type),
            .plainText("."),
            .token("init", .keyword),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testDotSyntaxInitializerCall() {
        let components = highlighter.highlight("let string: String = .init()")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("string:"),
            .whitespace(" "),
            .token("String", .type),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("."),
            .token("init", .keyword),
            .plainText("()")
        ])
    }

    func testAccessingPropertyAfterFunctionCallWithoutArguments() {
        let components = highlighter.highlight("call().property")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("()."),
            .token("property", .property)
        ])
    }

    func testAccessingPropertyAfterFunctionCallWithArguments() {
        let components = highlighter.highlight("call(argument).property")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("(argument)."),
            .token("property", .property)
        ])
    }

    func testCallingStaticMethodOnGenericType() {
        let components = highlighter.highlight("Array<String>.call()")

        XCTAssertEqual(components, [
            .token("Array", .type),
            .plainText("<"),
            .token("String", .type),
            .plainText(">."),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testPassingTypeToFunction() {
        let components = highlighter.highlight("call(String.self)")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("("),
            .token("String", .type),
            .plainText("."),
            .token("self", .keyword),
            .plainText(")")
        ])
    }

    func testPassingBoolToUnnamedArgument() {
        let components = highlighter.highlight("setCachingEnabled(true)")

        XCTAssertEqual(components, [
            .token("setCachingEnabled", .call),
            .plainText("("),
            .token("true", .keyword),
            .plainText(")")
        ])
    }

    func testIndentedFunctionCalls() {
        let components = highlighter.highlight("""
        variable
            .callOne()
            .callTwo()
        """)

        XCTAssertEqual(components, [
            .plainText("variable"),
            .whitespace("\n    "),
            .plainText("."),
            .token("callOne", .call),
            .plainText("()"),
            .whitespace("\n    "),
            .plainText("."),
            .token("callTwo", .call),
            .plainText("()")
        ])
    }

    func testXCTAssertCalls() {
        let components = highlighter.highlight("XCTAssertTrue(variable)")

        XCTAssertEqual(components, [
            .token("XCTAssertTrue", .call),
            .plainText("(variable)")
        ])
    }

    func testUsingTryKeywordWithinFunctionCall() {
        let components = highlighter.highlight("XCTAssertThrowsError(try function())")

        XCTAssertEqual(components, [
            .token("XCTAssertThrowsError", .call),
            .plainText("("),
            .token("try", .keyword),
            .whitespace(" "),
            .token("function", .call),
            .plainText("())")
        ])
    }

    func testCallingFunctionsWithProjectedPropertyWrapperValues() {
        let components = highlighter.highlight("""
        call($value)
        call(self.$value)
        """)

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("("),
            .token("$value", .property),
            .plainText(")"),
            .whitespace("\n"),
            .token("call", .call),
            .plainText("("),
            .token("self", .keyword),
            .plainText("."),
            .token("$value", .property),
            .plainText(")")
        ])
    }

    func testCallingFunctionWithInoutProjectedPropertyWrapperValue() {
        let components = highlighter.highlight("call(&$value)")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("(&"),
            .token("$value", .property),
            .plainText(")")
        ])
    }

    func testCallingMethodWithSameNameAsKeywordWithTrailingClosureSyntax() {
        let components = highlighter.highlight("publisher.catch { error in }")

        XCTAssertEqual(components, [
            .plainText("publisher."),
            .token("catch", .call),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("error"),
            .whitespace(" "),
            .token("in", .keyword),
            .whitespace(" "),
            .plainText("}")
        ])
    }
}
