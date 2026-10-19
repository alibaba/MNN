/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class LiteralTests: SyntaxHighlighterTestCase {
    func testStringLiteral() {
        let components = highlighter.highlight("let string = \"Hello, world!\"")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("string"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("\"Hello,", .string),
            .whitespace(" "),
            .token("world!\"", .string)
        ])
    }

    func testStringLiteralPassedToFunction() {
        let components = highlighter.highlight("call(\"Hello, world!\")")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("("),
            .token("\"Hello,", .string),
            .whitespace(" "),
            .token("world!\"", .string),
            .plainText(")")
        ])
    }

    func testStringLiteralWithEscapedQuote() {
        let components = highlighter.highlight("\"Hello \\\" World\"; call()")

        XCTAssertEqual(components, [
            .token("\"Hello", .string),
            .whitespace(" "),
            .token("\\\"", .string),
            .whitespace(" "),
            .token("World\"", .string),
            .plainText(";"),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testStringLiteralWithAttribute() {
        let components = highlighter.highlight("\"@escaping\"")
        XCTAssertEqual(components, [.token("\"@escaping\"", .string)])
    }

    func testStringLiteralInterpolation() {
        let components = highlighter.highlight("\"Hello \\(variable) world \\(call())\"")

        XCTAssertEqual(components, [
            .token("\"Hello", .string),
            .whitespace(" "),
            .plainText("\\(variable)"),
            .whitespace(" "),
            .token("world", .string),
            .whitespace(" "),
            .plainText("\\("),
            .token("call", .call),
            .plainText("())"),
            .token("\"", .string)
        ])
    }

    func testStringLiteralWithInterpolatedClosureArgumentShorthand() {
        let components = highlighter.highlight(#""\($0)""#)

        XCTAssertEqual(components, [
            .token("\"", .string),
            .plainText(#"\($0)"#),
            .token("\"", .string)
        ])
    }

    func testStringLiteralWithCustomIterpolation() {
        let components = highlighter.highlight("""
        "Hello \\(label: a, b) world \\(label: call())"
        """)

        XCTAssertEqual(components, [
            .token("\"Hello", .string),
            .whitespace(" "),
            .plainText("\\(label:"),
            .whitespace(" "),
            .plainText("a,"),
            .whitespace(" "),
            .plainText("b)"),
            .whitespace(" "),
            .token("world", .string),
            .whitespace(" "),
            .plainText("\\(label:"),
            .whitespace(" "),
            .token("call", .call),
            .plainText("())"),
            .token("\"", .string)
        ])
    }

    func testStringLiteralWithInterpolationSurroundedByBrackets() {
        let components = highlighter.highlight(#""[\(text)]""#)

        XCTAssertEqual(components, [
            .token(#""["#, .string),
            .plainText(#"\(text)"#),
            .token(#"]""#, .string)
        ])
    }

    func testStringLiteralWithInterpolationPrefixedByPunctuation() {
        let components = highlighter.highlight(#"".\(text)""#)

        XCTAssertEqual(components, [
            .token("\".", .string),
            .plainText(#"\(text)"#),
            .token("\"", .string)
        ])
    }

    func testStringLiteralWithInterpolationContainingString() {
        let components = highlighter.highlight(#""\(name ?? "name")""#)

        XCTAssertEqual(components, [
            .token("\"", .string),
            .plainText("\\(name"),
            .whitespace(" "),
            .plainText("??"),
            .whitespace(" "),
            .token("\"name\"", .string),
            .plainText(")"),
            .token("\"", .string)
        ])
    }

    func testMultiLineStringLiteral() {
        let components = highlighter.highlight("""
        let string = \"\"\"
        Hello \\(variable)
        \"\"\"
        """)

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("string"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("\"\"\"", .string),
            .whitespace("\n"),
            .token("Hello", .string),
            .whitespace(" "),
            .plainText("\\(variable)"),
            .whitespace("\n"),
            .token("\"\"\"", .string)
        ])
    }

    func testSingleLineRawStringLiteral() {
        let components = highlighter.highlight("""
        #"A raw string \\(withoutInterpolation) yes"#
        """)

        XCTAssertEqual(components, [
            .token("#\"A", .string),
            .whitespace(" "),
            .token("raw", .string),
            .whitespace(" "),
            .token("string", .string),
            .whitespace(" "),
            .token("\\(withoutInterpolation)", .string),
            .whitespace(" "),
            .token("yes\"#", .string)
        ])
    }

    func testMultiLineRawStringLiteral() {
        let components = highlighter.highlight("""
        #\"\"\"
        A raw string \\(withoutInterpolation)
        with multiple lines. #" Nested "#
        \"\"\"#
        """)

        XCTAssertEqual(components, [
            .token("#\"\"\"", .string),
            .whitespace("\n"),
            .token("A", .string),
            .whitespace(" "),
            .token("raw", .string),
            .whitespace(" "),
            .token("string", .string),
            .whitespace(" "),
            .token("\\(withoutInterpolation)", .string),
            .whitespace("\n"),
            .token("with", .string),
            .whitespace(" "),
            .token("multiple", .string),
            .whitespace(" "),
            .token("lines.", .string),
            .whitespace(" "),
            .token("#\"", .string),
            .whitespace(" "),
            .token("Nested", .string),
            .whitespace(" "),
            .token("\"#", .string),
            .whitespace("\n"),
            .token("\"\"\"#", .string)
        ])
    }

    func testRawStringWithInterpolation() {
        let components = highlighter.highlight("#\"Hello \\#(variable) world\"#")

        XCTAssertEqual(components, [
            .token("#\"Hello", .string),
            .whitespace(" "),
            .plainText("\\#(variable)"),
            .whitespace(" "),
            .token("world\"#", .string)
        ])
    }

    func testStringLiteralContainingOnlyNewLine() {
        let components = highlighter.highlight(#"text.split(separator: "\n")"#)

        XCTAssertEqual(components, [
            .plainText("text."),
            .token("split", .call),
            .plainText("(separator:"),
            .whitespace(" "),
            .token(#""\n""#, .string),
            .plainText(")")
        ])
    }

    func testDoubleLiteral() {
        let components = highlighter.highlight("let double = 1.13")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("double"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("1.13", .number)
        ])
    }

    func testIntegerLiteralWithSeparators() {
        let components = highlighter.highlight("let int = 1_000_000")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("int"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("1_000_000", .number)
        ])
    }

    func testKeyPathLiteral() {
        let components = highlighter.highlight("let value = object[keyPath: \\.property]")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("object[keyPath:"),
            .whitespace(" "),
            .plainText("\\."),
            .token("property", .property),
            .plainText("]")
        ])
    }

    func testKeyPathLiteralsAsArguments() {
        let components = highlighter.highlight(#"user.bind(\.name, to: \.text)"#)

        XCTAssertEqual(components, [
            .plainText("user."),
            .token("bind", .call),
            .plainText(#"(\."#),
            .token("name", .property),
            .plainText(","),
            .whitespace(" "),
            .plainText("to:"),
            .whitespace(" "),
            .plainText(#"\."#),
            .token("text", .property),
            .plainText(")")
        ])
    }
}
