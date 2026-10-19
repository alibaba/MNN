/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class CommentTests: SyntaxHighlighterTestCase {
    func testSingleLineComment() {
        let components = highlighter.highlight("call() // Hello call() var \"string\"\ncall()")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("()"),
            .whitespace(" "),
            .token("//", .comment),
            .whitespace(" "),
            .token("Hello", .comment),
            .whitespace(" "),
            .token("call()", .comment),
            .whitespace(" "),
            .token("var", .comment),
            .whitespace(" "),
            .token("\"string\"", .comment),
            .whitespace("\n"),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testMultiLineComment() {
        let components = highlighter.highlight("""
        struct Foo {}
        /* Comment
            Hello!
        */ call()
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Foo"),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace("\n"),
            .token("/*", .comment),
            .whitespace(" "),
            .token("Comment", .comment),
            .whitespace("\n    "),
            .token("Hello!", .comment),
            .whitespace("\n"),
            .token("*/", .comment),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testMultiLineCommentWithDoubleAsterisks() {
        let components = highlighter.highlight("""
        struct Foo {}
        /** Comment
            Hello!
        */ call()
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Foo"),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace("\n"),
            .token("/**", .comment),
            .whitespace(" "),
            .token("Comment", .comment),
            .whitespace("\n    "),
            .token("Hello!", .comment),
            .whitespace("\n"),
            .token("*/", .comment),
            .whitespace(" "),
            .token("call", .call),
            .plainText("()")
        ])
    }

    func testMutliLineDocumentationComment() {
        let components = highlighter.highlight("""
        /**
         *  Documentation
         */
        class MyClass {}
        """)

        XCTAssertEqual(components, [
            .token("/**", .comment),
            .whitespace("\n "),
            .token("*", .comment),
            .whitespace("  "),
            .token("Documentation", .comment),
            .whitespace("\n "),
            .token("*/", .comment),
            .whitespace("\n"),
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("MyClass"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testCommentStartingWithPunctuation() {
        let components = highlighter.highlight("//.call()")
        XCTAssertEqual(components, [.token("//.call()", .comment)])
    }

    func testCommentEndingWithComma() {
        let components = highlighter.highlight("""
        // Hello,
        class World {}
        """)

        XCTAssertEqual(components, [
            .token("//", .comment),
            .whitespace(" "),
            .token("Hello,", .comment),
            .whitespace("\n"),
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("World"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testCommentPrecededByComma() {
        let components = highlighter.highlight("""
        func find(
            string: String,//TODO: Remove
            options: Options
        )
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("find("),
            .whitespace("\n    "),
            .plainText("string:"),
            .whitespace(" "),
            .token("String", .type),
            .plainText(","),
            .token("//TODO:", .comment),
            .whitespace(" "),
            .token("Remove", .comment),
            .whitespace("\n    "),
            .plainText("options:"),
            .whitespace(" "),
            .token("Options", .type),
            .whitespace("\n"),
            .plainText(")")
        ])
    }
        
    func testCommentWithNumber() {
        let components = highlighter.highlight("// 1")

        XCTAssertEqual(components, [
            .token("//", .comment),
            .whitespace(" "),
            .token("1", .comment)
        ])
    }

    func testCommentWithNoWhiteSpaceToPunctuation() {
        let components = highlighter.highlight("""
        (/* Hello */)
        .// World
        (/**/)
        """)

         XCTAssertEqual(components, [
            .plainText("("),
            .token("/*", .comment),
            .whitespace(" "),
            .token("Hello", .comment),
            .whitespace(" "),
            .token("*/", .comment),
            .plainText(")"),
            .whitespace("\n"),
            .plainText("."),
            .token("//", .comment),
            .whitespace(" "),
            .token("World", .comment),
            .whitespace("\n"),
            .plainText("("),
            .token("/**/", .comment),
            .plainText(")"),
        ])
    }

    func testCommentsNextToCurlyBrackets() {
        let components = highlighter.highlight("""
        call {//commentA
        }//commentB
        """)

        XCTAssertEqual(components, [
            .token("call", .call),
            .whitespace(" "),
            .plainText("{"),
            .token("//commentA", .comment),
            .whitespace("\n"),
            .plainText("}"),
            .token("//commentB", .comment)
        ])
    }

    func testCommentWithinGenericTypeList() {
        let components = highlighter.highlight("""
        struct Box<One, /*Comment*/Two: Equatable, Three> {}
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Box<One,"),
            .whitespace(" "),
            .token("/*Comment*/", .comment),
            .plainText("Two:"),
            .whitespace(" "),
            .token("Equatable", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("Three>"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testCommentsNextToGenericTypeList() {
        let components = highlighter.highlight("""
        struct Box/*Start*/<Content>/*End*/ {}
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Box"),
            .token("/*Start*/", .comment),
            .plainText("<Content>"),
            .token("/*End*/", .comment),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testCommentsNextToInitialization() {
        let components = highlighter.highlight("/*Start*/Object()/*End*/")

        XCTAssertEqual(components, [
            .token("/*Start*/", .comment),
            .token("Object", .type),
            .plainText("()"),
            .token("/*End*/", .comment)
        ])
    }

    func testCommentsNextToProtocolName() {
        let components = highlighter.highlight("""
        struct Model<Value>: /*Start*/Equatable/*End*/ {}
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Model<Value>:"),
            .whitespace(" "),
            .token("/*Start*/", .comment),
            .token("Equatable", .type),
            .token("/*End*/", .comment),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testCommentsAfterOptionalTypes() {
        let components = highlighter.highlight("""
        struct Model {
            var one: String?//One
            var two: String?/*Two*/
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Model"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("one:"),
            .whitespace(" "),
            .token("String", .type),
            .plainText("?"),
            .token("//One", .comment),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("two:"),
            .whitespace(" "),
            .token("String", .type),
            .plainText("?"),
            .token("/*Two*/", .comment),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testCommentsAfterArrayTypes() {
        let components = highlighter.highlight("""
        struct Model {
            var one: [String]//One
            var two: [String]/*Two*/
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Model"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("one:"),
            .whitespace(" "),
            .plainText("["),
            .token("String", .type),
            .plainText("]"),
            .token("//One", .comment),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("two:"),
            .whitespace(" "),
            .plainText("["),
            .token("String", .type),
            .plainText("]"),
            .token("/*Two*/", .comment),
            .whitespace("\n"),
            .plainText("}")
        ])
    }
}
