/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class EnumTests: SyntaxHighlighterTestCase {
    func testEnumDotSyntaxInAssignment() {
        let components = highlighter.highlight("let value: Enum = .aCase")

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("value:"),
            .whitespace(" "),
            .token("Enum", .type),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("."),
            .token("aCase", .dotAccess)
        ])
    }

    func testEnumDotSyntaxAsArgument() {
        let components = highlighter.highlight("call(.aCase)")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("(."),
            .token("aCase", .dotAccess),
            .plainText(")")
        ])
    }

    func testEnumDotSyntaxWithAssociatedValueTreatedAsCall() {
        let components = highlighter.highlight("call(.error(error))")

        XCTAssertEqual(components, [
            .token("call", .call),
            .plainText("(."),
            .token("error", .call),
            .plainText("(error))")
        ])
    }

    func testUsingEnumInSubscript() {
        let components = highlighter.highlight("dictionary[.key]")

        XCTAssertEqual(components, [
            .plainText("dictionary[."),
            .token("key", .dotAccess),
            .plainText("]")
        ])
    }
}
