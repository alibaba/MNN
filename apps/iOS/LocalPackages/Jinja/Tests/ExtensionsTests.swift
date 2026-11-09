//
//  ExtensionsTests.swift
//  Jinja
//
//  Created by John Mai on 2025/7/10.
//

import XCTest
@testable import Jinja

final class ExtensionsTests: XCTestCase {
    func testStringReplacingOccurrences() throws {
        let text = "hello world hello python hello swift"
        let result = text.replacingOccurrences(of: "hello", with: "hi", count: 2)
        XCTAssertEqual(result, "hi world hi python hello swift")

        let result2 = text.replacingOccurrences(of: "hello", with: "hi", count: 0)
        XCTAssertEqual(result2, text)

        let result3 = text.replacingOccurrences(of: "hello", with: "hi", count: 5)
        XCTAssertEqual(result3, "hi world hi python hi swift")
    }
}
