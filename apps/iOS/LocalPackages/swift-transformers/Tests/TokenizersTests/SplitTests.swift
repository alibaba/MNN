//
//  SplitTests.swift
//
//
//  Created by Pedro Cuenca on 20240120.
//

import Tokenizers
import XCTest

class SplitTests: XCTestCase {
    func testSplitBehaviorMergedWithPrevious() {
        XCTAssertEqual(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious),
            ["the-", "final-", "-", "countdown"]
        )

        XCTAssertEqual(
            "the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious),
            ["the-", "final-", "-", "countdown-"]
        )

        XCTAssertEqual(
            "the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious),
            ["the-", "final-", "-", "countdown-", "-"]
        )

        XCTAssertEqual(
            "-the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious),
            ["-", "the-", "final-", "-", "countdown-", "-"]
        )

        XCTAssertEqual(
            "--the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious),
            ["-", "-", "the-", "final-", "-", "countdown-", "-"]
        )
    }

    func testSplitBehaviorMergedWithNext() {
        XCTAssertEqual(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext),
            ["the", "-final", "-", "-countdown"]
        )

        XCTAssertEqual(
            "-the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext),
            ["-the", "-final", "-", "-countdown"]
        )

        XCTAssertEqual(
            "--the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext),
            ["-", "-the", "-final", "-", "-countdown"]
        )

        XCTAssertEqual(
            "--the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext),
            ["-", "-the", "-final", "-", "-countdown", "-"]
        )
    }

    func testSplitBehaviorOther() {
        XCTAssertEqual(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated),
            ["the", "-", "final", "-", "-", "countdown"]
        )

        XCTAssertEqual(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed),
            ["the", "final", "countdown"]
        )
    }
}
