import Foundation
import XCTest

@testable import MarkdownUI

final class MarkdownContentBuilderTests: XCTestCase {
  func testEmpty() {
    // given
    @MarkdownContentBuilder func build() -> MarkdownContent {}

    // when
    let result = build()

    // then
    XCTAssertEqual(.init(), result)
  }

  func testExpressions() {
    // given
    @MarkdownContentBuilder func build() -> MarkdownContent {
      "**First** paragraph."
      Paragraph {
        Strong("Second")
        " paragraph."
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      MarkdownContent(
        blocks: [
          .paragraph(
            content: [
              .strong(children: [.text("First")]),
              .text(" paragraph."),
            ]
          ),
          .paragraph(
            content: [
              .strong(children: [.text("Second")]),
              .text(" paragraph."),
            ]
          ),
        ]
      ),
      result
    )
  }

  func testForLoops() {
    // given
    @MarkdownContentBuilder func build() -> MarkdownContent {
      for i in 0...3 {
        "\(i)"
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      MarkdownContent(
        blocks: [
          .paragraph(content: [.text("0")]),
          .paragraph(content: [.text("1")]),
          .paragraph(content: [.text("2")]),
          .paragraph(content: [.text("3")]),
        ]
      ),
      result
    )
  }

  func testIf() {
    @MarkdownContentBuilder func build() -> MarkdownContent {
      "Something is:"
      if true {
        "true"
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      MarkdownContent(
        blocks: [
          .paragraph(content: [.text("Something is:")]),
          .paragraph(content: [.text("true")]),
        ]
      ),
      result
    )
  }

  func testIfElse() {
    @MarkdownContentBuilder func build(_ value: Bool) -> MarkdownContent {
      "Something is:"
      if value {
        "true"
      } else {
        "false"
      }
    }

    // when
    let result1 = build(true)
    let result2 = build(false)

    // then
    XCTAssertEqual(
      MarkdownContent(
        blocks: [
          .paragraph(content: [.text("Something is:")]),
          .paragraph(content: [.text("true")]),
        ]
      ),
      result1
    )
    XCTAssertEqual(
      MarkdownContent(
        blocks: [
          .paragraph(content: [.text("Something is:")]),
          .paragraph(content: [.text("false")]),
        ]
      ),
      result2
    )
  }
}
