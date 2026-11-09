import Foundation
import XCTest

@testable import MarkdownUI

final class ListContentBuilderTests: XCTestCase {
  func testEmpty() {
    // given
    @ListContentBuilder func build() -> [ListItem] {}

    // when
    let result = build()

    // then
    XCTAssertEqual([], result)
  }

  func testExpressions() {
    // given
    @ListContentBuilder func build() -> [ListItem] {
      "Flour"
      ListItem {
        "Cheese"
      }
      ListItem {
        Paragraph {
          "Tomatoes"
        }
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      [
        .init(children: [.paragraph(content: [.text("Flour")])]),
        .init(children: [.paragraph(content: [.text("Cheese")])]),
        .init(children: [.paragraph(content: [.text("Tomatoes")])]),
      ],
      result
    )
  }

  func testForLoops() {
    // given
    @ListContentBuilder func build() -> [ListItem] {
      for i in 0...3 {
        "\(i)"
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      [
        .init(children: [.paragraph(content: [.text("0")])]),
        .init(children: [.paragraph(content: [.text("1")])]),
        .init(children: [.paragraph(content: [.text("2")])]),
        .init(children: [.paragraph(content: [.text("3")])]),
      ],
      result
    )
  }

  func testIf() {
    @ListContentBuilder func build() -> [ListItem] {
      "Something is:"
      if true {
        ListItem {
          "true"
        }
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      [
        .init(children: [.paragraph(content: [.text("Something is:")])]),
        .init(children: [.paragraph(content: [.text("true")])]),
      ],
      result
    )
  }

  func testIfElse() {
    @ListContentBuilder func build(_ value: Bool) -> [ListItem] {
      "Something is:"
      if value {
        ListItem {
          "true"
        }
      } else {
        "false"
      }
    }

    // when
    let result1 = build(true)
    let result2 = build(false)

    // then
    XCTAssertEqual(
      [
        .init(children: [.paragraph(content: [.text("Something is:")])]),
        .init(children: [.paragraph(content: [.text("true")])]),
      ],
      result1
    )
    XCTAssertEqual(
      [
        .init(children: [.paragraph(content: [.text("Something is:")])]),
        .init(children: [.paragraph(content: [.text("false")])]),
      ],
      result2
    )
  }
}
