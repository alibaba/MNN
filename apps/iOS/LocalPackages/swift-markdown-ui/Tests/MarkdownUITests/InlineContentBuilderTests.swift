import Foundation
import XCTest

@testable import MarkdownUI

final class InlineContentBuilderTests: XCTestCase {
  func testEmpty() {
    // given
    @InlineContentBuilder func build() -> InlineContent {}

    // when
    let result = build()

    // then
    XCTAssertEqual(.init(), result)
  }

  func testExpressions() {
    // given
    @InlineContentBuilder func build() -> InlineContent {
      "Hello"
      SoftBreak()
      "world!"
      LineBreak()
      Code("let a = b")
      Strikethrough {
        "This is a "
        Strong("mistake, ")
        Emphasis("right?")
      }
      InlineLink("Hurricane", destination: URL(string: "https://w.wiki/qYn")!)
      InlineImage("Puppy", source: URL(string: "https://picsum.photos/id/237/200/300")!)
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      InlineContent(
        inlines: [
          .text("Hello"),
          .softBreak,
          .text("world!"),
          .lineBreak,
          .code("let a = b"),
          .strikethrough(
            children: [
              .text("This is a "),
              .strong(children: [.text("mistake, ")]),
              .emphasis(children: [.text("right?")]),
            ]
          ),
          .link(destination: "https://w.wiki/qYn", children: [.text("Hurricane")]),
          .image(source: "https://picsum.photos/id/237/200/300", children: [.text("Puppy")]),
        ]
      ),
      result
    )
  }

  func testForLoops() {
    // given
    @InlineContentBuilder func build() -> InlineContent {
      for i in 0...3 {
        "\(i)"
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      InlineContent(
        inlines: [
          .text("0"),
          .text("1"),
          .text("2"),
          .text("3"),
        ]
      ),
      result
    )
  }

  func testIf() {
    @InlineContentBuilder func build() -> InlineContent {
      "Something is "
      if true {
        Emphasis {
          "true"
        }
      }
    }

    // when
    let result = build()

    // then
    XCTAssertEqual(
      InlineContent(
        inlines: [
          .text("Something is "),
          .emphasis(children: [.text("true")]),
        ]
      ),
      result
    )
  }

  func testIfElse() {
    @InlineContentBuilder func build(_ value: Bool) -> InlineContent {
      "Something is "
      if value {
        Emphasis {
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
      InlineContent(
        inlines: [
          .text("Something is "),
          .emphasis(children: [.text("true")]),
        ]
      ),
      result1
    )
    XCTAssertEqual(
      InlineContent(
        inlines: [
          .text("Something is "),
          .text("false"),
        ]
      ),
      result2
    )
  }
}
