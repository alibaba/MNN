import Foundation
import XCTest

@testable import MarkdownUI

final class HTMLTagTests: XCTestCase {
  func testInvalidTag() {
    XCTAssertNil(HTMLTag(""))
    XCTAssertNil(HTMLTag("foo"))
    XCTAssertNil(HTMLTag("<"))
    XCTAssertNil(HTMLTag("<>"))
  }

  func testOpeningTag() {
    // given
    let tag = HTMLTag("<sub>")

    // then
    XCTAssertEqual("sub", tag?.name)
  }

  func testOpeningTagWithAttributes() {
    // given
    let tag = HTMLTag(
      "<img src=\"img_girl.jpg\" alt=\"Girl in a jacket\" width=\"500\" height=\"600\">"
    )

    // then
    XCTAssertEqual("img", tag?.name)
  }

  func testClosingTag() {
    let tag = HTMLTag("</sub>")
    XCTAssertEqual(tag?.name, "sub")
  }

  func testSelfClosingTag() {
    XCTAssertEqual("br", HTMLTag("<br />")?.name)
  }
}
