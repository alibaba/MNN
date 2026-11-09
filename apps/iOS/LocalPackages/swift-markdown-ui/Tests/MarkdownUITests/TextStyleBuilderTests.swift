import MarkdownUI
import SwiftUI
import XCTest

final class TextStyleBuilderTests: XCTestCase {
  func testBuildEmpty() {
    // given
    @TextStyleBuilder func build() -> some TextStyle {}
    let textStyle = build()

    // when
    var attributes = AttributeContainer()
    textStyle._collectAttributes(in: &attributes)

    // then
    XCTAssertEqual(AttributeContainer(), attributes)
  }

  func testBuildOne() {
    // given
    @TextStyleBuilder func build() -> some TextStyle {
      ForegroundColor(.primary)
    }
    let textStyle = build()

    // when
    var attributes = AttributeContainer()
    textStyle._collectAttributes(in: &attributes)

    // then
    XCTAssertEqual(AttributeContainer().foregroundColor(.primary), attributes)
  }

  func testBuildMany() {
    // given
    @TextStyleBuilder func build() -> some TextStyle {
      ForegroundColor(.primary)
      BackgroundColor(.cyan)
      UnderlineStyle(.single)
    }
    let textStyle = build()

    // when
    var attributes = AttributeContainer()
    textStyle._collectAttributes(in: &attributes)

    // then
    XCTAssertEqual(
      AttributeContainer()
        .foregroundColor(.primary)
        .backgroundColor(.cyan)
        .underlineStyle(.single),
      attributes
    )
  }

  func testBuildOptional() {
    // given
    @TextStyleBuilder func makeTextStyle(_ condition: Bool) -> some TextStyle {
      ForegroundColor(.primary)
      if condition {
        BackgroundColor(.cyan)
      }
    }
    let textStyle1 = makeTextStyle(true)
    let textStyle2 = makeTextStyle(false)

    // when
    var attributes1 = AttributeContainer()
    textStyle1._collectAttributes(in: &attributes1)
    var attributes2 = AttributeContainer()
    textStyle2._collectAttributes(in: &attributes2)

    // then
    XCTAssertEqual(
      AttributeContainer()
        .foregroundColor(.primary)
        .backgroundColor(.cyan),
      attributes1
    )
    XCTAssertEqual(
      AttributeContainer()
        .foregroundColor(.primary),
      attributes2
    )
  }

  func testBuildEither() {
    // given
    @TextStyleBuilder func makeTextStyle(_ condition: Bool) -> some TextStyle {
      ForegroundColor(.primary)
      if condition {
        BackgroundColor(.cyan)
      } else {
        UnderlineStyle(.single)
      }
    }
    let textStyle1 = makeTextStyle(true)
    let textStyle2 = makeTextStyle(false)

    // when
    var attributes1 = AttributeContainer()
    textStyle1._collectAttributes(in: &attributes1)
    var attributes2 = AttributeContainer()
    textStyle2._collectAttributes(in: &attributes2)

    // then
    XCTAssertEqual(
      AttributeContainer()
        .foregroundColor(.primary)
        .backgroundColor(.cyan),
      attributes1
    )
    XCTAssertEqual(
      AttributeContainer()
        .foregroundColor(.primary)
        .underlineStyle(.single),
      attributes2
    )
  }
}
