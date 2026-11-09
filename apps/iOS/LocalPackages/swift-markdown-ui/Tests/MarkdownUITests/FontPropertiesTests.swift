#if !os(tvOS)
  import SwiftUI
  import XCTest

  @testable import MarkdownUI

  final class FontPropertiesTests: XCTestCase {
    func testFontWithProperties() {
      // given
      var fontProperties = FontProperties()

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(family: .custom("Menlo"))

      // then
      XCTAssertEqual(
        Font.custom("Menlo", fixedSize: FontProperties.defaultSize),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(familyVariant: .monospaced)

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default).monospaced(),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(capsVariant: .lowercaseSmallCaps)

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default).lowercaseSmallCaps(),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(digitVariant: .monospaced)

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default).monospacedDigit(),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(style: .italic)

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default).italic(),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(weight: .heavy)

      // then
      XCTAssertEqual(
        Font.system(size: FontProperties.defaultSize, design: .default).weight(.heavy),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(size: 42)

      // then
      XCTAssertEqual(
        Font.system(size: 42, design: .default),
        Font.withProperties(fontProperties)
      )

      // when
      fontProperties = FontProperties(scale: 1.5)

      // then
      XCTAssertEqual(
        Font.system(size: round(FontProperties.defaultSize * 1.5), design: .default),
        Font.withProperties(fontProperties)
      )
    }
  }
#endif
