import SwiftUI

extension Theme {
  /// The default Markdown theme.
  ///
  /// Style | Preview
  /// --- | ---
  /// Inline text | ![](BasicInlines)
  /// Headings | ![](Heading)
  /// Blockquote | ![](BlockquoteContent)
  /// Code block | ![](CodeBlock)
  /// Image | ![](Paragraph)
  /// Task list | ![](TaskList)
  /// Bulleted list | ![](NestedBulletedList)
  /// Numbered list | ![](NumberedList)
  /// Table | ![](Table-Collection)
  public static let basic = Theme()
    .code {
      FontFamilyVariant(.monospaced)
      FontSize(.em(0.94))
    }
    .heading1 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(2))
        }
    }
    .heading2 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(1.5))
        }
    }
    .heading3 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(1.17))
        }
    }
    .heading4 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(1))
        }
    }
    .heading5 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(0.83))
        }
    }
    .heading6 { configuration in
      configuration.label
        .markdownMargin(top: .rem(1.5), bottom: .rem(1))
        .markdownTextStyle {
          FontWeight(.semibold)
          FontSize(.em(0.67))
        }
    }
    .paragraph { configuration in
      configuration.label
        .fixedSize(horizontal: false, vertical: true)
        .relativeLineSpacing(.em(0.15))
        .markdownMargin(top: .zero, bottom: .em(1))
    }
    .blockquote { configuration in
      configuration.label
        .markdownTextStyle {
          FontStyle(.italic)
        }
        .relativePadding(.leading, length: .em(2))
        .relativePadding(.trailing, length: .em(1))
    }
    .codeBlock { configuration in
      ScrollView(.horizontal) {
        configuration.label
          .fixedSize(horizontal: false, vertical: true)
          .relativeLineSpacing(.em(0.15))
          .relativePadding(.leading, length: .rem(1))
          .markdownTextStyle {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.94))
          }
      }
      .markdownMargin(top: .zero, bottom: .em(1))
    }
    .table { configuration in
      configuration.label
        .fixedSize(horizontal: false, vertical: true)
        .markdownMargin(top: .zero, bottom: .em(1))
    }
    .tableCell { configuration in
      configuration.label
        .markdownTextStyle {
          if configuration.row == 0 {
            FontWeight(.semibold)
          }
        }
        .fixedSize(horizontal: false, vertical: true)
        .relativeLineSpacing(.em(0.15))
        .relativePadding(.horizontal, length: .em(0.72))
        .relativePadding(.vertical, length: .em(0.35))
    }
    .thematicBreak {
      Divider().markdownMargin(top: .em(2), bottom: .em(2))
    }
}
