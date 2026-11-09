import Foundation

/// A result builder that you can use to compose Markdown inline content.
///
/// You don't call the methods of the result builder directly. Instead, MarkdownUI annotates the `content` parameter of the
/// ``Paragraph``, ``Heading``, and ``TextTableColumn`` initializers with the `@InlineContentBuider` attribute,
/// implicitly calling this builder for you.
@resultBuilder public enum InlineContentBuilder {
  public static func buildBlock(_ components: InlineContentProtocol...) -> InlineContent {
    .init(components)
  }

  public static func buildExpression(_ expression: InlineContentProtocol) -> InlineContent {
    expression._inlineContent
  }

  public static func buildExpression(_ expression: String) -> InlineContent {
    .init(expression)
  }

  public static func buildArray(_ components: [InlineContentProtocol]) -> InlineContent {
    .init(components)
  }

  public static func buildOptional(_ component: InlineContentProtocol?) -> InlineContent {
    component?._inlineContent ?? .init()
  }

  public static func buildEither(first component: InlineContentProtocol) -> InlineContent {
    component._inlineContent
  }

  public static func buildEither(second component: InlineContentProtocol) -> InlineContent {
    component._inlineContent
  }
}
