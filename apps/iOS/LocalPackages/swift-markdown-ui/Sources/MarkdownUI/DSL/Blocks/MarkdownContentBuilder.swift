import Foundation

/// A result builder that you can use to compose Markdown content.
///
/// You don't call the methods of the result builder directly. Instead, Swift uses them to combine the elements
/// you declare in any closure with the `@MarkdownContentBuilder` attribute. In particular, you rely on
/// this behavior when you declare the `content` inside a Markdown view initializer such as
/// ``Markdown/init(baseURL:imageBaseURL:content:)``.
@resultBuilder public enum MarkdownContentBuilder {
  public static func buildBlock(_ components: MarkdownContentProtocol...) -> MarkdownContent {
    .init(components)
  }

  public static func buildExpression(_ expression: MarkdownContentProtocol) -> MarkdownContent {
    expression._markdownContent
  }

  public static func buildExpression(_ expression: String) -> MarkdownContent {
    .init(expression)
  }

  public static func buildArray(_ components: [MarkdownContentProtocol]) -> MarkdownContent {
    .init(components)
  }

  public static func buildOptional(_ component: MarkdownContentProtocol?) -> MarkdownContent {
    component?._markdownContent ?? .init()
  }

  public static func buildEither(first component: MarkdownContentProtocol) -> MarkdownContent {
    component._markdownContent
  }

  public static func buildEither(second component: MarkdownContentProtocol) -> MarkdownContent {
    component._markdownContent
  }
}
