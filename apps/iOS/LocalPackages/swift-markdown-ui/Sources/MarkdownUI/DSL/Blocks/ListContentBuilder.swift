import Foundation

/// A result builder that you can use to compose bulleted and numbered lists.
///
/// You don't call the methods of the result builder directly. Instead, Swift uses them to combine the elements
/// you declare in any closure with the `@ListContentBuilder` attribute. In particular, you rely on this
/// behavior when you declare the `content` inside a list element initializer such as
/// ``NumberedList/init(tight:start:items:)``.
@resultBuilder public enum ListContentBuilder {
  public static func buildBlock(_ components: [ListItem]...) -> [ListItem] {
    components.flatMap { $0 }
  }

  public static func buildExpression(_ expression: String) -> [ListItem] {
    [.init(expression)]
  }

  public static func buildExpression(_ expression: ListItem) -> [ListItem] {
    [expression]
  }

  public static func buildArray(_ components: [[ListItem]]) -> [ListItem] {
    components.flatMap { $0 }
  }

  public static func buildOptional(_ component: [ListItem]?) -> [ListItem] {
    component ?? []
  }

  public static func buildEither(first component: [ListItem]) -> [ListItem] {
    component
  }

  public static func buildEither(second component: [ListItem]) -> [ListItem] {
    component
  }
}
