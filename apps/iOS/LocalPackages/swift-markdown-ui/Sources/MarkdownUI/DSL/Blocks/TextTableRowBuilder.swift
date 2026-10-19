import Foundation

/// A result builder that creates table rows from closures.
///
/// You don't call the methods of the result builder directly. Instead, MarkdownUI annotates the `rows`
/// parameter of the ``TextTable/init(columns:rows:)`` initializer with the
/// `@TextTableRowBuilder` attribute, implicitly calling this builder for you.
@resultBuilder public enum TextTableRowBuilder<Value> {
  public static func buildBlock(_ components: [TextTableRow<Value>]...) -> [TextTableRow<Value>] {
    components.flatMap { $0 }
  }

  public static func buildExpression(_ expression: TextTableRow<Value>) -> [TextTableRow<Value>] {
    [expression]
  }

  public static func buildArray(_ components: [[TextTableRow<Value>]]) -> [TextTableRow<Value>] {
    components.flatMap { $0 }
  }

  public static func buildOptional(_ component: [TextTableRow<Value>]?) -> [TextTableRow<Value>] {
    component ?? []
  }

  public static func buildEither(first component: [TextTableRow<Value>]) -> [TextTableRow<Value>] {
    component
  }

  public static func buildEither(second component: [TextTableRow<Value>]) -> [TextTableRow<Value>] {
    component
  }
}
