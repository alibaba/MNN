import Foundation

/// A deleted or redacted text in a Markdown content block.
public struct Strikethrough: InlineContentProtocol {
  public var _inlineContent: InlineContent {
    .init(inlines: [.strikethrough(children: self.content.inlines)])
  }

  private let content: InlineContent

  init(content: InlineContent) {
    self.content = content
  }

  /// Creates a deleted inline by applying the deleted style to a string.
  /// - Parameter text: The text to delete.
  public init(_ text: String) {
    self.init(content: .init(inlines: [.text(text)]))
  }

  /// Creates a deleted inline by applying the deleted style to other inline content.
  /// - Parameter content: An inline content builder that returns the inlines to delete.
  public init(@InlineContentBuilder content: () -> InlineContent) {
    self.init(content: content())
  }
}
