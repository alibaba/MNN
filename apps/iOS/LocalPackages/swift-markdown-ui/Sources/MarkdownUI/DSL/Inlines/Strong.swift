import Foundation

/// A strong text in a Markdown content block.
public struct Strong: InlineContentProtocol {
  public var _inlineContent: InlineContent {
    .init(inlines: [.strong(children: self.content.inlines)])
  }

  private let content: InlineContent

  init(content: InlineContent) {
    self.content = content
  }

  /// Creates a strong inline by applying the strong style to a string.
  /// - Parameter text: The text to make strong.
  public init(_ text: String) {
    self.init(content: .init(inlines: [.text(text)]))
  }

  /// Creates a strong inline by applying the strong style to other inline content.
  /// - Parameter content: An inline content builder that returns the inlines to make strong.
  public init(@InlineContentBuilder content: () -> InlineContent) {
    self.init(content: content())
  }
}
