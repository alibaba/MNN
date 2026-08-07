import Foundation

/// A Markdown heading element.
///
/// You can use heading elements to display titles and subtitles or mark different sections in the content.
///
/// ```swift
/// Markdown {
///   Heading {
///     "The largest heading"
///   }
///   Heading(.level2) {
///     "The second largest heading"
///   }
///   Heading(.level6) {
///     "The smallest heading"
///   }
/// }
/// ```
///
/// ![](Heading)
public struct Heading: MarkdownContentProtocol {
  public enum Level: Int {
    case level1 = 1
    case level2 = 2
    case level3 = 3
    case level4 = 4
    case level5 = 5
    case level6 = 6
  }

  public var _markdownContent: MarkdownContent {
    .init(blocks: [.heading(level: self.level.rawValue, content: self.content.inlines)])
  }

  private let level: Level
  private let content: InlineContent

  /// Creates a heading element with the specified level and inline content.
  /// - Parameters:
  ///   - level: A level that determines the heading size.
  ///   - content: The inline content for the heading.
  public init(_ level: Level = .level1, @InlineContentBuilder content: () -> InlineContent) {
    self.level = level
    self.content = content()
  }
}
