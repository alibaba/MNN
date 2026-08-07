import Foundation

/// An inline content that can navigate to a URL.
///
/// You can create a link by providing a destination URL and a title. The title tells the purpose of the link and
/// can be any inline content, including an image.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "A picture of a black lab puppy:"
///   }
///   Paragraph {
///     InlineLink(destination: URL(string: "https://en.wikipedia.org/wiki/Labrador_Retriever")!) {
///       InlineImage(source: URL(string: "237-100x150")!)
///     }
///   }
/// }
/// ```
///
/// You can directly provide the title text for links with an unstyled title.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "Visit "
///     InlineLink("our site.", destination: URL(string: "https://www.example.com")!)
///   }
/// }
/// ```
public struct InlineLink: InlineContentProtocol {
  public var _inlineContent: InlineContent {
    .init(inlines: [.link(destination: self.destination, children: self.content.inlines)])
  }

  private let destination: String
  private let content: InlineContent

  init(destination: String, content: InlineContent) {
    self.destination = destination
    self.content = content
  }

  /// Creates a link to a given destination with an unstyled text title.
  /// - Parameters:
  ///   - text: The title of the link.
  ///   - destination: The URL for the link.
  public init(_ text: String, destination: URL) {
    self.init(destination: destination.absoluteString, content: .init(inlines: [.text(text)]))
  }

  /// Creates a link to a given destination with a title composed of other inlines.
  /// - Parameters:
  ///   - destination: The URL for the link.
  ///   - content: An inline content builder that returns the title of the link.
  public init(destination: URL, @InlineContentBuilder content: () -> InlineContent) {
    self.init(destination: destination.absoluteString, content: content())
  }
}
