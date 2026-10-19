import Foundation

/// An image in a Markdown content block.
///
/// You can use an image inline to embed an image in a paragraph.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "A picture of a black lab puppy:"
///   }
///   Paragraph {
///     InlineImage(source: URL(string: "https://picsum.photos/id/237/100/150")!)
///     InlineLink(destination: URL(string: "https://en.wikipedia.org/wiki/Labrador_Retriever")!) {
///       InlineImage(source: URL(string: "https://picsum.photos/id/237/100/150")!)
///     }
///   }
///   Paragraph {
///     "You can also insert images in a line of text, such as "
///     InlineImage(source: URL(string: "https://picsum.photos/id/237/100/150")!)
///     "."
///   }
/// }
/// ```
///
/// ![](InlineImage)
public struct InlineImage: InlineContentProtocol {
  public var _inlineContent: InlineContent {
    .init(inlines: [.image(source: self.source, children: self.content.inlines)])
  }

  private let source: String
  private let content: InlineContent

  init(source: String, content: InlineContent) {
    self.source = source
    self.content = content
  }

  /// Creates an inline image with the given source
  /// - Parameter source: The absolute or relative path to the image.
  public init(source: URL) {
    self.init(source: source.absoluteString, content: .init())
  }

  /// Creates an inline image with an alternate text.
  /// - Parameters:
  ///   - text: The alternate text for the image. A ``Markdown`` view uses this text
  ///           as the accessibility label of the image.
  ///   - source: The absolute or relative path to the image.
  public init(_ text: String, source: URL) {
    self.init(source: source.absoluteString, content: .init(inlines: [.text(text)]))
  }
}
