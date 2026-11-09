import SwiftUI

/// A type that loads images that are displayed within a line of text.
///
/// To configure the current inline image provider for a view hierarchy,
/// use the `markdownInlineImageProvider(_:)` modifier.
public protocol InlineImageProvider {
  /// Returns an image for the given URL.
  ///
  /// ``Markdown`` views call this method to load images within a line of text.
  ///
  /// - Parameters:
  ///   - url: The URL of the image to display.
  ///   - label: The accessibility label associated with the image.
  func image(with url: URL, label: String) async throws -> Image
}
