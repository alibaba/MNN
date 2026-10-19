import SwiftUI

extension View {
  /// Sets the image provider for the Markdown images in a view hierarchy.
  /// - Parameter imageProvider: The image provider to set. Use one of the built-in values, like
  ///                            ``ImageProvider/default`` or ``ImageProvider/asset``,
  ///                            or a custom image provider that you define by creating a type that
  ///                            conforms to the ``ImageProvider`` protocol.
  /// - Returns: A view that uses the specified image provider for itself and its child views.
  public func markdownImageProvider<I: ImageProvider>(_ imageProvider: I) -> some View {
    self.environment(\.imageProvider, .init(imageProvider))
  }
}

extension EnvironmentValues {
  var imageProvider: AnyImageProvider {
    get { self[ImageProviderKey.self] }
    set { self[ImageProviderKey.self] = newValue }
  }
}

private struct ImageProviderKey: EnvironmentKey {
  static let defaultValue: AnyImageProvider = .init(.default)
}
