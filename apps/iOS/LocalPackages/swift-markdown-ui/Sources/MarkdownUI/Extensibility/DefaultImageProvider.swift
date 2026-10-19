import NetworkImage
import SwiftUI

/// The default image provider, which loads images from the network.
public struct DefaultImageProvider: ImageProvider {
  public func makeImage(url: URL?) -> some View {
    NetworkImage(url: url) { state in
      switch state {
      case .empty, .failure:
        Color.clear
          .frame(width: 0, height: 0)
      case .success(let image, let idealSize):
        ResizeToFit(idealSize: idealSize) {
          image.resizable()
        }
      }
    }
  }
}

extension ImageProvider where Self == DefaultImageProvider {
  /// The default image provider, which loads images from the network.
  ///
  /// Use the `markdownImageProvider(_:)` modifier to configure this image provider for a view hierarchy.
  public static var `default`: Self {
    .init()
  }
}
