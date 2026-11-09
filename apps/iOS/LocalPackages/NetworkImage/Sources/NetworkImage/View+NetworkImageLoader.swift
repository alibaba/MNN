import SwiftUI

extension View {
  /// Sets the image loader for network images within this view.
  public func networkImageLoader<T: NetworkImageLoader>(_ networkImageLoader: T) -> some View {
    environment(\.networkImageLoader, networkImageLoader)
  }
}

extension EnvironmentValues {
  var networkImageLoader: NetworkImageLoader {
    get { self[NetworkImageLoaderKey.self] }
    set { self[NetworkImageLoaderKey.self] = newValue }
  }
}

private struct NetworkImageLoaderKey: EnvironmentKey {
  static let defaultValue: NetworkImageLoader = .default
}
