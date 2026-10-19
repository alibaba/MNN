import SwiftUI

extension EnvironmentValues {
  var baseURL: URL? {
    get { self[BaseURLKey.self] }
    set { self[BaseURLKey.self] = newValue }
  }

  var imageBaseURL: URL? {
    get { self[ImageBaseURLKey.self] }
    set { self[ImageBaseURLKey.self] = newValue }
  }
}

private struct BaseURLKey: EnvironmentKey {
  static var defaultValue: URL? = nil
}

private struct ImageBaseURLKey: EnvironmentKey {
  static var defaultValue: URL? = nil
}
