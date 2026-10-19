import SwiftUI

extension View {
  /// Sets the default text style for the Markdown inlines in a view hierarchy.
  ///
  /// Use this modifier inside a ``BlockStyle`` `body` block to customize
  /// the default text style for the block's Markdown inlines.
  ///
  /// - Parameter textStyle: A text style builder that returns the text style to use.
  public func markdownTextStyle<S: TextStyle>(
    @TextStyleBuilder textStyle: @escaping () -> S
  ) -> some View {
    self.transformEnvironment(\.textStyle) {
      $0 = $0.appending(textStyle())
    }
  }

  func textStyleFont() -> some View {
    TextStyleAttributesReader { attributes in
      self.font(attributes.fontProperties.map(Font.withProperties))
    }
  }

  func textStyleForegroundColor() -> some View {
    TextStyleAttributesReader { attributes in
      self.foregroundColor(attributes.foregroundColor)
    }
  }

  func textStyle(_ textStyle: TextStyle) -> some View {
    self.transformEnvironment(\.textStyle) {
      $0 = $0.appending(textStyle)
    }
  }
}

extension TextStyle {
  @TextStyleBuilder fileprivate func appending<S: TextStyle>(
    _ textStyle: S
  ) -> some TextStyle {
    self
    textStyle
  }
}

extension EnvironmentValues {
  fileprivate(set) var textStyle: TextStyle {
    get { self[TextStyleKey.self] }
    set { self[TextStyleKey.self] = newValue }
  }
}

private struct TextStyleKey: EnvironmentKey {
  static let defaultValue: TextStyle = FontProperties()
}
