import SwiftUI

extension View {
  /// Sets the current ``Theme`` for the Markdown contents in a view hierarchy.
  /// - Parameter theme: The theme to set.
  public func markdownTheme(_ theme: Theme) -> some View {
    self.environment(\.theme, theme)
  }

  /// Replaces a specific text style of the current ``Theme`` with the given text style.
  /// - Parameters:
  ///   - keyPath: The ``Theme`` key path to the text style to replace.
  ///   - textStyle: A text style builder that returns the new text style to use for the given key path.
  public func markdownTextStyle<S: TextStyle>(
    _ keyPath: WritableKeyPath<Theme, TextStyle>,
    @TextStyleBuilder textStyle: () -> S
  ) -> some View {
    self.environment((\EnvironmentValues.theme).appending(path: keyPath), textStyle())
  }

  /// Replaces a specific block style on the current ``Theme`` with a block style initialized with the given body closure.
  /// - Parameters:
  ///   - keyPath: The ``Theme`` key path to the block style to replace.
  ///   - body: A view builder that returns the customized block.
  public func markdownBlockStyle<Body: View>(
    _ keyPath: WritableKeyPath<Theme, BlockStyle<Void>>,
    @ViewBuilder body: @escaping () -> Body
  ) -> some View {
    self.environment((\EnvironmentValues.theme).appending(path: keyPath), .init(body: body))
  }

  /// Replaces a specific block style on the current ``Theme`` with a block style initialized with the given body closure.
  /// - Parameters:
  ///   - keyPath: The ``Theme`` key path to the block style to replace.
  ///   - body: A view builder that receives the block configuration and returns the customized block.
  public func markdownBlockStyle<Configuration, Body: View>(
    _ keyPath: WritableKeyPath<Theme, BlockStyle<Configuration>>,
    @ViewBuilder body: @escaping (_ configuration: Configuration) -> Body
  ) -> some View {
    self.environment((\EnvironmentValues.theme).appending(path: keyPath), .init(body: body))
  }

  /// Replaces the current ``Theme`` task list marker with the given list marker.
  public func markdownTaskListMarker(
    _ value: BlockStyle<TaskListMarkerConfiguration>
  ) -> some View {
    self.environment(\.theme.taskListMarker, value)
  }

  /// Replaces the current ``Theme`` bulleted list marker with the given list marker.
  public func markdownBulletedListMarker(
    _ value: BlockStyle<ListMarkerConfiguration>
  ) -> some View {
    self.environment(\.theme.bulletedListMarker, value)
  }

  /// Replaces the current ``Theme`` numbered list marker with the given list marker.
  public func markdownNumberedListMarker(
    _ value: BlockStyle<ListMarkerConfiguration>
  ) -> some View {
    self.environment(\.theme.numberedListMarker, value)
  }
}

extension EnvironmentValues {
  var theme: Theme {
    get { self[ThemeKey.self] }
    set { self[ThemeKey.self] = newValue }
  }
}

private struct ThemeKey: EnvironmentKey {
  static let defaultValue: Theme = .basic
}
