import SwiftUI

extension View {
  /// Sets the table border style for the Markdown tables in a view hierarchy.
  ///
  /// Use this modifier to customize the table border style inside the body of
  /// the ``Theme/table`` block style.
  ///
  /// - Parameter tableBorderStyle: The border style to set.
  public func markdownTableBorderStyle(_ tableBorderStyle: TableBorderStyle) -> some View {
    self.environment(\.tableBorderStyle, tableBorderStyle)
  }

  /// Sets the table background style for the Markdown tables in a view hierarchy.
  ///
  /// Use this modifier to customize the table background style inside the body of
  /// the ``Theme/table`` block style.
  ///
  /// - Parameter tableBackgroundStyle: The background style to set.
  public func markdownTableBackgroundStyle(
    _ tableBackgroundStyle: TableBackgroundStyle
  ) -> some View {
    self.environment(\.tableBackgroundStyle, tableBackgroundStyle)
  }
}

extension EnvironmentValues {
  var tableBorderStyle: TableBorderStyle {
    get { self[TableBorderStyleKey.self] }
    set { self[TableBorderStyleKey.self] = newValue }
  }

  var tableBackgroundStyle: TableBackgroundStyle {
    get { self[TableBackgroundStyleKey.self] }
    set { self[TableBackgroundStyleKey.self] = newValue }
  }
}

private struct TableBorderStyleKey: EnvironmentKey {
  static let defaultValue = TableBorderStyle(color: .secondary)
}

private struct TableBackgroundStyleKey: EnvironmentKey {
  static let defaultValue = TableBackgroundStyle.clear
}
