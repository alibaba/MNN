import SwiftUI

/// The properties of a Markdown table cell.
///
/// The theme ``Theme/tableCell`` block style receives a `TableCellConfiguration`
/// input in its `body` closure.
public struct TableCellConfiguration {
  /// A type-erased view of a Markdown table cell.
  public struct Label: View {
    init<L: View>(_ label: L) {
      self.body = AnyView(label)
    }

    public let body: AnyView
  }

  /// The table cell row index.
  public let row: Int

  /// The table cell column index.
  public let column: Int

  /// The table cell view.
  public let label: Label

  /// The table cell content.
  public let content: MarkdownContent
}
