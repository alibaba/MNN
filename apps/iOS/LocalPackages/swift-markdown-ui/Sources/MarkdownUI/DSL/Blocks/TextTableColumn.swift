import Foundation

/// A column that displays styled text or images for each row in a table element.
///
/// You create a column with an optional alignment, a title, and an inline content builder. The table
/// calls the inline content builder with the value for each row in the table.
///
/// The following example creates a column for a table with `Superhero` rows, displaying each superhero's name.
///
/// ```swift
/// TextTableColumn(alignment: .leading, title: "Name") { superhero in
///   Emphasis(superhero.name)
/// }
/// ```
///
/// You can specify a key path instead of an inline content builder for unstyled `String` properties.
///
/// ```swift
/// TextTableColumn(title: "Name", value: \.name)
/// ```
public struct TextTableColumn<RowValue> {
  let alignment: TextTableColumnAlignment?
  let title: InlineContent
  let content: (RowValue) -> InlineContent

  /// Creates a column with a styled title and values.
  /// - Parameters:
  ///   - alignment: The column alignment. If not specified, the table uses a leading alignment.
  ///   - title: An inline content builder that returns the styled title of the column.
  ///   - content: An inline content builder that returns the styled content for each row value of the column.
  public init(
    alignment: TextTableColumnAlignment? = nil,
    @InlineContentBuilder title: () -> InlineContent,
    @InlineContentBuilder content: @escaping (RowValue) -> InlineContent
  ) {
    self.alignment = alignment
    self.title = title()
    self.content = content
  }

  /// Creates a column with a styled title and unstyled row values.
  /// - Parameters:
  ///   - alignment: The column alignment. If not specified, the table uses a leading alignment.
  ///   - title: An inline content builder that returns the styled title of the column.
  ///   - value: The path to the property associated with the column.
  public init<V>(
    alignment: TextTableColumnAlignment? = nil,
    @InlineContentBuilder title: () -> InlineContent,
    value: KeyPath<RowValue, V>
  ) where V: LosslessStringConvertible {
    self.init(alignment: alignment, title: title) { rowValue in
      rowValue[keyPath: value].description
    }
  }

  /// Creates a column with an unstyled title and styled row values.
  /// - Parameters:
  ///   - alignment: The column alignment. If not specified, the table uses a leading alignment.
  ///   - title: The title of the column.
  ///   - content: An inline content builder that returns the styled content for each row value of the column.
  public init(
    alignment: TextTableColumnAlignment? = nil,
    title: String,
    @InlineContentBuilder content: @escaping (RowValue) -> InlineContent
  ) {
    self.init(alignment: alignment, title: { title }, content: content)
  }

  /// Creates a column with unstyled title and row values.
  /// - Parameters:
  ///   - alignment: The column alignment. If not specified, the table uses a leading alignment.
  ///   - title: The title of the column.
  ///   - value: The path to the property associated with the column.
  public init<V>(
    alignment: TextTableColumnAlignment? = nil,
    title: String,
    value: KeyPath<RowValue, V>
  ) where V: LosslessStringConvertible {
    self.init(alignment: alignment, title: { title }, value: value)
  }
}
