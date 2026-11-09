import Foundation

/// A row that represents a data value in a table.
///
/// Create instances of `TextTableRow` in the closure you provide to the `rows` parameter in
/// the ``TextTable/init(columns:rows:)`` initializer. The table provides the value of a
/// row to each column, which produces the cells for each row in the column.
public struct TextTableRow<Value> {
  let value: Value

  /// Creates a table row for the given value.
  ///
  /// The table provides the value of a row to each column, which produces the cells for each row in the column.
  ///
  /// The following example creates a row for one instance of the `Savings` type. The table delivers this
  /// value to its columns, which renders different fields of `Savings`.
  ///
  /// ```swift
  /// TextTableRow(Savings(month: "January", amount: 100))
  /// ```
  ///
  /// - Parameter value: The value of the row.
  public init(_ value: Value) {
    self.value = value
  }
}
