import SwiftUI

/// A text style that sets the underline line style of the text.
public struct UnderlineStyle: TextStyle {
  private let lineStyle: Text.LineStyle?

  /// Creates an underline style text style.
  /// - Parameter lineStyle: The line style.
  public init(_ lineStyle: Text.LineStyle?) {
    self.lineStyle = lineStyle
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.underlineStyle = self.lineStyle
  }
}
