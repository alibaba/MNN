import SwiftUI

/// A text style that sets the strikethrough line style of the text.
public struct StrikethroughStyle: TextStyle {
  private let lineStyle: Text.LineStyle?

  /// Creates a strikethrough text style.
  /// - Parameter lineStyle: The line style.
  public init(_ lineStyle: Text.LineStyle?) {
    self.lineStyle = lineStyle
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.strikethroughStyle = self.lineStyle
  }
}
