import SwiftUI

/// A text style that adjusts the font width.
@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
public struct FontWidth: TextStyle {
  private let width: Font.Width

  /// Creates a font width text style.
  /// - Parameter width: The font width.
  public init(_ width: Font.Width) {
    self.width = width
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.fontProperties?.width = self.width
  }
}
