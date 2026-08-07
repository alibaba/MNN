import SwiftUI

/// A text style that adjusts the font weight.
public struct FontWeight: TextStyle {
  private let weight: Font.Weight

  /// Creates a font weight text style.
  /// - Parameter weight: The font weight.
  public init(_ weight: Font.Weight) {
    self.weight = weight
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.fontProperties?.weight = self.weight
  }
}
