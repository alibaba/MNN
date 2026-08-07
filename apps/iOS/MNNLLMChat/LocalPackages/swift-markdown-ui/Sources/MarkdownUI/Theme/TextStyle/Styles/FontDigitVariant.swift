import Foundation

/// A text style that adjusts the font to use alternate glyphs for digits.
public struct FontDigitVariant: TextStyle {
  private let digitVariant: FontProperties.DigitVariant

  /// Creates a font digit variant text style.
  /// - Parameter digitVariant: The font digit variant.
  public init(_ digitVariant: FontProperties.DigitVariant) {
    self.digitVariant = digitVariant
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.fontProperties?.digitVariant = self.digitVariant
  }
}
