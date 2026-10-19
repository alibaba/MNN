import Foundation

/// A text style that sets the font size.
public struct FontSize: TextStyle {
  private enum Size {
    case points(CGFloat)
    case relative(RelativeSize)
  }

  private let size: Size

  /// Creates a font size text style that sets the size to a relative value.
  /// - Parameter relativeSize: The relative size of the font.
  public init(_ relativeSize: RelativeSize) {
    self.size = .relative(relativeSize)
  }

  /// Creates a font size text style that sets the size to a given value.
  /// - Parameter size: The size of the font measured in points.
  public init(_ size: CGFloat) {
    self.size = .points(size)
  }

  public func _collectAttributes(in attributes: inout AttributeContainer) {
    switch self.size {
    case .points(let value):
      attributes.fontProperties?.size = value
      attributes.fontProperties?.scale = 1
    case .relative(let relativeSize):
      switch relativeSize.unit {
      case .em:
        attributes.fontProperties?.scale *= relativeSize.value
      case .rem:
        attributes.fontProperties?.scale = relativeSize.value
      }
    }
  }
}
