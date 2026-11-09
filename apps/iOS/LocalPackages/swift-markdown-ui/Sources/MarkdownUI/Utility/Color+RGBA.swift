import SwiftUI

extension Color {
  /// Creates a constant color from an RGBA value.
  /// - Parameter rgba: A 32-bit value that represents the red, green, blue, and alpha components of the color.
  public init(rgba: UInt32) {
    self.init(
      red: CGFloat((rgba & 0xff00_0000) >> 24) / 255.0,
      green: CGFloat((rgba & 0x00ff_0000) >> 16) / 255.0,
      blue: CGFloat((rgba & 0x0000_ff00) >> 8) / 255.0,
      opacity: CGFloat(rgba & 0x0000_00ff) / 255.0
    )
  }

  /// Creates a context-dependent color with different values for light and dark appearances.
  /// - Parameters:
  ///   - light: The light appearance color value.
  ///   - dark: The dark appearance color value.
  public init(light: @escaping @autoclosure () -> Color, dark: @escaping @autoclosure () -> Color) {
    #if os(watchOS)
      self = dark()
    #elseif canImport(UIKit)
      self.init(
        uiColor: .init { traitCollection in
          switch traitCollection.userInterfaceStyle {
          case .unspecified, .light:
            return UIColor(light())
          case .dark:
            return UIColor(dark())
          @unknown default:
            return UIColor(light())
          }
        }
      )
    #elseif canImport(AppKit)
      self.init(
        nsColor: .init(name: nil) { appearance in
          if appearance.bestMatch(from: [.aqua, .darkAqua]) == .aqua {
            return NSColor(light())
          } else {
            return NSColor(dark())
          }
        }
      )
    #endif
  }
}
