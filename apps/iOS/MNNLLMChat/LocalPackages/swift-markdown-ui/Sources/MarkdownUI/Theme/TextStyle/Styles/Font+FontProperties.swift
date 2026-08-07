import SwiftUI

extension Font {
  static func withProperties(_ fontProperties: FontProperties) -> Font {
    var font: Font
    let size = round(fontProperties.size * fontProperties.scale)

    switch fontProperties.family {
    case .system(let design):
      font = .system(size: size, design: design)
    case .custom(let name):
      font = .custom(name, fixedSize: size)
    }

    switch fontProperties.familyVariant {
    case .normal:
      break  // do nothing
    case .monospaced:
      font = font.monospaced()
    }

    switch fontProperties.capsVariant {
    case .normal:
      break  // do nothing
    case .smallCaps:
      font = font.smallCaps()
    case .lowercaseSmallCaps:
      font = font.lowercaseSmallCaps()
    case .uppercaseSmallCaps:
      font = font.uppercaseSmallCaps()
    }

    switch fontProperties.digitVariant {
    case .normal:
      break  // do nothing
    case .monospaced:
      font = font.monospacedDigit()
    }

    if fontProperties.weight != .regular {
      font = font.weight(fontProperties.weight)
    }

    if #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) {
      if fontProperties.width != .standard {
        font = font.width(fontProperties.width)
      }
    }

    switch fontProperties.style {
    case .normal:
      break  // do nothing
    case .italic:
      font = font.italic()
    }

    return font
  }
}
