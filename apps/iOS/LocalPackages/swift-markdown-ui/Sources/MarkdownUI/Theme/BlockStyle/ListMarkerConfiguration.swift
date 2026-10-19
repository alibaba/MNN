import SwiftUI

/// The properties of a list marker in a mardown list.
///
/// The theme ``Theme/bulletedListMarker`` and ``Theme/numberedListMarker``
/// block styles receive a `ListMarkerConfiguration` input in their `body` closure.
public struct ListMarkerConfiguration {
  /// The list level (one-based) of the item to which the marker applies.
  public let listLevel: Int

  /// The position (one-based) of the item to which the marker applies.
  public let itemNumber: Int
}

extension BlockStyle where Configuration == ListMarkerConfiguration {
  /// A list marker style that uses decimal numbers beginning with 1.
  public static var decimal: Self {
    BlockStyle { configuration in
      Text("\(configuration.itemNumber).")
        .monospacedDigit()
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses uppercase roman numerals beginning with `I`.
  public static var upperRoman: Self {
    BlockStyle { configuration in
      Text(configuration.itemNumber.roman + ".")
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses lowercase roman numerals beginning with `i`.
  public static var lowerRoman: Self {
    BlockStyle { configuration in
      Text(configuration.itemNumber.roman.lowercased() + ".")
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses a dash.
  public static var dash: Self {
    BlockStyle { _ in
      Text("-")
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses a filled circle.
  public static var disc: Self {
    BlockStyle { _ in
      ListBullet.disc
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses a hollow circle.
  public static var circle: Self {
    BlockStyle { _ in
      ListBullet.circle
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that uses a filled square.
  public static var square: Self {
    BlockStyle { _ in
      ListBullet.square
        .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
    }
  }

  /// A list marker style that alternates between disc, circle, and square, depending on the list level.
  public static var discCircleSquare: Self {
    BlockStyle { configuration in
      let styles: [Self] = [.disc, .circle, .square]
      styles[min(configuration.listLevel, styles.count) - 1]
        .makeBody(configuration: configuration)
    }
  }
}
