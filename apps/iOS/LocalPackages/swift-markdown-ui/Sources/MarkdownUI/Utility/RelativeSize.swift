import SwiftUI

/// Represents a relative size or length, such as a width, padding, or font size.
///
/// `RelativeSize` values can be used in text styles like ``FontSize`` or modifiers like
/// `markdownMargin(top:bottom:)`, `relativeFrame(width:height:alignment:)`,
/// `relativeFrame(minWidth:alignment:)`, `relativePadding(_:length:)`,
/// and `relativeLineSpacing(_:)` to express parameters relative to the font size.
///
/// Use the ``em(_:)`` and the ``rem(_:)`` methods to create values relative to the current and
/// root font sizes, respectively. For example, in the following snippet, with a root font size of 17 points,
/// the line spacing will be resolved to 4.25 points (`17 * 2 * 0.125`) and the padding to 8.5 points
/// (`17 * 0.5`).
///
/// ```swift
/// label
///   .relativeLineSpacing(.em(0.125))
///   .relativePadding(length: .rem(0.5))
///   .markdownTextStyle {
///     FontWeight(.semibold)
///     FontSize(.em(2))
///   }
/// ```
public struct RelativeSize: Hashable {
  enum Unit: Hashable {
    case em
    case rem
  }

  var value: CGFloat
  var unit: Unit
}

extension RelativeSize {
  /// A size with a value of zero.
  public static let zero = RelativeSize(value: 0, unit: .rem)

  /// Creates a size value relative to the current font size.
  public static func em(_ value: CGFloat) -> RelativeSize {
    .init(value: value, unit: .em)
  }

  /// Creates a size value relative to the root font size.
  public static func rem(_ value: CGFloat) -> RelativeSize {
    .init(value: value, unit: .rem)
  }

  func points(relativeTo fontProperties: FontProperties? = nil) -> CGFloat {
    let fontProperties = fontProperties ?? .init()

    switch self.unit {
    case .em:
      return round(value * fontProperties.scaledSize)
    case .rem:
      return round(value * fontProperties.size)
    }
  }
}

extension View {
  /// Positions this view within an invisible frame with the specified size.
  ///
  /// This method behaves like the one in SwiftUI but takes `RelativeSize`
  /// values instead of `CGFloat` for the width and height.
  public func relativeFrame(
    width: RelativeSize? = nil,
    height: RelativeSize? = nil,
    alignment: Alignment = .center
  ) -> some View {
    TextStyleAttributesReader { attributes in
      self.frame(
        width: width?.points(relativeTo: attributes.fontProperties),
        height: height?.points(relativeTo: attributes.fontProperties),
        alignment: alignment
      )
    }
  }

  /// Positions this view within an invisible frame having the specified size constraints.
  ///
  /// This method behaves like the one in SwiftUI but takes `RelativeSize`
  /// values instead of `CGFloat` for the width and height.
  public func relativeFrame(
    minWidth: RelativeSize,
    alignment: Alignment = .center
  ) -> some View {
    TextStyleAttributesReader { attributes in
      self.frame(
        minWidth: minWidth.points(relativeTo: attributes.fontProperties),
        alignment: alignment
      )
    }
  }

  /// Adds an equal padding amount to specific edges of this view.
  ///
  /// This method behaves like the one in SwiftUI except that it takes a `RelativeSize`
  /// value instead of a `CGFloat` for the padding amount.
  public func relativePadding(_ edges: Edge.Set = .all, length: RelativeSize) -> some View {
    TextStyleAttributesReader { attributes in
      self.padding(edges, length.points(relativeTo: attributes.fontProperties))
    }
  }

  /// Sets the amount of space between lines of text in this view.
  ///
  /// This method behaves like the one in SwiftUI except that it takes a `RelativeSize`
  /// value instead of a `CGFloat` for the spacing amount.
  public func relativeLineSpacing(_ lineSpacing: RelativeSize) -> some View {
    TextStyleAttributesReader { attributes in
      self.lineSpacing(lineSpacing.points(relativeTo: attributes.fontProperties))
    }
  }
}
