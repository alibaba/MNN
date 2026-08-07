import SwiftUI

/// The characteristics of a font.
public struct FontProperties: Hashable {
  /// The font family.
  public enum Family: Hashable {
    /// The system font family.
    case system(Font.Design = .default)

    /// A custom font family with the given name.
    case custom(String)
  }

  /// The font family variant.
  public enum FamilyVariant: Hashable {
    /// No variant. Use the current font family.
    case normal

    /// A fixed-width font from the current family, if available.
    case monospaced
  }

  /// The font caps variant.
  public enum CapsVariant: Hashable {
    /// Don't use a font caps variant.
    case normal

    /// Enable all small capitals.
    case smallCaps

    /// Enable lowercase small capitals.
    case lowercaseSmallCaps

    /// Enable uppercase small capitals.
    case uppercaseSmallCaps
  }

  /// The font digit variant.
  public enum DigitVariant: Hashable {
    /// Don't use a font digit variant.
    case normal

    /// Enable monospaced digits.
    case monospaced
  }

  /// The font style.
  public enum Style {
    /// Don't use a font style.
    case normal

    /// Select an italic font.
    case italic
  }

  /// A platform-dependent default font size.
  public static var defaultSize: CGFloat {
    #if os(macOS)
      return 13
    #elseif os(iOS)
      return 17
    #elseif os(tvOS)
      return 29
    #elseif os(watchOS)
      return 16
    #else
      return 16
    #endif
  }

  /// A platform-dependent default font weight.
  public static var defaultWeight: Font.Weight {
    #if os(tvOS)
      return .medium
    #else
      return .regular
    #endif
  }

  /// The font family.
  public var family: Family = .system()

  /// The font family variant.
  public var familyVariant: FamilyVariant = .normal

  /// The font caps variant.
  public var capsVariant: CapsVariant = .normal

  /// The font digit variant.
  public var digitVariant: DigitVariant = .normal

  /// The font style.
  public var style: Style = .normal

  /// The font weight.
  public var weight: Font.Weight = Self.defaultWeight

  /// The font width.
  @available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
  public var width: Font.Width {
    get { (self.widthStorage as? Font.Width) ?? .standard }
    set { self.widthStorage = newValue }
  }

  private var widthStorage: AnyHashable?

  /// The font size.
  public var size: CGFloat = Self.defaultSize

  /// The font scale.
  public var scale: CGFloat = 1

  /// The scaled font size.
  public var scaledSize: CGFloat {
    round(self.size * self.scale)
  }

  /// Creates a font  properties value.
  /// - Parameters:
  ///   - family: The font family.
  ///   - familyVariant: The font family variant.
  ///   - capsVariant: The font caps variant.
  ///   - digitVariant: The font digit variant.
  ///   - style: The font style.
  ///   - weight: The font weight.
  ///   - width: The font width
  ///   - size: The font size.
  ///   - scale: The font scale.
  @available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
  public init(
    family: FontProperties.Family = .system(),
    familyVariant: FontProperties.FamilyVariant = .normal,
    capsVariant: FontProperties.CapsVariant = .normal,
    digitVariant: FontProperties.DigitVariant = .normal,
    style: FontProperties.Style = .normal,
    weight: Font.Weight = Self.defaultWeight,
    width: Font.Width,
    size: CGFloat = Self.defaultSize,
    scale: CGFloat = 1
  ) {
    self.init(
      family: family,
      familyVariant: familyVariant,
      capsVariant: capsVariant,
      digitVariant: digitVariant,
      style: style,
      weight: weight,
      size: size,
      scale: scale
    )
    self.width = width
  }

  /// Creates a font  properties value.
  /// - Parameters:
  ///   - family: The font family.
  ///   - familyVariant: The font family variant.
  ///   - capsVariant: The font caps variant.
  ///   - digitVariant: The font digit variant.
  ///   - style: The font style.
  ///   - weight: The font weight.
  ///   - size: The font size.
  ///   - scale: The font scale.
  public init(
    family: FontProperties.Family = .system(),
    familyVariant: FontProperties.FamilyVariant = .normal,
    capsVariant: FontProperties.CapsVariant = .normal,
    digitVariant: FontProperties.DigitVariant = .normal,
    style: FontProperties.Style = .normal,
    weight: Font.Weight = Self.defaultWeight,
    size: CGFloat = Self.defaultSize,
    scale: CGFloat = 1
  ) {
    self.family = family
    self.familyVariant = familyVariant
    self.capsVariant = capsVariant
    self.digitVariant = digitVariant
    self.style = style
    self.weight = weight
    self.size = size
    self.scale = scale
  }
}

extension FontProperties: TextStyle {
  public func _collectAttributes(in attributes: inout AttributeContainer) {
    attributes.fontProperties = self
  }
}
