import Foundation

/// A result builder you can use to compose text styles.
@resultBuilder public enum TextStyleBuilder {
  public static func buildBlock() -> some TextStyle {
    EmptyTextStyle()
  }

  public static func buildBlock(_ component: some TextStyle) -> some TextStyle {
    component
  }

  public static func buildEither<S0: TextStyle, S1: TextStyle>(
    first component: S0
  ) -> _Conditional<S0, S1> {
    _Conditional<S0, S1>.first(component)
  }

  public static func buildEither<S0: TextStyle, S1: TextStyle>(
    second component: S1
  ) -> _Conditional<S0, S1> {
    _Conditional<S0, S1>.second(component)
  }

  public static func buildLimitedAvailability(
    _ component: some TextStyle
  ) -> any TextStyle {
    component
  }

  public static func buildOptional(_ component: (some TextStyle)?) -> some TextStyle {
    component
  }

  public static func buildPartialBlock(first: some TextStyle) -> some TextStyle {
    first
  }

  public static func buildPartialBlock(
    accumulated: some TextStyle,
    next: some TextStyle
  ) -> some TextStyle {
    Pair(accumulated, next)
  }

  public enum _Conditional<First: TextStyle, Second: TextStyle>: TextStyle {
    case first(First)
    case second(Second)

    public func _collectAttributes(in attributes: inout AttributeContainer) {
      switch self {
      case .first(let first):
        first._collectAttributes(in: &attributes)
      case .second(let second):
        second._collectAttributes(in: &attributes)
      }
    }
  }

  private struct Pair<S0: TextStyle, S1: TextStyle>: TextStyle {
    let s0: S0
    let s1: S1

    init(_ s0: S0, _ s1: S1) {
      self.s0 = s0
      self.s1 = s1
    }

    func _collectAttributes(in attributes: inout AttributeContainer) {
      s0._collectAttributes(in: &attributes)
      s1._collectAttributes(in: &attributes)
    }
  }
}

extension Optional: TextStyle where Wrapped: TextStyle {
  public func _collectAttributes(in attributes: inout AttributeContainer) {
    self?._collectAttributes(in: &attributes)
  }
}
