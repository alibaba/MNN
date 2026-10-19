import SwiftUI

// MARK: - Deprecated after 5.0.0:

extension NetworkImage {
  @available(
    *,
    deprecated,
    message:
      "Use the initializer that takes a content closure receiving a 'NetworkImageState' value."
  )
  public init<P, I, F>(
    url: URL?,
    scale: CGFloat = 1,
    transaction: Transaction = .init(),
    @ViewBuilder content: @escaping (Image) -> I,
    @ViewBuilder placeholder: @escaping () -> P,
    @ViewBuilder fallback: @escaping () -> F
  ) where Content == _ConditionalContent<_ConditionalContent<P, I>, F>, P: View, I: View, F: View {
    self.init(
      url: url,
      scale: scale,
      transaction: transaction,
      content: { state in
        switch state {
        case .empty:
          placeholder()
        case .success(let image, _):
          content(image)
        case .failure:
          fallback()
        }
      }
    )
  }
}
