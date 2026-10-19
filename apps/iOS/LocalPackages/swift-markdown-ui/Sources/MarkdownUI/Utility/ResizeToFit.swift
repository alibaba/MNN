import SwiftUI

struct ResizeToFit<Content>: View where Content: View {
  private let idealSize: CGSize
  private let content: Content

  init(idealSize: CGSize, @ViewBuilder content: () -> Content) {
    self.idealSize = idealSize
    self.content = content()
  }

  var body: some View {
    if #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) {
      ResizeToFit2 { self.content }
    } else {
      ResizeToFit1(idealSize: self.idealSize, content: self.content)
    }
  }
}

// MARK: - Geometry reader based

private struct ResizeToFit1<Content>: View where Content: View {
  @State private var size: CGSize?

  let idealSize: CGSize
  let content: Content

  var body: some View {
    GeometryReader { proxy in
      let size = self.sizeThatFits(proposal: proxy.size)
      self.content
        .frame(width: size.width, height: size.height)
        .preference(key: SizePreference.self, value: size)
    }
    .frame(width: size?.width, height: size?.height)
    .onPreferenceChange(SizePreference.self) { size in
      self.size = size
    }
  }

  private func sizeThatFits(proposal: CGSize) -> CGSize {
    guard proposal.width < idealSize.width else {
      return idealSize
    }

    let aspectRatio = idealSize.width / idealSize.height
    return CGSize(width: proposal.width, height: proposal.width / aspectRatio)
  }
}

private struct SizePreference: PreferenceKey {
  static let defaultValue: CGSize? = nil

  static func reduce(value: inout CGSize?, nextValue: () -> CGSize?) {
    value = value ?? nextValue()
  }
}

// MARK: - Layout based

@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
private struct ResizeToFit2: Layout {
  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
    guard let view = subviews.first else {
      return .zero
    }

    var size = view.sizeThatFits(.unspecified)

    if let width = proposal.width, size.width > width {
      let aspectRatio = size.width / size.height
      size.width = width
      size.height = width / aspectRatio
    }
    return size
  }

  func placeSubviews(
    in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
  ) {
    guard let view = subviews.first else { return }
    view.place(at: bounds.origin, proposal: .init(bounds.size))
  }
}
