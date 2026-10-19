import MarkdownUI
import SDWebImageSwiftUI
import SwiftUI

struct ImageProvidersView: View {
  private let content = """
    You can tell a `Markdown` view to load images using a 3rd party library
    by configuring an `ImageProvider`. This example uses
    [**SDWebImage/SDWebImageSwiftUI**](https://github.com/SDWebImage/SDWebImageSwiftUI)
    to enable animated GIF rendering.

    ![](https://user-images.githubusercontent.com/373190/209442987-2aa9d73d-3bf2-46cb-b03a-5d9c0ab8475f.gif)
    """

  private let otherContent = """
    You can use the built-in `AssetImageProvider` and `AssetInlineImageProvider`
    to load images from image assets.

    ```swift
    Markdown {
      "![A dog](dog)"
      "A ![dog](smallDog) within a line of text."
      "― Photo by André Spieker"
    }
    .markdownImageProvider(.asset)
    .markdownInlineImageProvider(.asset)
    ```

    ![A dog](dog)

    An image ![dog](smallDog) within a line of text.

    ― Photo by André Spieker
    """

  var body: some View {
    DemoView {
      Markdown(self.content)
        .markdownImageProvider(.webImage)

      Section("Image Assets") {
        Markdown(self.otherContent)
          .markdownImageProvider(.asset)
          .markdownInlineImageProvider(.asset)
      }
    }
  }
}

struct ImageProvidersView_Previews: PreviewProvider {
  static var previews: some View {
    ImageProvidersView()
  }
}

// MARK: - WebImageProvider

struct WebImageProvider: ImageProvider {
  func makeImage(url: URL?) -> some View {
    ResizeToFit {
      WebImage(url: url)
        .resizable()
    }
  }
}

extension ImageProvider where Self == WebImageProvider {
  static var webImage: Self {
    .init()
  }
}

// MARK: - ResizeToFit

/// A layout that resizes its content to fit the container **only** if the content width is greater than the container width.
struct ResizeToFit: Layout {
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
