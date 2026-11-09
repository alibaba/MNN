import MarkdownUI
import SwiftUI

struct LazyLoadingView: View {
  struct Item: Identifiable {
    let id = UUID()
    let content = MarkdownContent {
      Heading(.level2) {
        "Try MarkdownUI"
      }
      Paragraph {
        Strong("MarkdownUI")
        " is a native Markdown renderer for SwiftUI"
        " compatible with the "
        InlineLink(
          "GitHub Flavored Markdown Spec",
          destination: URL(string: "https://github.github.com/gfm/")!
        )
        "."
      }
      Paragraph {
        InlineImage(source: .randomImage())
      }
    }
  }

  private let about = """
    This screen demonstrates how you can use the `Markdown` view inside a `LazyVStack` and
    avoid re-layouts when scrolling up content. By using a custom `ImageProvider` that
    shows a placeholder while the image is loading and fixing the height of the images,
    you can avoid jumps and other weird effects caused by re-layouts when scrolling up.

    > Note that this applies only when you plan to show Markdown content with images inside
    > a `LazyVStack` or a `List`.
    """

  let items = Array(repeating: (), count: 100).map(Item.init)

  var body: some View {
    ScrollView {
      LazyVStack {
        DisclosureGroup("About this demo") {
          Markdown {
            self.about
          }
          .frame(maxWidth: .infinity, alignment: .leading)
        }
        ForEach(self.items) { item in
          Markdown(item.content)
            .padding()
        }
      }
      .padding()
    }
    .markdownTheme(.gitHub)
    // Comment this line to see the effect of having re-layouts while scrolling up.
    .markdownImageProvider(.lazyImage(aspectRatio: 4 / 3))
  }
}

struct LazyLoadingView_Previews: PreviewProvider {
  static var previews: some View {
    LazyLoadingView()
  }
}

struct LazyImageProvider: ImageProvider {
  let aspectRatio: CGFloat

  func makeImage(url: URL?) -> some View {
    AsyncImage(url: url) { phase in
      switch phase {
      case .empty, .failure:
        Color(.secondarySystemBackground)
      case .success(let image):
        image.resizable().scaledToFill()
      @unknown default:
        Color.clear
      }
    }
    .aspectRatio(self.aspectRatio, contentMode: .fill)
  }
}

extension ImageProvider where Self == LazyImageProvider {
  static func lazyImage(aspectRatio: CGFloat) -> Self {
    LazyImageProvider(aspectRatio: aspectRatio)
  }
}

extension URL {
  static func randomImage() -> URL {
    let id: String = [
      "11", "23", "26", "31", "34", "58", "63", "91", "103", "119",
    ].randomElement()!
    return URL(string: "https://picsum.photos/id/\(id)/400/300")!
  }
}
