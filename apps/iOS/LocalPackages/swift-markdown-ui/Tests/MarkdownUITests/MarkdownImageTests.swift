#if os(iOS)
  import SnapshotTesting
  import SwiftUI
  import XCTest

  import MarkdownUI

  final class MarkdownImageTests: XCTestCase {
    private let layout = SwiftUISnapshotLayout.device(config: .iPhone8)

    override func setUpWithError() throws {
      try XCTSkipIf(UIDevice.current.userInterfaceIdiom == .pad, "Skipping on Mac Catalyst")
    }

    func testFailingImage() {
      let view = Markdown {
        #"""
        An image that fails to load:

        ![](https://picsum.photos/500/300)

        ― Photo by André Spieker
        """#
      }
      .border(Color.accentColor)
      .padding()

      assertSnapshot(of: view, as: .image(layout: layout))
    }

    func testRelativeImage() {
      let view = Markdown(baseURL: URL(string: "https://example.com/picsum/")) {
        #"""
        500x300 image:

        ![](237-500x300)

        ― Photo by André Spieker
        """#
      }
      .border(Color.accentColor)
      .padding()
      .markdownImageProvider(
        AssetImageProvider(
          name: { url in
            XCTAssertEqual(
              URL(string: "237-500x300", relativeTo: URL(string: "https://example.com/picsum/"))!,
              url
            )
            return url.lastPathComponent
          },
          bundle: .module
        )
      )

      assertSnapshot(of: view, as: .image(layout: layout))
    }

    func testImageLink() {
      let view = Markdown {
        #"""
        A link that contains an image instead of text:

        [![](https://example.com/picsum/237-100x150)](https://example.com)

        ― Photo by André Spieker
        """#
      }
      .border(Color.accentColor)
      .padding()
      .markdownImageProvider(AssetImageProvider(bundle: .module))

      assertSnapshot(of: view, as: .image(layout: layout))
    }

    func testMultipleImages() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Required API is not available for this test")
      }

      let view = Markdown {
        #"""
        [![](https://example.com/picsum/237-100x150)](https://example.com)
        ![](https://example.com/picsum/237-125x75)
        ![](https://example.com/picsum/237-500x300)
        ![](https://example.com/picsum/237-100x150)\#u{20}\#u{20}
        ![](https://example.com/picsum/237-125x75)

        ― Photo by André Spieker
        """#
      }
      .border(Color.accentColor)
      .padding()
      .markdownImageProvider(AssetImageProvider(bundle: .module))

      assertSnapshot(of: view, as: .image(layout: layout))
    }

    func testMultipleImagesSize() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Required API is not available for this test")
      }

      let view = Markdown {
        #"""
        ![](https://example.com/picsum/237-100x150)
        ![](https://example.com/picsum/237-125x75)

        ― Photo by André Spieker
        """#
      }
      .border(Color.accentColor)
      .padding()
      .markdownImageProvider(AssetImageProvider(bundle: .module))

      assertSnapshot(of: view, as: .image(layout: layout))
    }

    func testColorScheme() {
      let content = """
        This image is contextualized for either dark or light mode:

        ![](https://example.com/picsum/237-100x150#gh-dark-mode-only)
        ![](https://example.com/picsum/237-125x75#gh-light-mode-only)

        ― Photo by André Spieker
        """

      let view = VStack {
        Markdown(content)
          .background()
          .colorScheme(.light)
          .border(Color.accentColor)
          .padding()
        Markdown(content)
          .background()
          .colorScheme(.dark)
          .border(Color.accentColor)
          .padding()
      }
      .markdownImageProvider(AssetImageProvider(bundle: .module))

      assertSnapshot(of: view, as: .image(layout: layout))
    }
  }
#endif
