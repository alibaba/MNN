#if os(iOS)
  import SnapshotTesting
  import SwiftUI
  import XCTest

  import MarkdownUI

  final class MarkdownTableTests: XCTestCase {
    private let layout = SwiftUISnapshotLayout.device(config: .iPhone8)
    private let perceptualPrecision: Float = 0.98

    func testTable() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | Command | Description |
        | --- | --- |
        | git status | List all new or modified files |
        | git diff | Show file differences that haven't been staged |
        """#
      }
      .padding()
      .border(Color.accentColor)

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testTableAlignment() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | Default    | Leading    | Center     | Trailing   |
        | ---        | :---       |    :---:   |       ---: |
        | git status | git status | git status | git status |
        | git diff   | git diff   | git diff   | git diff   |
        """#
      }
      .padding()
      .border(Color.accentColor)

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testTableWithImages() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | First Header  | Second Header |
        | --- | --- |
        | ![](https://example.com/picsum/237-100x150) | ![](https://example.com/picsum/237-125x75) |
        | ![](https://example.com/picsum/237-500x300) | ![](https://example.com/picsum/237-100x150) |

        ― Photo by André Spieker
        """#
      }
      .padding()
      .border(Color.accentColor)
      .markdownImageProvider(AssetImageProvider(bundle: .module))

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testEmptyTable() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | First Header  | Second Header |
        | ------------- | ------------- |
        """#
      }
      .padding()
      .border(Color.accentColor)

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testTableSize() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | First Header  | Second Header |
        | ------------- | ------------- |
        | Content Cell  | Content Cell  |
        | Content Cell  | Content Cell  |
        """#
      }
      .padding()
      .border(Color.accentColor)

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testTableBackground() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | Command | Description |
        | --- | --- |
        | git status | List all new or modified files |
        | git diff | Show file differences that haven't been staged |
        """#
      }
      .padding()
      .border(Color.accentColor)
      .markdownBlockStyle(\.table) { configuration in
        configuration.label
          .markdownMargin(top: .zero, bottom: .em(1))
          .markdownTableBackgroundStyle(
            .alternatingRows(Color.clear, Color(.secondarySystemBackground), header: .mint)
          )
      }

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }

    func testTableBorder() throws {
      guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
        throw XCTSkip("Table rendering is not available")
      }

      let view = Markdown {
        #"""
        A table with some padding:

        | Command | Description |
        | --- | --- |
        | git status | List all new or modified files |
        | git diff | Show file differences that haven't been staged |
        """#
      }
      .padding()
      .border(Color.accentColor)
      .markdownBlockStyle(\.table) { configuration in
        configuration.label
          .markdownMargin(top: .zero, bottom: .em(1))
          .markdownTableBorderStyle(
            .init(
              .outsideBorders,
              color: Color.mint,
              strokeStyle: .init(lineWidth: 2, lineJoin: .round, dash: [4])
            )
          )
      }

      assertSnapshot(
        of: view, as: .image(perceptualPrecision: perceptualPrecision, layout: layout)
      )
    }
  }
#endif
