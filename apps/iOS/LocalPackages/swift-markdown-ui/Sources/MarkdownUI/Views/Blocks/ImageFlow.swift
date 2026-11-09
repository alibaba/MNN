import SwiftUI

@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
struct ImageFlow: View {
  private enum Item: Hashable {
    case image(RawImageData)
    case lineBreak
  }

  private let items: [Indexed<Item>]

  var body: some View {
    TextStyleAttributesReader { attributes in
      let spacing = RelativeSize.rem(0.25).points(relativeTo: attributes.fontProperties)

      FlowLayout(horizontalSpacing: spacing, verticalSpacing: spacing) {
        ForEach(self.items, id: \.self) { item in
          switch item.value {
          case .image(let data):
            ImageView(data: data)
          case .lineBreak:
            Spacer()
          }
        }
      }
    }
  }
}

@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
extension ImageFlow {
  init?(_ inlines: [InlineNode]) {
    var items: [Item] = []

    for inline in inlines {
      switch inline {
      case let .text(text) where text.isEmpty:
        continue
      case .softBreak:
        continue
      case .lineBreak:
        items.append(.lineBreak)
      case let .image(source, children):
        items.append(.image(.init(source: source, alt: children.renderPlainText())))
      case let .link(destination, children) where children.count == 1:
        guard var data = children.first?.imageData else {
          return nil
        }
        data.destination = destination
        items.append(.image(data))
      default:
        return nil
      }
    }

    self.items = items.indexed()
  }
}
