import NetworkImage
import SwiftUI

struct ImageListView: View {
  private struct Item: Identifiable {
    let id = UUID()
    let imageURL = URL.randomImageURL(size: .init(width: 400, height: 300))
  }

  private let items = Array(repeating: (), count: 100).map(Item.init)

  var body: some View {
    ScrollView {
      LazyVStack {
        ForEach(self.items) { item in
          NetworkImage(url: item.imageURL, transaction: .init(animation: .default)) { image in
            image.resizable().scaledToFill()
          }
          .aspectRatio(1.333, contentMode: .fill)
          .clipShape(RoundedRectangle(cornerRadius: 4))
          .overlay {
            RoundedRectangle(cornerRadius: 4)
              // This crashes in Xcode 15 beta 8 when running on visionOS simulator
              // .strokeBorder(Color.primary.opacity(0.25), lineWidth: 0.5)
              .strokeBorder(style: .init(lineWidth: 0.5))
              .foregroundColor(Color.primary.opacity(0.25))

          }
        }
      }
      .padding()
    }
  }
}

struct ImageListView_Previews: PreviewProvider {
  static var previews: some View {
    ImageListView()
  }
}
