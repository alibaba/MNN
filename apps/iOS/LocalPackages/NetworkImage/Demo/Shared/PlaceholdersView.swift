import NetworkImage
import SwiftUI

struct PlaceholdersView: View {
  private let url = URL(string: "https://picsum.photos/id/1025/300/200")
  private let invalidURL = URL(string: "https://example.com")

  private func exampleImage(url: URL?) -> some View {
    NetworkImage(url: url, transaction: .init(animation: .default)) { state in
      switch state {
      case .empty:
        ProgressView()
      case .success(let image, _):
        image
      case .failure:
        Image(systemName: "photo.fill")
          .imageScale(.large)
          .blendMode(.overlay)
      }
    }
    .frame(width: 200, height: 200)
    .background(Color.secondary.opacity(0.25))
    .clipShape(RoundedRectangle(cornerRadius: 8))
  }

  var body: some View {
    Form {
      Section {
        Text("This example uses a custom image loader that delays the loading of images.")
      }

      Section("Successful image") {
        self.exampleImage(url: self.url)
      }

      Section("Failing image") {
        self.exampleImage(url: self.invalidURL)
      }
    }
    .networkImageLoader(DelayNetworkImageLoader(delay: 2))
  }
}

struct PlaceholdersView_Previews: PreviewProvider {
  static var previews: some View {
    PlaceholdersView()
  }
}

private final class DelayNetworkImageLoader: NetworkImageLoader {
  private let delay: TimeInterval

  init(delay: TimeInterval) {
    self.delay = delay
  }

  func image(from url: URL) async throws -> CGImage {
    try await Task.sleep(nanoseconds: UInt64(self.delay * 1_000_000_000))
    return try await DefaultNetworkImageLoader.shared.image(from: url)
  }
}
