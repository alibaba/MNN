import SwiftUI

@MainActor final class NetworkImageModel: ObservableObject {
  @Published private(set) var source: ImageSource?
  @Published private(set) var image: NetworkImageState = .empty

  private var transaction = Transaction()
  private var imageLoader: NetworkImageLoader = DefaultNetworkImageLoader.shared

  // MARK: Actions

  func sourceChanged(_ source: ImageSource?) async {
    guard source != self.source else { return }

    self.source = source

    if let source {
      await loadImage(source: source)
    }
  }

  func transactionChanged(_ transaction: Transaction) {
    self.transaction = transaction
  }

  func imageLoaderChanged(_ imageLoader: NetworkImageLoader) {
    self.imageLoader = imageLoader
  }

  private func loadImageFinished(_ cgImage: CGImage, scale: CGFloat) {
    withTransaction(transaction) {
      self.image = .success(
        image: Image(decorative: cgImage, scale: scale),
        idealSize: CGSize(
          width: CGFloat(cgImage.width) / scale,
          height: CGFloat(cgImage.height) / scale
        )
      )
    }
  }

  private func loadImageFailed(_: Error) {
    self.image = .failure
  }

  // MARK: Effects

  private func loadImage(source: ImageSource) async {
    do {
      let cgImage = try await imageLoader.image(from: source.url)
      loadImageFinished(cgImage, scale: source.scale)
    } catch {
      loadImageFailed(error)
    }
  }
}
