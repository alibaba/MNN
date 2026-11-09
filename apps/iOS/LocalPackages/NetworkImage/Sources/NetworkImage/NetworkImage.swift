import SwiftUI

/// A view that displays an image located at a given URL.
///
/// A network image downloads and displays an image from a given URL; the download is asynchronous,
/// and the result is cached both in disk and memory.
///
/// The simplest way of creating a `NetworkImage` view is to pass the image URL to the
/// ``NetworkImage/init(url:scale:)`` initializer.
///
/// ```swift
/// NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200"))
///   .frame(width: 300, height: 200)
/// ```
///
/// To manipulate the loaded image, use the `content` parameter in
/// ``NetworkImage/init(url:scale:transaction:content:)-94eq8``.
///
/// ```swift
/// NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { image in
///   image
///     .resizable()
///     .scaledToFill()
///     .blur(radius: 4)
/// }
/// .frame(width: 150, height: 150)
/// .clipped()
/// ```
///
/// The view displays a standard placeholder that fills the available space until the image loads. You can
/// specify a custom placeholder by using the `placeholder` parameter in
/// ``NetworkImage/init(url:scale:transaction:content:placeholder:)``.
///
/// ```swift
/// NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { image in
///   image
///     .resizable()
///     .scaledToFill()
/// } placeholder: {
///   ZStack {
///     Color.secondary.opacity(0.25)
///     Image(systemName: "photo.fill")
///       .imageScale(.large)
///       .blendMode(.overlay)
///   }
/// }
/// .frame(width: 150, height: 150)
/// .clipped()
/// ```
///
/// To have more control over the image loading process, use
/// ``NetworkImage/init(url:scale:transaction:content:)-4lwrd``, which has a `content` closure parameter that
/// receives a ``NetworkImageState`` value indicating the state of the loading operation.
///
/// ```swift
/// NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { state in
///   switch state {
///   case .empty:
///     ProgressView()
///   case .success(let image, let idealSize):
///     image
///       .resizable()
///       .scaledToFill()
///   case .failure:
///     Image(systemName: "photo.fill")
///       .imageScale(.large)
///       .blendMode(.overlay)
///   }
/// }
/// .frame(width: 150, height: 150)
/// .background(Color.secondary.opacity(0.25))
/// .clipped()
/// ```
public struct NetworkImage<Content>: View where Content: View {
  @Environment(\.networkImageLoader) private var imageLoader
  @StateObject private var model = NetworkImageModel()

  private let source: ImageSource?
  private let transaction: Transaction
  private let content: (NetworkImageState) -> Content

  /// Loads and displays an image from the specified URL using
  /// a default placeholder until the image loads.
  ///
  /// - Parameters:
  ///   - url: The URL of the image to display.
  ///   - scale: The scale to use for the image. The default is `1`.
  public init(url: URL?, scale: CGFloat = 1) where Content == _OptionalContent<Image> {
    self.init(url: url, scale: scale) { state in
      _OptionalContent(state.image)
    }
  }

  /// Loads and displays a modifiable image from the specified URL using a
  /// default placeholder until the image loads.
  ///
  /// - Parameters:
  ///   - url: The URL where the image is located.
  ///   - scale: The scale to use for the image. The default is `1`.
  ///   - transaction: The transaction to use when the state changes.
  ///   - content: A closure that takes the loaded image as an input, and
  ///     returns the view to show. You can return the image directly, or
  ///     modify it as needed before returning it.
  public init<I>(
    url: URL?,
    scale: CGFloat = 1,
    transaction: Transaction = .init(),
    @ViewBuilder content: @escaping (Image) -> I
  ) where Content == _OptionalContent<I>, I: View {
    self.init(url: url, scale: scale, transaction: transaction) { state in
      _OptionalContent(state.image, content: content)
    }
  }

  /// Loads and displays a modifiable image from the specified URL using a
  /// custom placeholder until the image loads.
  ///
  /// - Parameters:
  ///   - url: The URL where the image is located.
  ///   - scale: The scale to use for the image. The default is `1`.
  ///   - transaction: The transaction to use when the state changes.
  ///   - content: A closure that takes the loaded image as an input, and
  ///     returns the view to show. You can return the image directly, or
  ///     modify it as needed before returning it.
  ///   - placeholder: A closure that returns the view to display while the image is loading.
  public init<I, P>(
    url: URL?,
    scale: CGFloat = 1,
    transaction: Transaction = .init(),
    @ViewBuilder content: @escaping (Image) -> I,
    @ViewBuilder placeholder: @escaping () -> P
  ) where Content == _ConditionalContent<I, P>, I: View, P: View {
    self.init(
      url: url,
      scale: scale,
      transaction: transaction,
      content: { state in
        if let image = state.image {
          content(image)
        } else {
          placeholder()
        }
      }
    )
  }

  /// Loads and displays a modifiable image from the specified URL, providing custom views for the different loading states.
  /// - Parameters:
  ///   - url: The URL where the image is located.
  ///   - scale: The scale to use for the image. The default is `1`.
  ///   - transaction: The transaction to use when the state changes.
  ///   - content: A closure that takes the loading state as an input, and
  ///     returns the view to display for the specifed state.
  public init(
    url: URL?,
    scale: CGFloat = 1,
    transaction: Transaction = .init(),
    @ViewBuilder content: @escaping (NetworkImageState) -> Content
  ) {
    self.source = url.map { ImageSource(url: $0, scale: scale) }
    self.transaction = transaction
    self.content = content
  }

  public var body: some View {
    if #available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *) {
      self.content(model.image)
        .task(id: source) {
          model.imageLoaderChanged(imageLoader)
          model.transactionChanged(transaction)
          await model.sourceChanged(source)
        }
    } else {
      self.content(model.image)
        .modifier(
          TaskModifier(id: source) { @MainActor in
            model.imageLoaderChanged(imageLoader)
            model.transactionChanged(transaction)
            await model.sourceChanged(source)
          }
        )
    }
  }
}

public struct _OptionalContent<Content>: View where Content: View {
  private let image: Image?
  private let content: (Image) -> Content

  init(_ image: Image?, content: @escaping (Image) -> Content) {
    self.image = image
    self.content = content
  }

  public var body: some View {
    if let image {
      self.content(image)
    } else {
      Image.empty
        .resizable()
        .redacted(reason: .placeholder)
    }
  }
}

extension _OptionalContent where Content == Image {
  init(_ image: Image?) {
    self.init(image, content: { $0 })
  }
}

extension Image {
  fileprivate static var empty: Image {
    #if canImport(UIKit)
      Image(uiImage: .init())
    #elseif os(macOS)
      Image(nsImage: .init())
    #endif
  }
}
