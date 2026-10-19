import CoreGraphics
import Foundation

/// A type that temporarily stores images in memory, keyed by the URL from which they were loaded.
public protocol NetworkImageCache: AnyObject {
  /// Returns the image associated with a given URL.
  func image(for url: URL) -> CGImage?

  /// Sets the image for the specified URL in the cache.
  func setImage(_ image: CGImage, for url: URL)
}

// MARK: - DefaultNetworkImageCache

/// The default network image cache.
public class DefaultNetworkImageCache {
  private enum Constants {
    static let defaultCountLimit = 100
  }

  private let cache = NSCache<NSURL, CGImage>()

  /// Creates a default network image cache.
  /// - Parameter countLimit: The maximum number of images that the cache should hold. If `0`,
  ///                         there is no count limit. The default value is `0`.
  public init(countLimit: Int = 0) {
    self.cache.countLimit = countLimit
  }

  /// A shared network image cache.
  public static let shared = DefaultNetworkImageCache(countLimit: Constants.defaultCountLimit)
}

extension DefaultNetworkImageCache: NetworkImageCache, @unchecked Sendable {
  public func image(for url: URL) -> CGImage? {
    self.cache.object(forKey: url as NSURL)
  }

  public func setImage(_ image: CGImage, for url: URL) {
    self.cache.setObject(image, forKey: url as NSURL)
  }
}

extension NetworkImageCache where Self == DefaultNetworkImageCache {
  /// The shared default network image cache.
  static var `default`: Self {
    .shared
  }
}
