import Foundation

extension URLSession {
  /// Returns a `URLSession` optimized for image downloading.
  ///
  /// - Parameters:
  ///   - memoryCapacity: The memory capacity of the cache, in bytes.
  ///   - diskCapacity: The disk capacity of the cache, in bytes.
  ///   - timeoutInterval: The timeout interval to use when waiting for data.
  public static func imageLoading(
    memoryCapacity: Int,
    diskCapacity: Int,
    timeoutInterval: TimeInterval
  ) -> URLSession {
    let configuration = URLSessionConfiguration.default

    configuration.requestCachePolicy = .returnCacheDataElseLoad
    configuration.urlCache = URLCache(memoryCapacity: memoryCapacity, diskCapacity: diskCapacity)
    configuration.timeoutIntervalForRequest = timeoutInterval
    configuration.httpAdditionalHeaders = ["Accept": "image/*"]

    return .init(configuration: configuration)
  }
}
