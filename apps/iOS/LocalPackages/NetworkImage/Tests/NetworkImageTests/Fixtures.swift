import CoreGraphics
import Foundation
import ImageIO

enum Fixtures {
  static let url = URL(string: "https://picsum.photos/id/237/300/200")!
  static let anotherURL = URL(string: "https://picsum.photos/id/1/200/300")!
  static let imageData = Data(
    base64Encoded: """
      iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42\
      mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=
      """
  )!
  static let image = CGImageSourceCreateImageAtIndex(
    CGImageSourceCreateWithData(imageData as CFData, nil)!, 0, nil
  )!
  static let okResponse = HTTPURLResponse(
    url: url,
    statusCode: 200,
    httpVersion: "HTTP/1.1",
    headerFields: nil
  )!
  static let internalServerErrorResponse = HTTPURLResponse(
    url: url,
    statusCode: 500,
    httpVersion: "HTTP/1.1",
    headerFields: nil
  )!
}
