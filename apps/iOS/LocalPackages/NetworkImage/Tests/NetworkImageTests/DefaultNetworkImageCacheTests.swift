import NetworkImage
import XCTest

final class DefaultNetworkImageCacheTests: XCTestCase {
  func testCacheMiss() {
    // given
    let cache = DefaultNetworkImageCache()

    // when
    let image = cache.image(for: Fixtures.url)

    // then
    XCTAssertNil(image)
  }

  func testCacheHit() {
    // given
    let cache = DefaultNetworkImageCache()
    cache.setImage(Fixtures.image, for: Fixtures.url)

    // when
    let image = cache.image(for: Fixtures.url)

    // then
    XCTAssertIdentical(image, Fixtures.image)
  }
}
