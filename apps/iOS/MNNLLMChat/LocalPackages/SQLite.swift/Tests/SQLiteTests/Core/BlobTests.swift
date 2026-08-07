import XCTest
import SQLite

class BlobTests: XCTestCase {

    func test_toHex() {
        let blob = Blob(bytes: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250, 255])
        XCTAssertEqual(blob.toHex(), "000a141e28323c46505a6496faff")
    }

    func test_toHex_empty() {
        let blob = Blob(bytes: [])
        XCTAssertEqual(blob.toHex(), "")
    }

    func test_description() {
        let blob = Blob(bytes: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250, 255])
        XCTAssertEqual(blob.description, "x'000a141e28323c46505a6496faff'")
    }

    func test_description_empty() {
        let blob = Blob(bytes: [])
        XCTAssertEqual(blob.description, "x''")
    }

    func test_init_array() {
        let blob = Blob(bytes: [42, 43, 44])
        XCTAssertEqual(blob.bytes, [42, 43, 44])
    }

    func test_init_unsafeRawPointer() {
        let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: 3)
        pointer.initialize(repeating: 42, count: 3)
        let blob = Blob(bytes: pointer, length: 3)
        XCTAssertEqual(blob.bytes, [42, 42, 42])
    }

    func test_equality() {
        let blob1 = Blob(bytes: [42, 42, 42])
        let blob2 = Blob(bytes: [42, 42, 42])
        let blob3 = Blob(bytes: [42, 42, 43])

        XCTAssertEqual(Blob(bytes: []), Blob(bytes: []))
        XCTAssertEqual(blob1, blob2)
        XCTAssertNotEqual(blob1, blob3)
    }
}
