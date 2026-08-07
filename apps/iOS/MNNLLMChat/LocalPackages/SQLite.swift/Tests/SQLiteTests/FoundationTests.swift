import XCTest
import SQLite

class FoundationTests: XCTestCase {
    func testDataFromBlob() {
        let data = Data([1, 2, 3])
        let blob = data.datatypeValue
        XCTAssertEqual([1, 2, 3], blob.bytes)
    }

    func testBlobToData() {
        let blob = Blob(bytes: [1, 2, 3])
        let data = Data.fromDatatypeValue(blob)
        XCTAssertEqual(Data([1, 2, 3]), data)
    }

    func testStringFromUUID() {
        let uuid = UUID(uuidString: "4ABE10C9-FF12-4CD4-90C1-4B429001BAD3")!
        let string = uuid.datatypeValue
        XCTAssertEqual("4ABE10C9-FF12-4CD4-90C1-4B429001BAD3", string)
    }

    func testUUIDFromString() {
        let string = "4ABE10C9-FF12-4CD4-90C1-4B429001BAD3"
        let uuid = UUID.fromDatatypeValue(string)
        XCTAssertEqual(UUID(uuidString: "4ABE10C9-FF12-4CD4-90C1-4B429001BAD3"), uuid)
    }

    func testURLFromString() {
        let string = "http://foo.com"
        let url = URL.fromDatatypeValue(string)
        XCTAssertEqual(URL(string: string), url)
    }

    func testStringFromURL() {
        let url = URL(string: "http://foo.com")!
        let string = url.datatypeValue
        XCTAssertEqual("http://foo.com", string)
    }
}
