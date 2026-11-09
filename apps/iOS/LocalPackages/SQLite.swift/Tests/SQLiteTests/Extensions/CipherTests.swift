#if SQLITE_SWIFT_SQLCIPHER
import XCTest
import SQLite
import SQLCipher

class CipherTests: XCTestCase {
    var db1: Connection!
    var db2: Connection!

    override func setUpWithError() throws {
        db1 = try Connection()
        db2 = try Connection()
        // db1

        try db1.key("hello")

        try db1.run("CREATE TABLE foo (bar TEXT)")
        try db1.run("INSERT INTO foo (bar) VALUES ('world')")

        // db2
        let key2 = keyData()
        try db2.key(Blob(bytes: key2.bytes, length: key2.length))

        try db2.run("CREATE TABLE foo (bar TEXT)")
        try db2.run("INSERT INTO foo (bar) VALUES ('world')")

        try super.setUpWithError()
    }

    func test_key() throws {
        XCTAssertEqual(1, try db1.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    func test_key_blob_literal() throws {
        let db = try Connection()
        try db.key("x'2DD29CA851E7B56E4697B0E1F08507293D761A05CE4D1B628663F411A8086D99'")
    }

    func test_rekey() throws {
        try db1.rekey("goodbye")
        XCTAssertEqual(1, try db1.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    func test_data_key() throws {
        XCTAssertEqual(1, try db2.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    func test_data_rekey() throws {
        let newKey = keyData()
        try db2.rekey(Blob(bytes: newKey.bytes, length: newKey.length))
        XCTAssertEqual(1, try db2.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    func test_keyFailure() throws {
        let path = "\(NSTemporaryDirectory())/db.sqlite3"
        _ = try? FileManager.default.removeItem(atPath: path)

        let connA = try Connection(path)
        defer { try? FileManager.default.removeItem(atPath: path) }

        try connA.key("hello")
        try connA.run("CREATE TABLE foo (bar TEXT)")

        let connB = try Connection(path, readonly: true)

        do {
            try connB.key("world")
            XCTFail("expected exception")
        } catch Result.error(_, let code, _) {
            XCTAssertEqual(SQLITE_NOTADB, code)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func test_open_db_encrypted_with_sqlcipher() throws {
        // $ sqlcipher Tests/SQLiteTests/fixtures/encrypted-[version].x.sqlite
        // sqlite> pragma key = 'sqlcipher-test';
        // sqlite> CREATE TABLE foo (bar TEXT);
        // sqlite> INSERT INTO foo (bar) VALUES ('world');
        guard let cipherVersion: String = db1.cipherVersion,
            cipherVersion.starts(with: "3.") || cipherVersion.starts(with: "4.") else { return }

        let encryptedFile = cipherVersion.starts(with: "3.") ?
                fixture("encrypted-3.x", withExtension: "sqlite") :
                fixture("encrypted-4.x", withExtension: "sqlite")

        try FileManager.default.setAttributes([FileAttributeKey.immutable: 1], ofItemAtPath: encryptedFile)
        XCTAssertFalse(FileManager.default.isWritableFile(atPath: encryptedFile))

        defer {
            // ensure file can be cleaned up afterwards
            try? FileManager.default.setAttributes([FileAttributeKey.immutable: 0], ofItemAtPath: encryptedFile)
        }

        let conn = try Connection(encryptedFile)
        try conn.key("sqlcipher-test")
        XCTAssertEqual(1, try conn.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    func test_export() throws {
        let tmp = temporaryFile()
        try db1.sqlcipher_export(.uri(tmp), key: "mykey")

        let conn = try Connection(tmp)
        try conn.key("mykey")
        XCTAssertEqual(1, try conn.scalar("SELECT count(*) FROM foo") as? Int64)
    }

    private func keyData(length: Int = 64) -> NSData {
        let keyData = NSMutableData(length: length)!
        let result  = SecRandomCopyBytes(kSecRandomDefault, length,
                                         keyData.mutableBytes.assumingMemoryBound(to: UInt8.self))
        XCTAssertEqual(0, result)
        return NSData(data: keyData)
    }
}
#endif
