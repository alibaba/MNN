import XCTest
@testable import SQLite

#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

class StatementTests: SQLiteTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
    }

    func test_cursor_to_blob() throws {
        try insertUsers("alice")
        let statement = try db.prepare("SELECT email FROM users")
        XCTAssert(try statement.step())
        let blob = statement.row[0] as Blob
        XCTAssertEqual("alice@example.com", String(bytes: blob.bytes, encoding: .utf8)!)
    }

    func test_zero_sized_blob_returns_null() throws {
        let blobs = Table("blobs")
        let blobColumn = Expression<Blob>("blob_column")
        try db.run(blobs.create { $0.column(blobColumn) })
        try db.run(blobs.insert(blobColumn <- Blob(bytes: [])))
        let blobValue = try db.scalar(blobs.select(blobColumn).limit(1, offset: 0))
        XCTAssertEqual([], blobValue.bytes)
    }

    func test_prepareRowIterator() throws {
        let names = ["a", "b", "c"]
        try insertUsers(names)

        let emailColumn = Expression<String>("email")
        let statement = try db.prepare("SELECT email FROM users")
        let emails = try statement.prepareRowIterator().map { $0[emailColumn] }

        XCTAssertEqual(names.map({ "\($0)@example.com" }), emails.sorted())
    }

    /// Check that a statement reset will close the implicit transaction, allowing wal file to checkpoint
    func test_reset_statement() throws {
        // insert single row
        try insertUsers("bob")

        // prepare a statement and read a single row. This will increment the cursor which
        // prevents the implicit transaction from closing.
        // https://www.sqlite.org/lang_transaction.html#implicit_versus_explicit_transactions
        let statement = try db.prepare("SELECT email FROM users")
        _ = try statement.step()

        // verify implicit transaction is not closed, and the users table is still locked
        XCTAssertThrowsError(try db.run("DROP TABLE users")) { error in
            if case let Result.error(_, code, _) = error {
                XCTAssertEqual(code, SQLITE_LOCKED)
            } else {
                XCTFail("unexpected error")
            }
        }

        // reset the prepared statement, unlocking the table and allowing the implicit transaction to close
        statement.reset()

        // truncate succeeds
        try db.run("DROP TABLE users")
    }
}
