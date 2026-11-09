import XCTest
import Foundation
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

class ConnectionPragmaTests: SQLiteTestCase {
    func test_userVersion() {
        db.userVersion = 2
        XCTAssertEqual(2, db.userVersion!)
    }

    func test_sqlite_version() {
        XCTAssertTrue(db.sqliteVersion >= .init(major: 3, minor: 0))
    }

    func test_foreignKeys_defaults_to_false() {
        XCTAssertFalse(db.foreignKeys)
    }

    func test_foreignKeys_sets_value() {
        db.foreignKeys = true
        XCTAssertTrue(db.foreignKeys)
    }

    func test_defer_foreignKeys_defaults_to_false() {
        XCTAssertFalse(db.deferForeignKeys)
    }

    func test_defer_foreignKeys_sets_value() {
        db.deferForeignKeys = true
        XCTAssertTrue(db.deferForeignKeys)
    }
}
