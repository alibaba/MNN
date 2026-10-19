import XCTest
import Foundation
import Dispatch
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

class ConnectionTests: SQLiteTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
    }

    func test_init_withInMemory_returnsInMemoryConnection() throws {
        let db = try Connection(.inMemory)
        XCTAssertEqual("", db.description)
    }

    func test_init_returnsInMemoryByDefault() throws {
        let db = try Connection()
        XCTAssertEqual("", db.description)
    }

    func test_init_withTemporary_returnsTemporaryConnection() throws {
        let db = try Connection(.temporary)
        XCTAssertEqual("", db.description)
    }

    func test_init_withURI_returnsURIConnection() throws {
        let db = try Connection(.uri("\(NSTemporaryDirectory())/SQLite.swift Tests.sqlite3"))
        let url = URL(fileURLWithPath: db.description)
        XCTAssertEqual(url.lastPathComponent, "SQLite.swift Tests.sqlite3")
    }

    func test_init_withString_returnsURIConnection() throws {
        let db = try Connection("\(NSTemporaryDirectory())/SQLite.swift Tests.sqlite3")
        let url = URL(fileURLWithPath: db.description)
        XCTAssertEqual(url.lastPathComponent, "SQLite.swift Tests.sqlite3")
    }

    func test_init_with_Uri_and_Parameters() throws {
        let testDb = fixture("test", withExtension: "sqlite")
        _ = try Connection(.uri(testDb, parameters: [.cache(.shared)]))
    }

    func test_location_without_Uri_parameters() {
        let location: Connection.Location = .uri("foo")
        XCTAssertEqual(location.description, "foo")
    }

    func test_location_with_Uri_parameters() {
        let location: Connection.Location = .uri("foo", parameters: [.mode(.readOnly), .cache(.private)])
        XCTAssertEqual(location.description, "file:foo?mode=ro&cache=private")
    }

    func test_readonly_returnsFalseOnReadWriteConnections() {
        XCTAssertFalse(db.readonly)
    }

    func test_readonly_returnsTrueOnReadOnlyConnections() throws {
        let db = try Connection(readonly: true)
        XCTAssertTrue(db.readonly)
    }

    func test_changes_returnsZeroOnNewConnections() {
        XCTAssertEqual(0, db.changes)
    }

    func test_lastInsertRowid_returnsLastIdAfterInserts() throws {
        try insertUser("alice")
        XCTAssertEqual(1, db.lastInsertRowid)
    }

    func test_lastInsertRowid_doesNotResetAfterError() throws {
        XCTAssert(db.lastInsertRowid == 0)
        try insertUser("alice")
        XCTAssertEqual(1, db.lastInsertRowid)
        XCTAssertThrowsError(
            try db.run("INSERT INTO \"users\" (email, age, admin) values ('invalid@example.com', 12, 'invalid')")
        ) { error in
            if case SQLite.Result.error(_, let code, _) = error {
                XCTAssertEqual(SQLITE_CONSTRAINT, code)
            } else {
                XCTFail("expected error")
            }
        }
        XCTAssertEqual(1, db.lastInsertRowid)
    }

    func test_changes_returnsNumberOfChanges() throws {
        try insertUser("alice")
        XCTAssertEqual(1, db.changes)
        try insertUser("betsy")
        XCTAssertEqual(1, db.changes)
    }

    func test_totalChanges_returnsTotalNumberOfChanges() throws {
        XCTAssertEqual(0, db.totalChanges)
        try insertUser("alice")
        XCTAssertEqual(1, db.totalChanges)
        try insertUser("betsy")
        XCTAssertEqual(2, db.totalChanges)
    }

    func test_useExtendedErrorCodes_returnsFalseDefault() throws {
        XCTAssertFalse(db.usesExtendedErrorCodes)
    }

    func test_prepare_preparesAndReturnsStatements() throws {
        _ = try db.prepare("SELECT * FROM users WHERE admin = 0")
        _ = try db.prepare("SELECT * FROM users WHERE admin = ?", 0)
        _ = try db.prepare("SELECT * FROM users WHERE admin = ?", [0])
        _ = try db.prepare("SELECT * FROM users WHERE admin = $admin", ["$admin": 0])
    }

    func test_run_preparesRunsAndReturnsStatements() throws {
        try db.run("SELECT * FROM users WHERE admin = 0")
        try db.run("SELECT * FROM users WHERE admin = ?", 0)
        try db.run("SELECT * FROM users WHERE admin = ?", [0])
        try db.run("SELECT * FROM users WHERE admin = $admin", ["$admin": 0])
        assertSQL("SELECT * FROM users WHERE admin = 0", 4)
    }

    func test_vacuum() throws {
        try db.vacuum()
    }

    func test_scalar_preparesRunsAndReturnsScalarValues() throws {
        XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users WHERE admin = 0") as? Int64)
        XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users WHERE admin = ?", 0) as? Int64)
        XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users WHERE admin = ?", [0]) as? Int64)
        XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users WHERE admin = $admin", ["$admin": 0]) as? Int64)
        assertSQL("SELECT count(*) FROM users WHERE admin = 0", 4)
    }

    func test_execute_comment() throws {
        try db.run("-- this is a comment\nSELECT 1")
        assertSQL("-- this is a comment", 0)
        assertSQL("SELECT 1", 0)
    }

    func test_transaction_executesBeginDeferred() throws {
        try db.transaction(.deferred) {}

        assertSQL("BEGIN DEFERRED TRANSACTION")
    }

    func test_transaction_executesBeginImmediate() throws {
        try db.transaction(.immediate) {}

        assertSQL("BEGIN IMMEDIATE TRANSACTION")
    }

    func test_transaction_executesBeginExclusive() throws {
        try db.transaction(.exclusive) {}

        assertSQL("BEGIN EXCLUSIVE TRANSACTION")
    }

    func test_backup_copiesDatabase() throws {
        let target = try Connection()

        try insertUsers("alice", "betsy")

        let backup = try db.backup(usingConnection: target)
        try backup.step()

        let users = try target.prepare("SELECT email FROM users ORDER BY email")
        XCTAssertEqual(users.map { $0[0] as? String }, ["alice@example.com", "betsy@example.com"])
    }

    func test_transaction_beginsAndCommitsTransactions() throws {
        let stmt = try db.prepare("INSERT INTO users (email) VALUES (?)", "alice@example.com")

        try db.transaction {
            try stmt.run()
        }

        assertSQL("BEGIN DEFERRED TRANSACTION")
        assertSQL("INSERT INTO users (email) VALUES ('alice@example.com')")
        assertSQL("COMMIT TRANSACTION")
        assertSQL("ROLLBACK TRANSACTION", 0)
    }

    func test_transaction_rollsBackTransactionsIfCommitsFail() throws {
        let sqliteVersion = String(describing: try db.scalar("SELECT sqlite_version()")!)
            .split(separator: ".").compactMap { Int($0) }
        // PRAGMA defer_foreign_keys only supported in SQLite >= 3.8.0
        guard sqliteVersion[0] == 3 && sqliteVersion[1] >= 8 else {
            NSLog("skipping test for SQLite version \(sqliteVersion)")
            return
        }
        // This test case needs to emulate an environment where the individual statements succeed, but committing the
        // transaction fails. Using deferred foreign keys is one option to achieve this.
        try db.execute("PRAGMA foreign_keys = ON;")
        try db.execute("PRAGMA defer_foreign_keys = ON;")
        let stmt = try db.prepare("INSERT INTO users (email, manager_id) VALUES (?, ?)", "alice@example.com", 100)

        do {
            try db.transaction {
                try stmt.run()
            }
            XCTFail("expected error")
        } catch let Result.error(_, code, _) {
            XCTAssertEqual(SQLITE_CONSTRAINT, code)
        } catch let error {
            XCTFail("unexpected error: \(error)")
        }

        assertSQL("BEGIN DEFERRED TRANSACTION")
        assertSQL("INSERT INTO users (email, manager_id) VALUES ('alice@example.com', 100)")
        assertSQL("COMMIT TRANSACTION")
        assertSQL("ROLLBACK TRANSACTION")

        // Run another transaction to ensure that a subsequent transaction does not fail with an "cannot start a
        // transaction within a transaction" error.
        let stmt2 = try db.prepare("INSERT INTO users (email) VALUES (?)", "alice@example.com")
        try db.transaction {
            try stmt2.run()
        }
    }

    func test_transaction_beginsAndRollsTransactionsBack() throws {
        let stmt = try db.prepare("INSERT INTO users (email) VALUES (?)", "alice@example.com")

        do {
            try db.transaction {
                try stmt.run()
                try stmt.run()
            }
        } catch {
        }

        assertSQL("BEGIN DEFERRED TRANSACTION")
        assertSQL("INSERT INTO users (email) VALUES ('alice@example.com')", 2)
        assertSQL("ROLLBACK TRANSACTION")
        assertSQL("COMMIT TRANSACTION", 0)
    }

    func test_savepoint_beginsAndCommitsSavepoints() throws {
        try db.savepoint("1") {
            try db.savepoint("2") {
                try db.run("INSERT INTO users (email) VALUES (?)", "alice@example.com")
            }
        }

        assertSQL("SAVEPOINT '1'")
        assertSQL("SAVEPOINT '2'")
        assertSQL("INSERT INTO users (email) VALUES ('alice@example.com')")
        assertSQL("RELEASE SAVEPOINT '2'")
        assertSQL("RELEASE SAVEPOINT '1'")
        assertSQL("ROLLBACK TO SAVEPOINT '2'", 0)
        assertSQL("ROLLBACK TO SAVEPOINT '1'", 0)
    }

    func test_savepoint_beginsAndRollsSavepointsBack() throws {
        let stmt = try db.prepare("INSERT INTO users (email) VALUES (?)", "alice@example.com")

        do {
            try db.savepoint("1") {
                try db.savepoint("2") {
                    try stmt.run()
                    try stmt.run()
                    try stmt.run()
                }
                try db.savepoint("2") {
                    try stmt.run()
                    try stmt.run()
                    try stmt.run()
                }
            }
        } catch {
        }

        assertSQL("SAVEPOINT '1'")
        assertSQL("SAVEPOINT '2'")
        assertSQL("INSERT INTO users (email) VALUES ('alice@example.com')", 2)
        assertSQL("ROLLBACK TO SAVEPOINT '2'")
        assertSQL("ROLLBACK TO SAVEPOINT '1'")
        assertSQL("RELEASE SAVEPOINT '2'", 0)
        assertSQL("RELEASE SAVEPOINT '1'", 0)
    }

    func test_updateHook_setsUpdateHook_withInsert() throws {
        try async { done in
            db.updateHook { operation, db, table, rowid in
                XCTAssertEqual(Connection.Operation.insert, operation)
                XCTAssertEqual("main", db)
                XCTAssertEqual("users", table)
                XCTAssertEqual(1, rowid)
                done()
            }
            try insertUser("alice")
        }
    }

    func test_updateHook_setsUpdateHook_withUpdate() throws {
        try insertUser("alice")
        try async { done in
            db.updateHook { operation, db, table, rowid in
                XCTAssertEqual(Connection.Operation.update, operation)
                XCTAssertEqual("main", db)
                XCTAssertEqual("users", table)
                XCTAssertEqual(1, rowid)
                done()
            }
            try db.run("UPDATE users SET email = 'alice@example.com'")
        }
    }

    func test_updateHook_setsUpdateHook_withDelete() throws {
        try insertUser("alice")
        try async { done in
            db.updateHook { operation, db, table, rowid in
                XCTAssertEqual(Connection.Operation.delete, operation)
                XCTAssertEqual("main", db)
                XCTAssertEqual("users", table)
                XCTAssertEqual(1, rowid)
                done()
            }
            try db.run("DELETE FROM users WHERE id = 1")
        }
    }

    func test_commitHook_setsCommitHook() throws {
        try async { done in
            db.commitHook {
                done()
            }
            try db.transaction {
                try insertUser("alice")
            }
            XCTAssertEqual(1, try db.scalar("SELECT count(*) FROM users") as? Int64)
        }
    }

    func test_rollbackHook_setsRollbackHook() throws {
        try async { done in
            db.rollbackHook(done)
            do {
                try db.transaction {
                    try insertUser("alice")
                    try insertUser("alice") // throw
                }
            } catch {
            }
            XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users") as? Int64)
        }
    }

    func test_commitHook_withRollback_rollsBack() throws {
        try async { done in
            db.commitHook {
                throw NSError(domain: "com.stephencelis.SQLiteTests", code: 1, userInfo: nil)
            }
            db.rollbackHook(done)
            do {
                try db.transaction {
                    try insertUser("alice")
                }
            } catch {
            }
            XCTAssertEqual(0, try db.scalar("SELECT count(*) FROM users") as? Int64)
        }
    }

    // https://github.com/stephencelis/SQLite.swift/issues/1071
    #if !os(Linux)
    func test_createFunction_withArrayArguments() throws {
        db.createFunction("hello") { $0[0].map { "Hello, \($0)!" } }

        XCTAssertEqual("Hello, world!", try db.scalar("SELECT hello('world')") as? String)
        XCTAssert(try db.scalar("SELECT hello(NULL)") == nil)
    }

    func test_createFunction_createsQuotableFunction() throws {
        db.createFunction("hello world") { $0[0].map { "Hello, \($0)!" } }

        XCTAssertEqual("Hello, world!", try db.scalar("SELECT \"hello world\"('world')") as? String)
        XCTAssert(try db.scalar("SELECT \"hello world\"(NULL)") == nil)
    }

    func test_createCollation_createsCollation() throws {
        try db.createCollation("NODIACRITIC") { lhs, rhs in
            lhs.compare(rhs, options: .diacriticInsensitive)
        }
        XCTAssertEqual(1, try db.scalar("SELECT ? = ? COLLATE NODIACRITIC", "cafe", "café") as? Int64)
    }

    func test_createCollation_createsQuotableCollation() throws {
        try db.createCollation("NO DIACRITIC") { lhs, rhs in
            lhs.compare(rhs, options: .diacriticInsensitive)
        }
        XCTAssertEqual(1, try db.scalar("SELECT ? = ? COLLATE \"NO DIACRITIC\"", "cafe", "café") as? Int64)
    }

    func XXX_test_interrupt_interruptsLongRunningQuery() throws {
        let semaphore = DispatchSemaphore(value: 0)
        db.createFunction("sleep") { _ in
            DispatchQueue.global(qos: .background).async {
                self.db.interrupt()
                semaphore.signal()
            }
            semaphore.wait()
            return nil
        }
        let stmt = try db.prepare("SELECT sleep()")
        XCTAssertThrowsError(try stmt.run()) { error in
            if case Result.error(_, let code, _) = error {
                XCTAssertEqual(code, SQLITE_INTERRUPT)
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }
    #endif

    func test_concurrent_access_single_connection() throws {
        // test can fail on iOS/tvOS 9.x: SQLite compile-time differences?
        guard #available(iOS 10.0, OSX 10.10, tvOS 10.0, watchOS 2.2, *) else { return }

        let conn = try Connection("\(NSTemporaryDirectory())/\(UUID().uuidString)")
        try conn.execute("DROP TABLE IF EXISTS test; CREATE TABLE test(value);")
        try conn.run("INSERT INTO test(value) VALUES(?)", 0)
        let queue = DispatchQueue(label: "Readers", attributes: [.concurrent])

        let nReaders = 5
        let semaphores =  Array(repeating: DispatchSemaphore(value: 100), count: nReaders)
        for index in 0..<nReaders {
            queue.async {
                while semaphores[index].signal() == 0 {
                    _ = try! conn.scalar("SELECT value FROM test")
                }
            }
        }
        semaphores.forEach { $0.wait() }
    }
}
