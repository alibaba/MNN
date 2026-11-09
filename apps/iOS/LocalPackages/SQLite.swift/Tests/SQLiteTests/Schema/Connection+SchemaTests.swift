import XCTest
@testable import SQLite

class ConnectionSchemaTests: SQLiteTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
    }

    func test_foreignKeyCheck() throws {
        let errors = try db.foreignKeyCheck()
        XCTAssert(errors.isEmpty)
    }

    func test_foreignKeyCheck_with_table() throws {
        let errors = try db.foreignKeyCheck(table: "users")
        XCTAssert(errors.isEmpty)
    }

    func test_foreignKeyCheck_table_not_found() throws {
        XCTAssertThrowsError(try db.foreignKeyCheck(table: "xxx")) { error in
            guard case Result.error(let message, _, _) = error else {
                assertionFailure("invalid error type")
                return
            }
            XCTAssertEqual(message, "no such table: xxx")
        }
    }

    func test_integrityCheck_global() throws {
        let results = try db.integrityCheck()
        XCTAssert(results.isEmpty)
    }

    func test_partial_integrityCheck_table() throws {
        guard db.supports(.partialIntegrityCheck) else { return }
        let results = try db.integrityCheck(table: "users")
        XCTAssert(results.isEmpty)
    }

    func test_integrityCheck_table_not_found() throws {
        guard db.supports(.partialIntegrityCheck) else { return }
        XCTAssertThrowsError(try db.integrityCheck(table: "xxx")) { error in
            guard case Result.error(let message, _, _) = error else {
                assertionFailure("invalid error type")
                return
            }
            XCTAssertEqual(message, "no such table: xxx")
        }
    }
}
