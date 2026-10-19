import XCTest
@testable import SQLite

class SelectTests: SQLiteTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
        try createUsersDataTable()
    }

    func createUsersDataTable() throws {
        try db.execute("""
            CREATE TABLE users_name (
                id INTEGER,
                user_id INTEGER REFERENCES users(id),
                name TEXT
            )
            """
        )
    }

    func test_select_columns_from_multiple_tables() throws {
        let usersData = Table("users_name")
        let users = Table("users")

        let name = Expression<String>("name")
        let id = Expression<Int64>("id")
        let userID = Expression<Int64>("user_id")
        let email = Expression<String>("email")

        try insertUser("Joey")
        try db.run(usersData.insert(
            id <- 1,
            userID <- 1,
            name <- "Joey"
        ))

        try db.prepare(users.select(name, email).join(usersData, on: userID == users[id])).forEach {
            XCTAssertEqual($0[name], "Joey")
            XCTAssertEqual($0[email], "Joey@example.com")
        }
    }
}
