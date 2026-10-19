import XCTest
@testable import SQLite

class SchemaChangerTests: SQLiteTestCase {
    var schemaChanger: SchemaChanger!
    var schema: SchemaReader!

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()

        try insertUsers("bob")

        schema = SchemaReader(connection: db)
        schemaChanger = SchemaChanger(connection: db)
    }

    func test_empty_migration_does_not_change_column_definitions() throws {
        let previous = try schema.columnDefinitions(table: "users")
        try schemaChanger.alter(table: "users") { _ in
        }
        let current = try schema.columnDefinitions(table: "users")

        XCTAssertEqual(previous, current)
    }

    func test_empty_migration_does_not_change_index_definitions() throws {
        let previous = try schema.indexDefinitions(table: "users")
        try schemaChanger.alter(table: "users") { _ in
        }
        let current = try schema.indexDefinitions(table: "users")

        XCTAssertEqual(previous, current)
    }

    func test_empty_migration_does_not_change_foreign_key_definitions() throws {
        let previous = try schema.foreignKeys(table: "users")
        try schemaChanger.alter(table: "users") { _ in
        }
        let current = try schema.foreignKeys(table: "users")

        XCTAssertEqual(previous, current)
    }

    func test_empty_migration_does_not_change_the_row_count() throws {
        let previous = try db.scalar(users.count)
        try schemaChanger.alter(table: "users") { _ in
        }
        let current = try db.scalar(users.count)

        XCTAssertEqual(previous, current)
    }

    func test_drop_column() throws {
        try schemaChanger.alter(table: "users") { table in
            table.drop(column: "age")
        }
        let columns = try schema.columnDefinitions(table: "users").map(\.name)
        XCTAssertFalse(columns.contains("age"))
    }

    func test_drop_column_legacy() throws {
        schemaChanger = .init(connection: db, version: .init(major: 3, minor: 24)) // DROP COLUMN introduced in 3.35.0

        try schemaChanger.alter(table: "users") { table in
            table.drop(column: "age")
        }
        let columns = try schema.columnDefinitions(table: "users").map(\.name)
        XCTAssertFalse(columns.contains("age"))
    }

    func test_rename_column() throws {
        try schemaChanger.alter(table: "users") { table in
            table.rename(column: "age", to: "age2")
        }

        let columns = try schema.columnDefinitions(table: "users").map(\.name)
        XCTAssertFalse(columns.contains("age"))
        XCTAssertTrue(columns.contains("age2"))
    }

    func test_rename_column_legacy() throws {
        schemaChanger = .init(connection: db, version: .init(major: 3, minor: 24)) // RENAME COLUMN introduced in 3.25.0

        try schemaChanger.alter(table: "users") { table in
            table.rename(column: "age", to: "age2")
        }

        let columns = try schema.columnDefinitions(table: "users").map(\.name)
        XCTAssertFalse(columns.contains("age"))
        XCTAssertTrue(columns.contains("age2"))
    }

    func test_add_column() throws {
        let column = Expression<String>("new_column")
        let newColumn = ColumnDefinition(name: "new_column",
                                         type: .TEXT,
                                         nullable: true,
                                         defaultValue: .stringLiteral("foo"))

        try schemaChanger.alter(table: "users") { table in
            table.add(column: newColumn)
        }

        let columns = try schema.columnDefinitions(table: "users")
        XCTAssertTrue(columns.contains(newColumn))

        XCTAssertEqual(try db.pluck(users.select(column))?[column], "foo")
    }

    func test_add_column_primary_key_fails() throws {
        let newColumn = ColumnDefinition(name: "new_column",
                                         primaryKey: .init(autoIncrement: false, onConflict: nil),
                                         type: .TEXT)

        XCTAssertThrowsError(try schemaChanger.alter(table: "users") { table in
            table.add(column: newColumn)
        }) { error in
            if case SchemaChanger.Error.invalidColumnDefinition(_) = error {
                XCTAssertEqual("Invalid column definition: can not add primary key column", error.localizedDescription)
            } else {
                XCTFail("invalid error: \(error)")
            }
        }
    }

    func test_drop_table() throws {
        try schemaChanger.drop(table: "users")
        XCTAssertThrowsError(try db.scalar(users.count)) { error in
            if case Result.error(let message, _, _) =  error {
                XCTAssertEqual(message, "no such table: users")
            } else {
                XCTFail("unexpected error \(error)")
            }
        }
    }

    func test_drop_table_if_exists_true() throws {
        try schemaChanger.drop(table: "xxx", ifExists: true)
    }

    func test_drop_table_if_exists_false() throws {
        XCTAssertThrowsError(try schemaChanger.drop(table: "xxx", ifExists: false)) { error in
            if case Result.error(let message, _, _) =  error {
                XCTAssertEqual(message, "no such table: xxx")
            } else {
                XCTFail("unexpected error \(error)")
            }
        }
    }

    func test_rename_table() throws {
        try schemaChanger.rename(table: "users", to: "users_new")
        let users_new = Table("users_new")
        XCTAssertEqual((try db.scalar(users_new.count)) as Int, 1)
    }
}
