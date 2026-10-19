import XCTest
@testable import SQLite

class SchemaReaderTests: SQLiteTestCase {
    private var schemaReader: SchemaReader!

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()

        schemaReader = db.schema
    }

    func test_columnDefinitions() throws {
        let columns = try schemaReader.columnDefinitions(table: "users")
        XCTAssertEqual(columns, [
            ColumnDefinition(name: "id",
                             primaryKey: .init(autoIncrement: false, onConflict: nil),
                             type: .INTEGER,
                             nullable: true,
                             defaultValue: .NULL,
                             references: nil),
            ColumnDefinition(name: "email",
                             primaryKey: nil,
                             type: .TEXT,
                             nullable: false,
                             defaultValue: .NULL,
                             references: nil),
            ColumnDefinition(name: "age",
                             primaryKey: nil,
                             type: .INTEGER,
                             nullable: true,
                             defaultValue: .NULL,
                             references: nil),
            ColumnDefinition(name: "salary",
                             primaryKey: nil,
                             type: .REAL,
                             nullable: true,
                             defaultValue: .NULL,
                             references: nil),
            ColumnDefinition(name: "admin",
                             primaryKey: nil,
                             type: .NUMERIC,
                             nullable: false,
                             defaultValue: .numericLiteral("0"),
                             references: nil),
            ColumnDefinition(name: "manager_id",
                             primaryKey: nil, type: .INTEGER,
                             nullable: true,
                             defaultValue: .NULL,
                             references: .init(table: "users", column: "manager_id", primaryKey: "id", onUpdate: nil, onDelete: nil)),
            ColumnDefinition(name: "created_at",
                             primaryKey: nil,
                             type: .NUMERIC,
                             nullable: true,
                             defaultValue: .NULL,
                             references: nil)
        ])
    }

    func test_columnDefinitions_parses_conflict_modifier() throws {
        try db.run("CREATE TABLE t (\"id\" INTEGER PRIMARY KEY ON CONFLICT IGNORE AUTOINCREMENT)")

        XCTAssertEqual(
            try schemaReader.columnDefinitions(table: "t"), [
            ColumnDefinition(
                name: "id",
                primaryKey: .init(autoIncrement: true, onConflict: .IGNORE),
                type: .INTEGER,
                nullable: true,
                defaultValue: .NULL,
                references: nil)
            ]
        )
    }

    func test_columnDefinitions_detects_missing_autoincrement() throws {
        try db.run("CREATE TABLE t (\"id\" INTEGER PRIMARY KEY)")

        XCTAssertEqual(
            try schemaReader.columnDefinitions(table: "t"), [
            ColumnDefinition(
                    name: "id",
                    primaryKey: .init(autoIncrement: false),
                    type: .INTEGER,
                    nullable: true,
                    defaultValue: .NULL,
                    references: nil)
            ]
        )
    }

    func test_columnDefinitions_composite_primary_keys() throws {
        try db.run("""
        CREATE TABLE t (
          col1 INTEGER,
          col2 INTEGER,
          col3 INTEGER,
          PRIMARY KEY (col1, col2)
        );
        """)

        XCTAssertEqual(
            try schemaReader.columnDefinitions(table: "t"), [
            ColumnDefinition(
                    name: "col1",
                    primaryKey: .init(autoIncrement: false),
                    type: .INTEGER,
                    nullable: true,
                    defaultValue: .NULL,
                    references: nil),
            ColumnDefinition(
                    name: "col2",
                    primaryKey: .init(autoIncrement: false),
                    type: .INTEGER,
                    nullable: true,
                    defaultValue: .NULL,
                    references: nil),
            ColumnDefinition(
                    name: "col3",
                    primaryKey: nil,
                    type: .INTEGER,
                    nullable: true,
                    defaultValue: .NULL,
                    references: nil)
            ]
        )
    }

    func test_indexDefinitions_no_index() throws {
        let indexes = try schemaReader.indexDefinitions(table: "users")
        XCTAssertTrue(indexes.isEmpty)
    }

    func test_indexDefinitions_with_index() throws {
        try db.run("CREATE UNIQUE INDEX index_users ON users (age DESC) WHERE age IS NOT NULL")
        let indexes = try schemaReader.indexDefinitions(table: "users")

        XCTAssertEqual(indexes, [
            IndexDefinition(
                table: "users",
                name: "index_users",
                unique: true,
                columns: ["age"],
                where: "age IS NOT NULL",
                orders: ["age": .DESC]
            )
        ])
    }

    func test_foreignKeys_info_empty() throws {
        try db.run("CREATE TABLE t (\"id\" INTEGER PRIMARY KEY)")

        let foreignKeys = try schemaReader.foreignKeys(table: "t")
        XCTAssertTrue(foreignKeys.isEmpty)
    }

    func test_foreignKeys() throws {
        let linkTable = Table("test_links")

        let idColumn = SQLite.Expression<Int64>("id")
        let testIdColumn = SQLite.Expression<Int64>("test_id")

        try db.run(linkTable.create(block: { definition in
            definition.column(idColumn, primaryKey: .autoincrement)
            definition.column(testIdColumn, unique: false, check: nil, references: users, Expression<Int64>("id"))
        }))

        let foreignKeys = try schemaReader.foreignKeys(table: "test_links")
        XCTAssertEqual(foreignKeys, [
            .init(table: "users", column: "test_id", primaryKey: "id", onUpdate: nil, onDelete: nil)
        ])
    }

    func test_foreignKeys_references_column() throws {
        let sql = """
          CREATE TABLE artist(
            artistid    INTEGER PRIMARY KEY,
            artistname  TEXT
          );
          CREATE TABLE track(
            trackid     INTEGER,
            trackname   TEXT,
            trackartist INTEGER REFERENCES artist(artistid)
          );
          """
        try db.execute(sql)
        let trackColumns = try db.schema.foreignKeys(table: "track")
        XCTAssertEqual(trackColumns.map { $0.toSQL() }.joined(separator: "\n"), """
            REFERENCES "artist" ("artistid")
            """)
    }

    func test_foreignKeys_references_null_column() throws {
        let sql = """
          CREATE TABLE artist(
            artistid    INTEGER PRIMARY KEY,
            artistname  TEXT
          );
          CREATE TABLE track(
            trackid     INTEGER,
            trackname   TEXT,
            trackartist INTEGER REFERENCES artist
          );
          """
        try db.execute(sql)
        let trackColumns = try db.schema.foreignKeys(table: "track")
        XCTAssertEqual(trackColumns.map { $0.toSQL() }.joined(separator: "\n"), """
            REFERENCES "artist"
            """)
    }

    func test_tableDefinitions() throws {
        let tables = try schemaReader.tableDefinitions()
        XCTAssertEqual(tables.count, 1)
        XCTAssertEqual(tables.first?.name, "users")
    }

    func test_objectDefinitions() throws {
        let tables = try schemaReader.objectDefinitions()

        XCTAssertEqual(tables.map { table in [table.name, table.tableName, table.type.rawValue]}, [
            ["users", "users", "table"],
            ["sqlite_autoindex_users_1", "users", "index"]
        ])
    }

    func test_objectDefinitions_temporary() throws {
        let tables = try schemaReader.objectDefinitions(temp: true)
        XCTAssert(tables.isEmpty)

        try db.run("CREATE TEMPORARY TABLE foo (bar TEXT)")

        let tables2 = try schemaReader.objectDefinitions(temp: true)
        XCTAssertEqual(tables2.map { table in [table.name, table.tableName, table.type.rawValue]}, [
            ["foo", "foo", "table"]
        ])
    }

    func test_objectDefinitions_indexes() throws {
        let emailIndex = users.createIndex(Expression<String>("email"), unique: false, ifNotExists: true)
        try db.run(emailIndex)

        let indexes = try schemaReader.objectDefinitions(type: .index)
            .filter { !$0.isInternal }

        XCTAssertEqual(indexes.map { index in [index.name, index.tableName, index.type.rawValue, index.sql]}, [
            ["index_users_on_email",
             "users",
             "index",
             "CREATE INDEX \"index_users_on_email\" ON \"users\" (\"email\")"]
        ])
    }

    func test_objectDefinitions_triggers() throws {
        let trigger = """
        CREATE TRIGGER test_trigger
            AFTER INSERT ON users BEGIN
                UPDATE USERS SET name = "update" WHERE id = NEW.rowid;
            END;
        """

        try db.run(trigger)

        let triggers = try schemaReader.objectDefinitions(type: .trigger)

        XCTAssertEqual(triggers.map { trigger in [trigger.name, trigger.tableName, trigger.type.rawValue]}, [
            ["test_trigger", "users", "trigger"]
        ])
    }

    func test_objectDefinitionsFilterByType() throws {
        let tables = try schemaReader.objectDefinitions(type: .table)

        XCTAssertEqual(tables.map { table in [table.name, table.tableName, table.type.rawValue]}, [
            ["users", "users", "table"]
        ])
        XCTAssertTrue((try schemaReader.objectDefinitions(type: .trigger)).isEmpty)
    }

    func test_objectDefinitionsFilterByName() throws {
        let tables = try schemaReader.objectDefinitions(name: "users")

        XCTAssertEqual(tables.map { table in [table.name, table.tableName, table.type.rawValue]}, [
            ["users", "users", "table"]
        ])
        XCTAssertTrue((try schemaReader.objectDefinitions(name: "xxx")).isEmpty)
    }
}
