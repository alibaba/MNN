import XCTest
@testable import SQLite

class ColumnDefinitionTests: XCTestCase {
    var definition: ColumnDefinition!
    var expected: String!

    static let definitions: [(String, ColumnDefinition)] = [
        ("\"id\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL",
        ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)),

        ("\"other_id\" INTEGER NOT NULL REFERENCES \"other_table\" (\"some_id\")",
        ColumnDefinition(name: "other_id", primaryKey: nil, type: .INTEGER, nullable: false, defaultValue: .NULL,
                         references: .init(table: "other_table", column: "", primaryKey: "some_id", onUpdate: nil, onDelete: nil))),

        ("\"text\" TEXT",
        ColumnDefinition(name: "text", primaryKey: nil, type: .TEXT, nullable: true, defaultValue: .NULL, references: nil)),

        ("\"text\" TEXT NOT NULL",
        ColumnDefinition(name: "text", primaryKey: nil, type: .TEXT, nullable: false, defaultValue: .NULL, references: nil)),

        ("\"text_column\" TEXT DEFAULT 'fo\"o'",
        ColumnDefinition(name: "text_column", primaryKey: nil, type: .TEXT, nullable: true,
        defaultValue: .stringLiteral("fo\"o"), references: nil)),

        ("\"integer_column\" INTEGER DEFAULT 123",
        ColumnDefinition(name: "integer_column", primaryKey: nil, type: .INTEGER, nullable: true,
                          defaultValue: .numericLiteral("123"), references: nil)),

        ("\"real_column\" REAL DEFAULT 123.123",
        ColumnDefinition(name: "real_column", primaryKey: nil, type: .REAL, nullable: true,
                          defaultValue: .numericLiteral("123.123"), references: nil))
    ]

    #if !os(Linux)
    override class var defaultTestSuite: XCTestSuite {
        let suite = XCTestSuite(forTestCaseClass: ColumnDefinitionTests.self)

        for (expected, column) in ColumnDefinitionTests.definitions {
            let test = ColumnDefinitionTests(selector: #selector(verify))
            test.definition = column
            test.expected = expected
            suite.addTest(test)
        }
        return suite
    }

    @objc func verify() {
        XCTAssertEqual(definition.toSQL(), expected)
    }
    #endif

    func testNullableByDefault() {
        let test = ColumnDefinition(name: "test", type: .REAL)
        XCTAssertEqual(test.name, "test")
        XCTAssertTrue(test.nullable)
        XCTAssertEqual(test.defaultValue, .NULL)
        XCTAssertEqual(test.type, .REAL)
        XCTAssertNil(test.references)
        XCTAssertNil(test.primaryKey)
    }
}

class AffinityTests: XCTestCase {
    func test_init() {
        XCTAssertEqual(ColumnDefinition.Affinity("TEXT"), .TEXT)
        XCTAssertEqual(ColumnDefinition.Affinity("text"), .TEXT)
        XCTAssertEqual(ColumnDefinition.Affinity("INTEGER"), .INTEGER)
        XCTAssertEqual(ColumnDefinition.Affinity("BLOB"), .BLOB)
        XCTAssertEqual(ColumnDefinition.Affinity("REAL"), .REAL)
        XCTAssertEqual(ColumnDefinition.Affinity("NUMERIC"), .NUMERIC)
    }

    // [Determination Of Column Affinity](https://sqlite.org/datatype3.html#determination_of_column_affinity)
    // Rule 1
    func testIntegerAffinity() {
        let declared = [
            "INT",
            "INTEGER",
            "TINYINT",
            "SMALLINT",
            "MEDIUMINT",
            "BIGINT",
            "UNSIGNED BIG INT",
            "INT2",
            "INT8"
        ]
        XCTAssertTrue(declared.allSatisfy({ColumnDefinition.Affinity($0) == .INTEGER}))
    }

    // Rule 2
    func testTextAffinity() {
        let declared = [
            "CHARACTER(20)",
            "VARCHAR(255)",
            "VARYING CHARACTER(255)",
            "NCHAR(55)",
            "NATIVE CHARACTER(70)",
            "NVARCHAR(100)",
            "TEXT",
            "CLOB"
        ]
        XCTAssertTrue(declared.allSatisfy({ColumnDefinition.Affinity($0) == .TEXT}))
    }

    // Rule 3
    func testBlobAffinity() {
        XCTAssertEqual(ColumnDefinition.Affinity("BLOB"), .BLOB)
    }

    // Rule 4
    func testRealAffinity() {
        let declared = [
            "REAL",
            "DOUBLE",
            "DOUBLE PRECISION",
            "FLOAT"
        ]
        XCTAssertTrue(declared.allSatisfy({ColumnDefinition.Affinity($0) == .REAL}))
    }

    // Rule 5
    func testNumericAffinity() {
        let declared = [
            "NUMERIC",
            "DECIMAL(10,5)",
            "BOOLEAN",
            "DATE",
            "DATETIME"
        ]
        XCTAssertTrue(declared.allSatisfy({ColumnDefinition.Affinity($0) == .NUMERIC}))
    }

    func test_returns_NUMERIC_for_unknown_type() {
        XCTAssertEqual(ColumnDefinition.Affinity("baz"), .NUMERIC)
    }
}

class IndexDefinitionTests: XCTestCase {
    var definition: IndexDefinition!
    var expected: String!
    var ifNotExists: Bool!

    static let definitions: [(IndexDefinition, Bool, String)] = [
        (IndexDefinition(table: "tests", name: "index_tests",
                         unique: false,
                         columns: ["test_column"],
                         where: nil,
                         orders: nil),
        false,
        "CREATE INDEX \"index_tests\" ON \"tests\" (\"test_column\")"),

        (IndexDefinition(table: "tests", name: "index_tests",
                         unique: true,
                         columns: ["test_column"],
                         where: nil,
                         orders: nil),
        false,
        "CREATE UNIQUE INDEX \"index_tests\" ON \"tests\" (\"test_column\")"),

        (IndexDefinition(table: "tests", name: "index_tests",
                         unique: true,
                         columns: ["test_column", "bar_column"],
                         where: "test_column IS NOT NULL",
                         orders: nil),
        false,
        "CREATE UNIQUE INDEX \"index_tests\" ON \"tests\" (\"test_column\", \"bar_column\") WHERE test_column IS NOT NULL"),

        (IndexDefinition(table: "tests", name: "index_tests",
                         unique: true,
                         columns: ["test_column", "bar_column"],
                         where: nil,
                         orders: ["test_column": .DESC]),
        false,
        "CREATE UNIQUE INDEX \"index_tests\" ON \"tests\" (\"test_column\" DESC, \"bar_column\")"),

        (IndexDefinition(table: "tests", name: "index_tests",
                         unique: false,
                         columns: ["test_column"],
                         where: nil,
                         orders: nil),
        true,
        "CREATE INDEX IF NOT EXISTS \"index_tests\" ON \"tests\" (\"test_column\")")
    ]

    #if !os(Linux)
    override class var defaultTestSuite: XCTestSuite {
        let suite = XCTestSuite(forTestCaseClass: IndexDefinitionTests.self)

        for (column, ifNotExists, expected) in IndexDefinitionTests.definitions {
            let test = IndexDefinitionTests(selector: #selector(verify))
            test.definition = column
            test.expected = expected
            test.ifNotExists = ifNotExists
            suite.addTest(test)
        }
        return suite
    }

    @objc func verify() {
        XCTAssertEqual(definition.toSQL(ifNotExists: ifNotExists), expected)
    }
    #endif

    func test_validate() {

        let longIndex = IndexDefinition(
                table: "tests",
                name: String(repeating: "x", count: 65),
                unique: false,
                columns: ["test_column"],
                where: nil,
                orders: nil)

        XCTAssertThrowsError(try longIndex.validate()) { error in
            XCTAssertEqual(error.localizedDescription,
                           "Index name 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' " +
                           "on table 'tests' is too long; the limit is 64 characters")
        }
    }

    func test_rename() {
        let index = IndexDefinition(table: "tests", name: "index_tests_something",
                                    unique: true,
                                    columns: ["test_column"],
                                    where: "test_column IS NOT NULL",
                                    orders: nil)

        let renamedIndex = index.renameTable(to: "foo")

        XCTAssertEqual(renamedIndex,
           IndexDefinition(
                table: "foo",
                name: "index_tests_something",
                unique: true,
                columns: ["test_column"],
                where: "test_column IS NOT NULL",
                orders: nil
           )
        )
    }
}

class ForeignKeyDefinitionTests: XCTestCase {
    func test_toSQL() {
        XCTAssertEqual(
            ColumnDefinition.ForeignKey(
                table: "foo",
                column: "bar",
                primaryKey: "bar_id",
                onUpdate: nil,
                onDelete: "SET NULL"
            ).toSQL(), """
               REFERENCES "foo" ("bar_id") ON DELETE SET NULL
               """
        )
    }
}

class TableDefinitionTests: XCTestCase {
    func test_quoted_columnList() {
        let definition = TableDefinition(name: "foo", columns: [
            ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil),
            ColumnDefinition(name: "baz", primaryKey: nil, type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)
        ], indexes: [])

        XCTAssertEqual(definition.quotedColumnList, """
                                                    "id", "baz"
                                                    """)
    }

    func test_toSQL() {
        let definition = TableDefinition(name: "foo", columns: [
            ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)
        ], indexes: [])

        XCTAssertEqual(definition.toSQL(), """
                                           CREATE TABLE foo ( \"id\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL )
                                           """)
    }

    func test_toSQL_temp_table() {
        let definition = TableDefinition(name: "foo", columns: [
            ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)
        ], indexes: [])

        XCTAssertEqual(definition.toSQL(temporary: true), """
                                                          CREATE TEMPORARY TABLE foo ( \"id\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL )
                                                          """)
    }

    func test_copySQL() {
        let from = TableDefinition(name: "from_table", columns: [
            ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)
        ], indexes: [])

        let to = TableDefinition(name: "to_table", columns: [
            ColumnDefinition(name: "id", primaryKey: .init(), type: .INTEGER, nullable: false, defaultValue: .NULL, references: nil)
        ], indexes: [])

        XCTAssertEqual(from.copySQL(to: to), """
                                             INSERT INTO "to_table" ("id") SELECT "id" FROM "from_table"
                                             """)
    }
}

class PrimaryKeyTests: XCTestCase {
    func test_toSQL() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(autoIncrement: false).toSQL(),
            "PRIMARY KEY"
        )
    }

    func test_toSQL_autoincrement() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(autoIncrement: true).toSQL(),
            "PRIMARY KEY AUTOINCREMENT"
        )
    }

    func test_toSQL_on_conflict() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(autoIncrement: false, onConflict: .ROLLBACK).toSQL(),
            "PRIMARY KEY ON CONFLICT ROLLBACK"
        )
    }

    func test_fromSQL() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(sql: "PRIMARY KEY"),
            ColumnDefinition.PrimaryKey(autoIncrement: false)
        )
    }

    func test_fromSQL_invalid_sql_is_nil() {
        XCTAssertNil(ColumnDefinition.PrimaryKey(sql: "FOO"))
    }

    func test_fromSQL_autoincrement() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(sql: "PRIMARY KEY AUTOINCREMENT"),
            ColumnDefinition.PrimaryKey(autoIncrement: true)
        )
    }

    func test_fromSQL_on_conflict() {
        XCTAssertEqual(
            ColumnDefinition.PrimaryKey(sql: "PRIMARY KEY ON CONFLICT ROLLBACK"),
            ColumnDefinition.PrimaryKey(autoIncrement: false, onConflict: .ROLLBACK)
        )
    }
}

class LiteralValueTests: XCTestCase {
    func test_recognizes_TRUE() {
        XCTAssertEqual(LiteralValue("TRUE"), .TRUE)
    }

    func test_recognizes_FALSE() {
        XCTAssertEqual(LiteralValue("FALSE"), .FALSE)
    }

    func test_recognizes_NULL() {
        XCTAssertEqual(LiteralValue("NULL"), .NULL)
    }

    func test_recognizes_nil() {
        XCTAssertEqual(LiteralValue(nil), .NULL)
    }

    func test_recognizes_CURRENT_TIME() {
        XCTAssertEqual(LiteralValue("CURRENT_TIME"), .CURRENT_TIME)
    }

    func test_recognizes_CURRENT_TIMESTAMP() {
        XCTAssertEqual(LiteralValue("CURRENT_TIMESTAMP"), .CURRENT_TIMESTAMP)
    }

    func test_recognizes_CURRENT_DATE() {
        XCTAssertEqual(LiteralValue("CURRENT_DATE"), .CURRENT_DATE)
    }

    func test_recognizes_double_quote_string_literals() {
        XCTAssertEqual(LiteralValue("\"foo\""), .stringLiteral("foo"))
    }

    func test_recognizes_single_quote_string_literals() {
        XCTAssertEqual(LiteralValue("\'foo\'"), .stringLiteral("foo"))
    }

    func test_unquotes_double_quote_string_literals() {
        XCTAssertEqual(LiteralValue("\"fo\"\"o\""), .stringLiteral("fo\"o"))
    }

    func test_unquotes_single_quote_string_literals() {
        XCTAssertEqual(LiteralValue("'fo''o'"), .stringLiteral("fo'o"))
    }

    func test_recognizes_numeric_literals() {
        XCTAssertEqual(LiteralValue("1.2"), .numericLiteral("1.2"))
        XCTAssertEqual(LiteralValue("0xdeadbeef"), .numericLiteral("0xdeadbeef"))
    }

    func test_recognizes_blob_literals() {
        XCTAssertEqual(LiteralValue("X'deadbeef'"), .blobLiteral("deadbeef"))
        XCTAssertEqual(LiteralValue("x'deadbeef'"), .blobLiteral("deadbeef"))
    }

    func test_description_TRUE() {
        XCTAssertEqual(LiteralValue.TRUE.description, "TRUE")
    }

    func test_description_FALSE() {
        XCTAssertEqual(LiteralValue.FALSE.description, "FALSE")
    }

    func test_description_NULL() {
        XCTAssertEqual(LiteralValue.NULL.description, "NULL")
    }

    func test_description_CURRENT_TIME() {
        XCTAssertEqual(LiteralValue.CURRENT_TIME.description, "CURRENT_TIME")
    }

    func test_description_CURRENT_TIMESTAMP() {
        XCTAssertEqual(LiteralValue.CURRENT_TIMESTAMP.description, "CURRENT_TIMESTAMP")
    }

    func test_description_CURRENT_DATE() {
        XCTAssertEqual(LiteralValue.CURRENT_DATE.description, "CURRENT_DATE")
    }

    func test_description_string_literal() {
        XCTAssertEqual(LiteralValue.stringLiteral("foo").description, "'foo'")
    }

    func test_description_numeric_literal() {
        XCTAssertEqual(LiteralValue.numericLiteral("1.2").description, "1.2")
        XCTAssertEqual(LiteralValue.numericLiteral("0xdeadbeef").description, "0xdeadbeef")
    }

    func test_description_blob_literal() {
        XCTAssertEqual(LiteralValue.blobLiteral("deadbeef").description, "X'deadbeef'")
    }
}
