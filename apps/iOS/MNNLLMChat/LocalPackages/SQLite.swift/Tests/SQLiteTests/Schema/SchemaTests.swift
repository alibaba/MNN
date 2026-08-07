import XCTest
import SQLite

class SchemaTests: XCTestCase {

    func test_drop_compilesDropTableExpression() {
        XCTAssertEqual("DROP TABLE \"table\"", table.drop())
        XCTAssertEqual("DROP TABLE IF EXISTS \"table\"", table.drop(ifExists: true))
    }

    func test_drop_compilesDropVirtualTableExpression() {
        XCTAssertEqual("DROP TABLE \"virtual_table\"", virtualTable.drop())
        XCTAssertEqual("DROP TABLE IF EXISTS \"virtual_table\"", virtualTable.drop(ifExists: true))
    }

    func test_drop_compilesDropViewExpression() {
        XCTAssertEqual("DROP VIEW \"view\"", _view.drop())
        XCTAssertEqual("DROP VIEW IF EXISTS \"view\"", _view.drop(ifExists: true))
    }

    func test_create_withBuilder_compilesCreateTableExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (" +
                "\"blob\" BLOB NOT NULL, " +
                "\"blobOptional\" BLOB, " +
                "\"double\" REAL NOT NULL, " +
                "\"doubleOptional\" REAL, " +
                "\"int64\" INTEGER NOT NULL, " +
                "\"int64Optional\" INTEGER, " +
                "\"string\" TEXT NOT NULL, " +
                "\"stringOptional\" TEXT" +
            ")",
            table.create { t in
                t.column(data)
                t.column(dataOptional)
                t.column(double)
                t.column(doubleOptional)
                t.column(int64)
                t.column(int64Optional)
                t.column(string)
                t.column(stringOptional)
            }
        )
        XCTAssertEqual(
            "CREATE TEMPORARY TABLE \"table\" (\"int64\" INTEGER NOT NULL)",
            table.create(temporary: true) { $0.column(int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE IF NOT EXISTS \"table\" (\"int64\" INTEGER NOT NULL)",
            table.create(ifNotExists: true) { $0.column(int64) }
        )
        XCTAssertEqual(
            "CREATE TEMPORARY TABLE IF NOT EXISTS \"table\" (\"int64\" INTEGER NOT NULL)",
            table.create(temporary: true, ifNotExists: true) { $0.column(int64) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL) WITHOUT ROWID",
            table.create(withoutRowid: true) { $0.column(int64) }
        )
        XCTAssertEqual(
            "CREATE TEMPORARY TABLE IF NOT EXISTS \"table\" (\"int64\" INTEGER NOT NULL) WITHOUT ROWID",
            table.create(temporary: true, ifNotExists: true, withoutRowid: true) { $0.column(int64) }
        )
    }

    func test_create_withQuery_compilesCreateTableExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" AS SELECT \"int64\" FROM \"view\"",
            table.create(_view.select(int64))
        )
    }

    // thoroughness test for ambiguity
    // swiftlint:disable:next function_body_length
    func test_column_compilesColumnDefinitionExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL)",
            table.create { t in t.column(int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE)",
            table.create { t in t.column(int64, unique: true) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL DEFAULT (0))",
            table.create { t in t.column(int64, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, unique: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, unique: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, unique: true, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE DEFAULT (0))",
            table.create { t in t.column(int64, unique: true, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, unique: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, unique: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, unique: true, check: int64 > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, unique: true, check: int64Optional > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64, unique: true, check: int64 > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64, unique: true, check: int64Optional > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, check: int64 > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, check: int64Optional > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64, check: int64 > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64, check: int64Optional > 0, defaultValue: 0) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, primaryKey: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, primaryKey: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, primaryKey: true, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, primaryKey: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, primaryKey: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, primaryKey: true, check: int64 > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64Optional\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64, primaryKey: true, check: int64Optional > 0, defaultValue: int64) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER)",
            table.create { t in t.column(int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE)",
            table.create { t in t.column(int64Optional, unique: true) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64Optional, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64Optional, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER DEFAULT (0))",
            table.create { t in t.column(int64Optional, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, unique: true, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE DEFAULT (0))",
            table.create { t in t.column(int64Optional, unique: true, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0) DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0) DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, check: int64 > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) DEFAULT (\"int64\"))",
            table.create { t in t.column(int64Optional, check: int64Optional > 0, defaultValue: int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64\" > 0) DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, check: int64 > 0, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) DEFAULT (\"int64Optional\"))",
            table.create { t in t.column(int64Optional, check: int64Optional > 0, defaultValue: int64Optional) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64Optional, check: int64 > 0, defaultValue: 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) DEFAULT (0))",
            table.create { t in t.column(int64Optional, check: int64Optional > 0, defaultValue: 0) }
        )
    }

    func test_column_withIntegerExpression_compilesPrimaryKeyAutoincrementColumnDefinitionExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL)",
            table.create { t in t.column(int64, primaryKey: .autoincrement) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL CHECK (\"int64\" > 0))",
            table.create { t in t.column(int64, primaryKey: .autoincrement, check: int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL CHECK (\"int64Optional\" > 0))",
            table.create { t in t.column(int64, primaryKey: .autoincrement, check: int64Optional > 0) }
        )
    }

    func test_column_withIntegerExpression_compilesReferentialColumnDefinitionExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, references: qualifiedTable, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, unique: true, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, primaryKey: true, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, check: int64Optional > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, unique: true, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64, primaryKey: true, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0) REFERENCES
             \"table\" (\"int64\"))
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(int64, unique: true, check: int64Optional > 0, references: table, int64) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"int64\" INTEGER PRIMARY KEY NOT NULL CHECK (\"int64Optional\" > 0) REFERENCES
             \"table\" (\"int64\"))
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(int64, primaryKey: true, check: int64Optional > 0, references: table, int64) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER PRIMARY KEY REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, primaryKey: true, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, check: int64Optional > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER PRIMARY KEY CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, primaryKey: true, check: int64 > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, unique: true, check: int64Optional > 0, references: table, int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"int64Optional\" INTEGER PRIMARY KEY CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\"))",
            table.create { t in t.column(int64Optional, primaryKey: true, check: int64Optional > 0, references: table, int64) }
        )
    }

    // swiftlint:disable:next function_body_length
    func test_column_withStringExpression_compilesCollatedColumnDefinitionExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL COLLATE RTRIM)",
            table.create { t in t.column(string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"string\" != '') COLLATE RTRIM)",
            table.create { t in t.column(string, check: string != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"stringOptional\" != '') COLLATE RTRIM)",
            table.create { t in t.column(string, check: stringOptional != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"string\" != '') COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: string != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"stringOptional\" != '') COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: stringOptional != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"string\" != '') DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: string != "", defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"stringOptional\" != '') DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: stringOptional != "", defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"string\" != '') DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: string != "", defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL UNIQUE CHECK (\"stringOptional\" != '') DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, unique: true, check: stringOptional != "", defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"string\" != '') DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, check: string != "", defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"stringOptional\" != '') DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(string, check: stringOptional != "", defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"string\" != '') DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, check: string != "", defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"string\" TEXT NOT NULL CHECK (\"stringOptional\" != '') DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(string, check: stringOptional != "", defaultValue: "string", collate: .rtrim) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"string\" != '') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, check: string != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"stringOptional\" != '') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, check: stringOptional != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT DEFAULT (\"stringOptional\") COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, defaultValue: stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"string\" != '') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, check: string != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"stringOptional\" != '') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, check: stringOptional != "", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE DEFAULT (\"stringOptional\") COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, defaultValue: stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE DEFAULT ('string') COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"string\" != '') DEFAULT (\"string\") COLLATE RTRIM)",
            table.create { t in t.column(stringOptional, unique: true, check: string != "", defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"string\" != '')
             DEFAULT (\"stringOptional\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional,
                                         unique: true,
                                         check: string != "",
                                         defaultValue: stringOptional,
                                         collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"stringOptional\" != '')
             DEFAULT (\"string\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, unique: true, check: stringOptional != "",
                                         defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"stringOptional\" != '')
             DEFAULT (\"stringOptional\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, unique: true, check: stringOptional != "",
                                         defaultValue: stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"string\" != '')
             DEFAULT ('string') COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, unique: true, check: string != "",
                                         defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT UNIQUE CHECK (\"stringOptional\" != '')
             DEFAULT ('string') COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, unique: true, check: stringOptional != "",
                                         defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"string\" != '')
             DEFAULT (\"string\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: string != "",
                                         defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"stringOptional\" != '')
             DEFAULT (\"string\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: stringOptional != "",
                                         defaultValue: string, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"string\" != '')
             DEFAULT (\"stringOptional\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: string != "",
                                         defaultValue: stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"stringOptional\" != '')
             DEFAULT (\"stringOptional\") COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: stringOptional != "",
                                         defaultValue: stringOptional, collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"string\" != '')
             DEFAULT ('string') COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: string != "",
                                         defaultValue: "string", collate: .rtrim) }
        )
        XCTAssertEqual(
            """
            CREATE TABLE \"table\" (\"stringOptional\" TEXT CHECK (\"stringOptional\" != '')
             DEFAULT ('string') COLLATE RTRIM)
            """.replacingOccurrences(of: "\n", with: ""),
            table.create { t in t.column(stringOptional, check: stringOptional != "",
                                         defaultValue: "string", collate: .rtrim) }
        )
    }

    func test_primaryKey_compilesPrimaryKeyExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (PRIMARY KEY (\"int64\"))",
            table.create { t in t.primaryKey(int64) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (PRIMARY KEY (\"int64\", \"string\"))",
            table.create { t in t.primaryKey(int64, string) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (PRIMARY KEY (\"int64\", \"string\", \"double\"))",
            table.create { t in t.primaryKey(int64, string, double) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (PRIMARY KEY (\"int64\", \"string\", \"double\", \"date\"))",
            table.create { t in t.primaryKey(int64, string, double, date) }
        )
    }

    func test_unique_compilesUniqueExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (UNIQUE (\"int64\"))",
            table.create { t in t.unique(int64) }
        )
    }

    func test_check_compilesCheckExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (CHECK ((\"int64\" > 0)))",
            table.create { t in t.check(int64 > 0) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (CHECK ((\"int64Optional\" > 0)))",
            table.create { t in t.check(int64Optional > 0) }
        )
    }

    func test_foreignKey_compilesForeignKeyExpression() {
        XCTAssertEqual(
            "CREATE TABLE \"table\" (FOREIGN KEY (\"string\") REFERENCES \"table\" (\"string\"))",
            table.create { t in t.foreignKey(string, references: table, string) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (FOREIGN KEY (\"stringOptional\") REFERENCES \"table\" (\"string\"))",
            table.create { t in t.foreignKey(stringOptional, references: table, string) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (FOREIGN KEY (\"string\") REFERENCES \"table\" (\"string\") ON UPDATE CASCADE ON DELETE SET NULL)",
            table.create { t in t.foreignKey(string, references: table, string, update: .cascade, delete: .setNull) }
        )

        XCTAssertEqual(
            "CREATE TABLE \"table\" (FOREIGN KEY (\"string\", \"string\") REFERENCES \"table\" (\"string\", \"string\"))",
            table.create { t in t.foreignKey((string, string), references: table, (string, string)) }
        )
        XCTAssertEqual(
            "CREATE TABLE \"table\" (FOREIGN KEY (\"string\", \"string\", \"string\") REFERENCES \"table\" (\"string\", \"string\", \"string\"))",
            table.create { t in t.foreignKey((string, string, string), references: table, (string, string, string)) }
        )
    }

    func test_addColumn_compilesAlterTableExpression() {
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL DEFAULT (1)",
            table.addColumn(int64, defaultValue: 1)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0) DEFAULT (1)",
            table.addColumn(int64, check: int64 > 0, defaultValue: 1)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0) DEFAULT (1)",
            table.addColumn(int64, check: int64Optional > 0, defaultValue: 1)
        )

        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER",
            table.addColumn(int64Optional)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64\" > 0)",
            table.addColumn(int64Optional, check: int64 > 0)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0)",
            table.addColumn(int64Optional, check: int64Optional > 0)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER DEFAULT (1)",
            table.addColumn(int64Optional, defaultValue: 1)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64\" > 0) DEFAULT (1)",
            table.addColumn(int64Optional, check: int64 > 0, defaultValue: 1)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) DEFAULT (1)",
            table.addColumn(int64Optional, check: int64Optional > 0, defaultValue: 1)
        )
    }

    func test_addColumn_withIntegerExpression_compilesReferentialAlterTableExpression() {
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL UNIQUE REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, unique: true, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, check: int64 > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, check: int64Optional > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, unique: true, check: int64 > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64\" INTEGER NOT NULL UNIQUE CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64, unique: true, check: int64Optional > 0, references: table, int64)
        )

        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER UNIQUE REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, unique: true, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, check: int64 > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, check: int64Optional > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER UNIQUE CHECK (\"int64\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, unique: true, check: int64 > 0, references: table, int64)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"int64Optional\" INTEGER UNIQUE CHECK (\"int64Optional\" > 0) REFERENCES \"table\" (\"int64\")",
            table.addColumn(int64Optional, unique: true, check: int64Optional > 0, references: table, int64)
        )
    }

    func test_addColumn_withStringExpression_compilesCollatedAlterTableExpression() {
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"string\" TEXT NOT NULL DEFAULT ('string') COLLATE RTRIM",
            table.addColumn(string, defaultValue: "string", collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"string\" TEXT NOT NULL CHECK (\"string\" != '') DEFAULT ('string') COLLATE RTRIM",
            table.addColumn(string, check: string != "", defaultValue: "string", collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"string\" TEXT NOT NULL CHECK (\"stringOptional\" != '') DEFAULT ('string') COLLATE RTRIM",
            table.addColumn(string, check: stringOptional != "", defaultValue: "string", collate: .rtrim)
        )

        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"stringOptional\" TEXT COLLATE RTRIM",
            table.addColumn(stringOptional, collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"stringOptional\" TEXT CHECK (\"string\" != '') COLLATE RTRIM",
            table.addColumn(stringOptional, check: string != "", collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"stringOptional\" TEXT CHECK (\"stringOptional\" != '') COLLATE RTRIM",
            table.addColumn(stringOptional, check: stringOptional != "", collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"stringOptional\" TEXT CHECK (\"string\" != '') DEFAULT ('string') COLLATE RTRIM",
            table.addColumn(stringOptional, check: string != "", defaultValue: "string", collate: .rtrim)
        )
        XCTAssertEqual(
            "ALTER TABLE \"table\" ADD COLUMN \"stringOptional\" TEXT CHECK (\"stringOptional\" != '') DEFAULT ('string') COLLATE RTRIM",
            table.addColumn(stringOptional, check: stringOptional != "", defaultValue: "string", collate: .rtrim)
        )
    }

    func test_rename_compilesAlterTableRenameToExpression() {
        XCTAssertEqual("ALTER TABLE \"old\" RENAME TO \"table\"", Table("old").rename(table))
    }

    func test_createIndex_compilesCreateIndexExpression() {
        XCTAssertEqual("CREATE INDEX \"index_table_on_int64\" ON \"table\" (\"int64\")", table.createIndex(int64))

        XCTAssertEqual(
            "CREATE UNIQUE INDEX \"index_table_on_int64\" ON \"table\" (\"int64\")",
            table.createIndex(int64, unique: true)
        )
        XCTAssertEqual(
            "CREATE INDEX IF NOT EXISTS \"index_table_on_int64\" ON \"table\" (\"int64\")",
            table.createIndex(int64, ifNotExists: true)
        )
        XCTAssertEqual(
            "CREATE UNIQUE INDEX IF NOT EXISTS \"index_table_on_int64\" ON \"table\" (\"int64\")",
            table.createIndex(int64, unique: true, ifNotExists: true)
        )
        XCTAssertEqual(
            "CREATE UNIQUE INDEX IF NOT EXISTS \"main\".\"index_table_on_int64\" ON \"table\" (\"int64\")",
            qualifiedTable.createIndex(int64, unique: true, ifNotExists: true)
        )
    }

    func test_dropIndex_compilesCreateIndexExpression() {
        XCTAssertEqual("DROP INDEX \"index_table_on_int64\"", table.dropIndex(int64))
        XCTAssertEqual("DROP INDEX IF EXISTS \"index_table_on_int64\"", table.dropIndex(int64, ifExists: true))
    }

    func test_create_onView_compilesCreateViewExpression() {
        XCTAssertEqual(
            "CREATE VIEW \"view\" AS SELECT \"int64\" FROM \"table\"",
            _view.create(table.select(int64))
        )
        XCTAssertEqual(
            "CREATE TEMPORARY VIEW \"view\" AS SELECT \"int64\" FROM \"table\"",
            _view.create(table.select(int64), temporary: true)
        )
        XCTAssertEqual(
            "CREATE VIEW IF NOT EXISTS \"view\" AS SELECT \"int64\" FROM \"table\"",
            _view.create(table.select(int64), ifNotExists: true)
        )
        XCTAssertEqual(
            "CREATE TEMPORARY VIEW IF NOT EXISTS \"view\" AS SELECT \"int64\" FROM \"table\"",
            _view.create(table.select(int64), temporary: true, ifNotExists: true)
        )
    }

    func test_create_onVirtualTable_compilesCreateVirtualTableExpression() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING \"custom\"('foo', 'bar')",
            virtualTable.create(Module("custom", ["foo", "bar"]))
        )
    }

    func test_rename_onVirtualTable_compilesAlterTableRenameToExpression() {
        XCTAssertEqual(
            "ALTER TABLE \"old\" RENAME TO \"virtual_table\"",
            VirtualTable("old").rename(virtualTable)
        )
    }

}
