import Foundation

/*
    https://www.sqlite.org/lang_altertable.html

   The only schema altering commands directly supported by SQLite are the "rename table" and "add column"
   commands shown above.

    (SQLite 3.25.0: RENAME COLUMN)
    (SQLite 3.35.0: DROP COLUMN)

   However, applications can make other arbitrary changes to the format of a table using a
   simple sequence of operations. The steps to make arbitrary changes to the schema design of some table X are as follows:

        1. If foreign key constraints are enabled, disable them using PRAGMA foreign_keys=OFF.
        2. Start a transaction.
        3. Remember the format of all indexes and triggers associated with table X
           (SELECT sql FROM sqlite_master WHERE tbl_name='X' AND type='index')
        4. Use CREATE TABLE to construct a new table "new_X" that is in the desired revised format of table X.
        5. Transfer content from X into new_X using a statement like: INSERT INTO new_X SELECT ... FROM X.
        6. Drop the old table X: DROP TABLE X.
        7. Change the name of new_X to X using: ALTER TABLE new_X RENAME TO X.
        8. Use CREATE INDEX and CREATE TRIGGER to reconstruct indexes and triggers associated with table X.
        9. If any views refer to table X in a way that is affected by the schema change, then drop those views using DROP VIEW
       10. If foreign key constraints were originally enabled then run PRAGMA foreign_key_check
       11. Commit the transaction started in step 2.
       12. If foreign keys constraints were originally enabled, reenable them now.
*/
public class SchemaChanger: CustomStringConvertible {
    public enum Error: LocalizedError {
        case invalidColumnDefinition(String)
        case foreignKeyError([ForeignKeyError])

        public var errorDescription: String? {
            switch self {
            case .foreignKeyError(let errors):
                return "Foreign key errors: \(errors)"
            case .invalidColumnDefinition(let message):
                return "Invalid column definition: \(message)"
            }
        }
    }

    public enum Operation {
        case addColumn(ColumnDefinition)
        case dropColumn(String)
        case renameColumn(String, String)
        case renameTable(String)

        /// Returns non-nil if the operation can be executed with a simple SQL statement
        func toSQL(_ table: String, version: SQLiteVersion) -> String? {
            switch self {
            case .addColumn(let definition):
                return "ALTER TABLE \(table.quote()) ADD COLUMN \(definition.toSQL())"
            case .renameColumn(let from, let to) where SQLiteFeature.renameColumn.isSupported(by: version):
                return "ALTER TABLE \(table.quote()) RENAME COLUMN \(from.quote()) TO \(to.quote())"
            case .dropColumn(let column) where SQLiteFeature.dropColumn.isSupported(by: version):
                return "ALTER TABLE \(table.quote()) DROP COLUMN \(column.quote())"
            default: return nil
            }
        }

        func validate() throws {
            switch self {
            case .addColumn(let definition):
                // The new column may take any of the forms permissible in a CREATE TABLE statement, with the following restrictions:
                // - The column may not have a PRIMARY KEY or UNIQUE constraint.
                // - The column may not have a default value of CURRENT_TIME, CURRENT_DATE, CURRENT_TIMESTAMP, or an expression in parentheses
                // - If a NOT NULL constraint is specified, then the column must have a default value other than NULL.
                guard definition.primaryKey == nil else {
                    throw Error.invalidColumnDefinition("can not add primary key column")
                }
                let invalidValues: [LiteralValue] = [.CURRENT_TIME, .CURRENT_DATE, .CURRENT_TIMESTAMP]
                if invalidValues.contains(definition.defaultValue) {
                    throw Error.invalidColumnDefinition("Invalid default value")
                }
                if !definition.nullable && definition.defaultValue == .NULL {
                    throw Error.invalidColumnDefinition("NOT NULL columns must have a default value other than NULL")
                }
            case .dropColumn:
                // The DROP COLUMN command only works if the column is not referenced by any other parts of the schema
                // and is not a PRIMARY KEY and does not have a UNIQUE constraint
                break
            default: break
            }
        }
    }

    public class AlterTableDefinition {
        fileprivate var operations: [Operation] = []

        let name: String

        init(name: String) {
            self.name = name
        }

        public func add(column: ColumnDefinition) {
            operations.append(.addColumn(column))
        }

        public func drop(column: String) {
            operations.append(.dropColumn(column))
        }

        public func rename(column: String, to: String) {
            operations.append(.renameColumn(column, to))
        }
    }

    private let connection: Connection
    private let schemaReader: SchemaReader
    private let version: SQLiteVersion
    static let tempPrefix = "tmp_"
    typealias Block = () throws -> Void
    public typealias AlterTableDefinitionBlock = (AlterTableDefinition) -> Void

    struct Options: OptionSet {
        let rawValue: Int
        static let `default`: Options = []
        static let temp = Options(rawValue: 1)
    }

    public convenience init(connection: Connection) {
        self.init(connection: connection,
                  version: connection.sqliteVersion)
    }

    init(connection: Connection, version: SQLiteVersion) {
        self.connection = connection
        schemaReader = connection.schema
        self.version = version
    }

    public func alter(table: String, block: AlterTableDefinitionBlock) throws {
        let alterTableDefinition = AlterTableDefinition(name: table)
        block(alterTableDefinition)

        for operation in alterTableDefinition.operations {
            try run(table: table, operation: operation)
        }
    }

    public func drop(table: String, ifExists: Bool = true) throws {
        try dropTable(table, ifExists: ifExists)
    }

    // Beginning with release 3.25.0 (2018-09-15), references to the table within trigger bodies and
    // view definitions are also renamed.
    public func rename(table: String, to: String) throws {
        try connection.run("ALTER TABLE \(table.quote()) RENAME TO \(to.quote())")
    }

    private func run(table: String, operation: Operation) throws {
        try operation.validate()

        if let sql = operation.toSQL(table, version: version) {
            try connection.run(sql)
        } else {
            try doTheTableDance(table: table, operation: operation)
        }
    }

    private func doTheTableDance(table: String, operation: Operation) throws {
        try connection.transaction {
            try disableRefIntegrity {
                let tempTable = "\(SchemaChanger.tempPrefix)\(table)"
                try moveTable(from: table, to: tempTable, options: [.temp], operation: operation)
                try rename(table: tempTable, to: table)
                let foreignKeyErrors = try connection.foreignKeyCheck()
                if foreignKeyErrors.count > 0 {
                    throw Error.foreignKeyError(foreignKeyErrors)
                }
            }
        }
    }

    private func disableRefIntegrity(block: Block) throws {
        let oldForeignKeys = connection.foreignKeys
        let oldDeferForeignKeys = connection.deferForeignKeys

        connection.deferForeignKeys = true
        connection.foreignKeys = false

        defer {
            connection.deferForeignKeys = oldDeferForeignKeys
            connection.foreignKeys = oldForeignKeys
        }

        try block()
    }

    private func moveTable(from: String, to: String, options: Options = .default, operation: Operation? = nil) throws {
        try copyTable(from: from, to: to, options: options, operation: operation)
        try dropTable(from, ifExists: true)
    }

    private func copyTable(from: String, to: String, options: Options = .default, operation: Operation?) throws {
        let fromDefinition = TableDefinition(
            name: from,
            columns: try schemaReader.columnDefinitions(table: from),
            indexes: try schemaReader.indexDefinitions(table: from)
        )
        let toDefinition   = fromDefinition
                .apply(.renameTable(to))
                .apply(operation)

        try createTable(definition: toDefinition, options: options)
        try createTableIndexes(definition: toDefinition)
        if case .dropColumn = operation {
            try copyTableContents(from: fromDefinition.apply(operation), to: toDefinition)
        } else {
            try copyTableContents(from: fromDefinition, to: toDefinition)
        }
    }

    private func createTable(definition: TableDefinition, options: Options) throws {
        try connection.run(definition.toSQL(temporary: options.contains(.temp)))
    }

    private func createTableIndexes(definition: TableDefinition) throws {
        for index in definition.indexes {
            try index.validate()
            try connection.run(index.toSQL())
        }
    }

    private func dropTable(_ table: String, ifExists: Bool) throws {
        try connection.run("DROP TABLE \(ifExists ? "IF EXISTS" : "") \(table.quote())")
    }

    private func copyTableContents(from: TableDefinition, to: TableDefinition) throws {
        try connection.run(from.copySQL(to: to))
    }

    public var description: String {
        "SQLiteSchemaChanger: \(connection.description)"
    }
}

extension IndexDefinition {
    func renameTable(to: String) -> IndexDefinition {
        func indexName() -> String {
            if to.starts(with: SchemaChanger.tempPrefix) {
                return "\(SchemaChanger.tempPrefix)\(name)"
            } else if table.starts(with: SchemaChanger.tempPrefix) {
                return name.replacingOccurrences(of: SchemaChanger.tempPrefix, with: "")
            } else {
                return name
            }
        }
        return IndexDefinition(table: to, name: indexName(), unique: unique, columns: columns, where: `where`, orders: orders)
    }

    func renameColumn(from: String, to: String) -> IndexDefinition {
        IndexDefinition(table: table, name: name, unique: unique, columns: columns.map {
            $0 == from ? to : $0
        }, where: `where`, orders: orders)
    }
}

extension TableDefinition {
    func apply(_ operation: SchemaChanger.Operation?) -> TableDefinition {
        switch operation {
        case .none: return self
        case .addColumn: fatalError("Use 'ALTER TABLE ADD COLUMN (...)'")
        case .dropColumn(let column):
            return TableDefinition(name: name,
                columns: columns.filter { $0.name != column },
                indexes: indexes.filter { !$0.columns.contains(column) }
            )
        case .renameColumn(let from, let to):
            return TableDefinition(
                name: name,
                columns: columns.map { $0.rename(from: from, to: to) },
                indexes: indexes.map { $0.renameColumn(from: from, to: to) }
        )
        case .renameTable(let to):
            return TableDefinition(name: to, columns: columns, indexes: indexes.map { $0.renameTable(to: to) })
        }
    }
}
