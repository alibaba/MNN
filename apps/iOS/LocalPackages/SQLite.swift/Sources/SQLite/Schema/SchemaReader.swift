import Foundation

public class SchemaReader {
    private let connection: Connection

    init(connection: Connection) {
        self.connection = connection
    }

    // https://sqlite.org/pragma.html#pragma_table_info
    //
    // This pragma returns one row for each column in the named table. Columns in the result set include the
    // column name, data type, whether or not the column can be NULL, and the default value for the column. The
    // "pk" column in the result set is zero for columns that are not part of the primary key, and is the
    // index of the column in the primary key for columns that are part of the primary key.
    public func columnDefinitions(table: String) throws -> [ColumnDefinition] {
        func parsePrimaryKey(column: String) throws -> ColumnDefinition.PrimaryKey? {
            try createTableSQL(name: table).flatMap { .init(sql: $0) }
        }

        let foreignKeys: [String: [ColumnDefinition.ForeignKey]] =
            Dictionary(grouping: try foreignKeys(table: table), by: { $0.column })

        return try connection.prepareRowIterator("PRAGMA table_info(\(table.quote()))")
            .map { (row: Row) -> ColumnDefinition in
                ColumnDefinition(
                    name: row[TableInfoTable.nameColumn],
                    primaryKey: (row[TableInfoTable.primaryKeyColumn] ?? 0) > 0 ?
                        try parsePrimaryKey(column: row[TableInfoTable.nameColumn]) : nil,
                    type: ColumnDefinition.Affinity(row[TableInfoTable.typeColumn]),
                    nullable: row[TableInfoTable.notNullColumn] == 0,
                    defaultValue: LiteralValue(row[TableInfoTable.defaultValueColumn]),
                    references: foreignKeys[row[TableInfoTable.nameColumn]]?.first
                )
            }
    }

    public func objectDefinitions(name: String? = nil,
                                  type: ObjectDefinition.ObjectType? = nil,
                                  temp: Bool = false) throws -> [ObjectDefinition] {
        var query: QueryType = SchemaTable.get(for: connection, temp: temp)
        if let name {
            query = query.where(SchemaTable.nameColumn == name)
        }
        if let type {
            query = query.where(SchemaTable.typeColumn == type.rawValue)
        }
        return try connection.prepare(query).map { row -> ObjectDefinition in
            guard let type = ObjectDefinition.ObjectType(rawValue: row[SchemaTable.typeColumn]) else {
                fatalError("unexpected type")
            }
            return ObjectDefinition(
                type: type,
                name: row[SchemaTable.nameColumn],
                tableName: row[SchemaTable.tableNameColumn],
                rootpage: row[SchemaTable.rootPageColumn] ?? 0,
                sql: row[SchemaTable.sqlColumn]
            )
        }
    }

    public func indexDefinitions(table: String) throws -> [IndexDefinition] {
        func indexSQL(name: String) throws -> String? {
            try objectDefinitions(name: name, type: .index)
                .compactMap(\.sql)
                .first
        }

        func columns(name: String) throws -> [String] {
            try connection.prepareRowIterator("PRAGMA index_info(\(name.quote()))")
                .compactMap { row in
                    row[IndexInfoTable.nameColumn]
                }
        }

        return try connection.prepareRowIterator("PRAGMA index_list(\(table.quote()))")
            .compactMap { row -> IndexDefinition? in
                let name = row[IndexListTable.nameColumn]
                guard !name.starts(with: "sqlite_") else {
                    // Indexes SQLite creates implicitly for internal use start with "sqlite_".
                    // See https://www.sqlite.org/fileformat2.html#intschema
                    return nil
                }
                return IndexDefinition(
                    table: table,
                    name: name,
                    unique: row[IndexListTable.uniqueColumn] == 1,
                    columns: try columns(name: name),
                    indexSQL: try indexSQL(name: name)
                )
            }
    }

    func foreignKeys(table: String) throws -> [ColumnDefinition.ForeignKey] {
        try connection.prepareRowIterator("PRAGMA foreign_key_list(\(table.quote()))")
            .map { row in
                ColumnDefinition.ForeignKey(
                    table: row[ForeignKeyListTable.tableColumn],
                    column: row[ForeignKeyListTable.fromColumn],
                    primaryKey: row[ForeignKeyListTable.toColumn],
                    onUpdate: row[ForeignKeyListTable.onUpdateColumn] == TableBuilder.Dependency.noAction.rawValue
                        ? nil : row[ForeignKeyListTable.onUpdateColumn],
                    onDelete: row[ForeignKeyListTable.onDeleteColumn] == TableBuilder.Dependency.noAction.rawValue
                        ? nil : row[ForeignKeyListTable.onDeleteColumn]
                )
            }
    }

    func tableDefinitions() throws -> [TableDefinition] {
        try objectDefinitions(type: .table)
                .map { table in
                    TableDefinition(
                        name: table.name,
                        columns: try columnDefinitions(table: table.name),
                        indexes: try indexDefinitions(table: table.name)
                    )
                }
    }

    private func createTableSQL(name: String) throws -> String? {
        try (
            objectDefinitions(name: name, type: .table) +
            objectDefinitions(name: name, type: .table, temp: true)
        ).compactMap(\.sql).first
    }
}

private enum SchemaTable {
    private static let name = Table("sqlite_schema", database: "main")
    private static let tempName = Table("sqlite_schema", database: "temp")
    // legacy names (< 3.33.0)
    private static let masterName = Table("sqlite_master")
    private static let tempMasterName = Table("sqlite_temp_master")

    static func get(for connection: Connection, temp: Bool = false) -> Table {
        if connection.supports(.sqliteSchemaTable) {
            return temp ? SchemaTable.tempName : SchemaTable.name
        } else {
            return temp ? SchemaTable.tempMasterName : SchemaTable.masterName
        }
    }

    // columns
    static let typeColumn = Expression<String>("type")
    static let nameColumn = Expression<String>("name")
    static let tableNameColumn = Expression<String>("tbl_name")
    static let rootPageColumn = Expression<Int64?>("rootpage")
    static let sqlColumn = Expression<String?>("sql")
}

private enum TableInfoTable {
    static let idColumn = Expression<Int64>("cid")
    static let nameColumn = Expression<String>("name")
    static let typeColumn = Expression<String>("type")
    static let notNullColumn = Expression<Int64>("notnull")
    static let defaultValueColumn = Expression<String?>("dflt_value")
    static let primaryKeyColumn = Expression<Int64?>("pk")
}

private enum IndexInfoTable {
    // The rank of the column within the index. (0 means left-most.)
    static let seqnoColumn = Expression<Int64>("seqno")
    // The rank of the column within the table being indexed.
    // A value of -1 means rowid and a value of -2 means that an expression is being used.
    static let cidColumn = Expression<Int64>("cid")
    // The name of the column being indexed. This columns is NULL if the column is the rowid or an expression.
    static let nameColumn = Expression<String?>("name")
}

private enum IndexListTable {
    // A sequence number assigned to each index for internal tracking purposes.
    static let seqColumn = Expression<Int64>("seq")
    // The name of the index
    static let nameColumn = Expression<String>("name")
    // "1" if the index is UNIQUE and "0" if not.
    static let uniqueColumn = Expression<Int64>("unique")
    // "c" if the index was created by a CREATE INDEX statement,
    // "u" if the index was created by a UNIQUE constraint, or
    // "pk" if the index was created by a PRIMARY KEY constraint.
    static let originColumn = Expression<String>("origin")
    // "1" if the index is a partial index and "0" if not.
    static let partialColumn = Expression<Int64>("partial")
}

private enum ForeignKeyListTable {
    static let idColumn = Expression<Int64>("id")
    static let seqColumn = Expression<Int64>("seq")
    static let tableColumn = Expression<String>("table")
    static let fromColumn = Expression<String>("from")
    static let toColumn = Expression<String?>("to") // when null, use primary key
    static let onUpdateColumn = Expression<String>("on_update")
    static let onDeleteColumn = Expression<String>("on_delete")
    static let matchColumn = Expression<String>("match")
}
