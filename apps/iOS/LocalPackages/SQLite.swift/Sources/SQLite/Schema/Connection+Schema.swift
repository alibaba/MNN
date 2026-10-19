import Foundation

public extension Connection {
    var schema: SchemaReader { SchemaReader(connection: self) }

    // There are four columns in each result row.
    // The first column is the name of the table that
    // contains the REFERENCES clause.
    // The second column is the rowid of the row that contains the
    // invalid REFERENCES clause, or NULL if the child table is a WITHOUT ROWID table.
    // The third column is the name of the table that is referred to.
    // The fourth column is the index of the specific foreign key constraint that failed.
    //
    // https://sqlite.org/pragma.html#pragma_foreign_key_check
    func foreignKeyCheck(table: String? = nil) throws -> [ForeignKeyError] {
        try run("PRAGMA foreign_key_check" + (table.map { "(\($0.quote()))" } ?? ""))
            .compactMap { (row: [Binding?]) -> ForeignKeyError? in
                guard let table = row[0] as? String,
                      let rowId = row[1] as? Int64,
                      let target = row[2] as? String else { return nil }

                return ForeignKeyError(from: table, rowId: rowId, to: target)
            }
    }

    // This pragma does a low-level formatting and consistency check of the database.
    // https://sqlite.org/pragma.html#pragma_integrity_check
    func integrityCheck(table: String? = nil) throws -> [String] {
        precondition(table == nil || supports(.partialIntegrityCheck), "partial integrity check not supported")

        return try run("PRAGMA integrity_check" + (table.map { "(\($0.quote()))" } ?? ""))
            .compactMap { $0[0] as? String }
            .filter { $0 != "ok" }
    }
}
