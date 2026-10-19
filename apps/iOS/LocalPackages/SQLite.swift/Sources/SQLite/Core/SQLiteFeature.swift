import Foundation

enum SQLiteFeature {
    case partialIntegrityCheck      // PRAGMA integrity_check(table)
    case sqliteSchemaTable          // sqlite_master => sqlite_schema
    case renameColumn               // ALTER TABLE ... RENAME COLUMN
    case dropColumn                 // ALTER TABLE ... DROP COLUMN

    func isSupported(by version: SQLiteVersion) -> Bool {
        switch self {
        case .partialIntegrityCheck, .sqliteSchemaTable:
            return version >= .init(major: 3, minor: 33)
        case .renameColumn:
            return version >= .init(major: 3, minor: 25)
        case .dropColumn:
            return version >= .init(major: 3, minor: 35)
        }
    }
}

extension Connection {
    func supports(_ feature: SQLiteFeature) -> Bool {
        feature.isSupported(by: sqliteVersion)
    }
}
