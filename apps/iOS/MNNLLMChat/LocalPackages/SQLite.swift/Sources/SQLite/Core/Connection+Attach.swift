import Foundation
#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

extension Connection {
    #if SQLITE_SWIFT_SQLCIPHER
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#attach
    public func attach(_ location: Location, as schemaName: String, key: String? = nil) throws {
        if let key {
            try run("ATTACH DATABASE ? AS ? KEY ?", location.description, schemaName, key)
        } else {
            try run("ATTACH DATABASE ? AS ?", location.description, schemaName)
        }
    }
    #else
    /// See  https://www3.sqlite.org/lang_attach.html
    public func attach(_ location: Location, as schemaName: String) throws {
        try run("ATTACH DATABASE ? AS ?", location.description, schemaName)
    }
    #endif

    /// See https://www3.sqlite.org/lang_detach.html
    public func detach(_ schemaName: String) throws {
        try run("DETACH DATABASE ?", schemaName)
    }
}
