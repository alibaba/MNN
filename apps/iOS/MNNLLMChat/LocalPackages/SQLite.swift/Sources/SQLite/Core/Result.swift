#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

public enum Result: Error {

    fileprivate static let successCodes: Set = [SQLITE_OK, SQLITE_ROW, SQLITE_DONE]

    /// Represents a SQLite specific [error code](https://sqlite.org/rescode.html)
    ///
    /// - message: English-language text that describes the error
    ///
    /// - code: SQLite [error code](https://sqlite.org/rescode.html#primary_result_code_list)
    ///
    /// - statement: the statement which produced the error
    case error(message: String, code: Int32, statement: Statement?)

    /// Represents a SQLite specific [extended error code] (https://sqlite.org/rescode.html#primary_result_codes_versus_extended_result_codes)
    ///
    /// - message: English-language text that describes the error
    ///
    /// - extendedCode: SQLite [extended error code](https://sqlite.org/rescode.html#extended_result_code_list)
    ///
    /// - statement: the statement which produced the error
    case extendedError(message: String, extendedCode: Int32, statement: Statement?)

    init?(errorCode: Int32, connection: Connection, statement: Statement? = nil) {
        guard !Result.successCodes.contains(errorCode) else { return nil }

        let message = String(cString: sqlite3_errmsg(connection.handle))

        guard connection.usesExtendedErrorCodes else {
            self = .error(message: message, code: errorCode, statement: statement)
            return
        }

        let extendedErrorCode = sqlite3_extended_errcode(connection.handle)
        self = .extendedError(message: message, extendedCode: extendedErrorCode, statement: statement)
    }

}

extension Result: CustomStringConvertible {

    public var description: String {
        switch self {
        case let .error(message, errorCode, statement):
            if let statement {
                return "\(message) (\(statement)) (code: \(errorCode))"
            } else {
                return "\(message) (code: \(errorCode))"
            }
        case let .extendedError(message, extendedCode, statement):
            if let statement {
                return "\(message) (\(statement)) (extended code: \(extendedCode))"
            } else {
                return "\(message) (extended code: \(extendedCode))"
            }
        }
    }
}
