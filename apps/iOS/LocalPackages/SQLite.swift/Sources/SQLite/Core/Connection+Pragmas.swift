import Foundation

public typealias UserVersion = Int32

public extension Connection {
    /// The user version of the database.
    /// See SQLite [PRAGMA user_version](https://sqlite.org/pragma.html#pragma_user_version)
    var userVersion: UserVersion? {
        get {
            (try? scalar("PRAGMA user_version") as? Int64)?.map(Int32.init)
        }
        set {
            _ = try? run("PRAGMA user_version = \(newValue ?? 0)")
        }
    }

    /// The version of SQLite.
    /// See SQLite [sqlite_version()](https://sqlite.org/lang_corefunc.html#sqlite_version)
    var sqliteVersion: SQLiteVersion {
        guard let version = (try? scalar("SELECT sqlite_version()")) as? String,
              let splits = .some(version.split(separator: ".", maxSplits: 3)), splits.count == 3,
              let major = Int(splits[0]), let minor = Int(splits[1]), let point = Int(splits[2]) else {
            return .zero
        }
        return .init(major: major, minor: minor, point: point)
    }

    // Changing the foreign_keys setting affects the execution of all statements prepared using the database
    // connection, including those prepared before the setting was changed.
    //
    // https://sqlite.org/pragma.html#pragma_foreign_keys
    var foreignKeys: Bool {
        get { getBoolPragma("foreign_keys") }
        set { setBoolPragma("foreign_keys", newValue) }
    }

    var deferForeignKeys: Bool {
        get { getBoolPragma("defer_foreign_keys") }
        set { setBoolPragma("defer_foreign_keys", newValue) }
    }

    private func getBoolPragma(_ key: String) -> Bool {
        guard let binding = try? scalar("PRAGMA \(key)"),
              let intBinding = binding as? Int64 else { return false }
        return intBinding == 1
    }

    private func setBoolPragma(_ key: String, _ newValue: Bool) {
        _ = try? run("PRAGMA \(key) = \(newValue ? "1" : "0")")
    }
}
