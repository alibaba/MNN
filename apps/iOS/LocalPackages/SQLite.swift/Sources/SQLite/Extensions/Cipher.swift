#if SQLITE_SWIFT_SQLCIPHER
import SQLCipher

/// Extension methods for [SQLCipher](https://www.zetetic.net/sqlcipher/).
/// @see [sqlcipher api](https://www.zetetic.net/sqlcipher/sqlcipher-api/)
extension Connection {

    /// - Returns: the SQLCipher version
    ///
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#cipher_version
    public var cipherVersion: String? {
        (try? scalar("PRAGMA cipher_version")) as? String
    }

    /// Specify the key for an encrypted database.  This routine should be
    /// called right after sqlite3_open().
    ///
    /// @param key The key to use.The key itself can be a passphrase, which is converted to a key
    ///            using [PBKDF2](https://en.wikipedia.org/wiki/PBKDF2) key derivation. The result
    ///            is used as the encryption key for the database.
    ///
    ///            Alternatively, it is possible to specify an exact byte sequence using a blob literal.
    ///            With this method, it is the calling application's responsibility to ensure that the data
    ///            provided is a 64 character hex string, which will be converted directly to 32 bytes (256 bits)
    ///            of key data.
    ///            e.g. x'2DD29CA851E7B56E4697B0E1F08507293D761A05CE4D1B628663F411A8086D99'
    /// @param db name of the database, defaults to 'main'
    ///
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#sqlite3_key
    public func key(_ key: String, db: String = "main") throws {
        try _key_v2(db: db, keyPointer: key, keySize: key.utf8.count)
    }

    public func key(_ key: Blob, db: String = "main") throws {
        try _key_v2(db: db, keyPointer: key.bytes, keySize: key.bytes.count)
    }

    /// Same as `key(_ key: String, db: String = "main")`, running "PRAGMA cipher_migrate;"
    /// immediately after calling `sqlite3_key_v2`, which performs the migration of
    /// SQLCipher database created by older major version of SQLCipher, to be able to
    /// open this database with new major version of SQLCipher
    /// (e.g. to open database created by SQLCipher version 3.x.x with SQLCipher version 4.x.x).
    /// As "PRAGMA cipher_migrate;" is time-consuming, it is recommended to use this function
    /// only after failure of `key(_ key: String, db: String = "main")`, if older versions of
    /// your app may ise older version of SQLCipher
    ///
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#cipher_migrate
    /// and https://discuss.zetetic.net/t/upgrading-to-sqlcipher-4/3283
    /// for more details regarding SQLCipher upgrade
    public func keyAndMigrate(_ key: String, db: String = "main") throws {
        try _key_v2(db: db, keyPointer: key, keySize: key.utf8.count, migrate: true)
    }

    /// Same as `[`keyAndMigrate(_ key: String, db: String = "main")` accepting byte array as key
    public func keyAndMigrate(_ key: Blob, db: String = "main") throws {
        try _key_v2(db: db, keyPointer: key.bytes, keySize: key.bytes.count, migrate: true)
    }

    /// Change the key on an open database. NB: only works if the database is already encrypted.
    ///
    /// To change the key on an existing encrypted database, it must first be unlocked with the
    /// current encryption key. Once the database is readable and writeable, rekey can be used
    /// to re-encrypt every page in the database with a new key.
    ///
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#sqlite3_rekey
    public func rekey(_ key: String, db: String = "main") throws {
        try _rekey_v2(db: db, keyPointer: key, keySize: key.utf8.count)
    }

    public func rekey(_ key: Blob, db: String = "main") throws {
        try _rekey_v2(db: db, keyPointer: key.bytes, keySize: key.bytes.count)
    }

    /// Converts a non-encrypted database to an encrypted one.
    ///
    /// See https://www.zetetic.net/sqlcipher/sqlcipher-api/#sqlcipher_export
    public func sqlcipher_export(_ location: Location, key: String) throws {
        let schemaName = "cipher_export"

        try attach(location, as: schemaName, key: key)
        try run("SELECT sqlcipher_export(?)", schemaName)
        try detach(schemaName)
    }

    // MARK: - private
    private func _key_v2(db: String,
                         keyPointer: UnsafePointer<UInt8>,
                         keySize: Int,
                         migrate: Bool = false) throws {
        try check(sqlite3_key_v2(handle, db, keyPointer, Int32(keySize)))
        if migrate {
            // Run "PRAGMA cipher_migrate;" immediately after `sqlite3_key_v2`
            // per recommendation of SQLCipher authors
            let migrateResult = try scalar("PRAGMA cipher_migrate;")
            if (migrateResult as? String) != "0" {
                // "0" is the result of successful migration
                throw Result.error(message: "Error in cipher migration, result \(migrateResult.debugDescription)", code: 1, statement: nil)
            }
        }
        try cipher_key_check()
    }

    private func _rekey_v2(db: String, keyPointer: UnsafePointer<UInt8>, keySize: Int) throws {
        try check(sqlite3_rekey_v2(handle, db, keyPointer, Int32(keySize)))
    }

    // When opening an existing database, sqlite3_key_v2 will not immediately throw an error if
    // the key provided is incorrect. To test that the database can be successfully opened with the
    // provided key, it is necessary to perform some operation on the database (i.e. read from it).
    private func cipher_key_check() throws {
        _ = try scalar("SELECT count(*) FROM sqlite_master;")
    }
}
#endif
