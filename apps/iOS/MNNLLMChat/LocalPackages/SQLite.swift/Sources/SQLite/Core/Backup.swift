//
// SQLite.swift
// https://github.com/stephencelis/SQLite.swift
// Copyright © 2014-2015 Stephen Celis.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

import Foundation
import Dispatch
#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

/// An object representing database backup.
///
/// See: <https://www.sqlite.org/backup.html>
public final class Backup {

    /// The name of the database to backup
    public enum DatabaseName {

        /// The main database
        case main

        /// The temporary database
        case temp

        /// A database added to the connection with ATTACH statement
        case attached(name: String)

        var name: String {
            switch self {
            case .main:
                return "main"
            case .temp:
                return "temp"
            case .attached(let name):
                return name
            }
        }
    }

    /// Number of pages to copy while performing a backup step
    public enum Pages {

        /// Indicates all remaining pages should be copied
        case all

        /// Indicates the maximal number of pages to be copied in single step
        case limited(number: Int32)

        var number: Int32 {
            switch self {
            case .all:
                return -1
            case .limited(let number):
                return number
            }
        }
    }

    /// Total number of pages to copy
    ///
    /// See: <https://www.sqlite.org/c3ref/backup_finish.html#sqlite3backuppagecount>
    public var pageCount: Int32 {
        return handle.map { sqlite3_backup_pagecount($0) } ?? 0
    }

    /// Number of remaining pages to copy.
    ///
    /// See: <https://www.sqlite.org/c3ref/backup_finish.html#sqlite3backupremaining>
    public var remainingPages: Int32 {
        return handle.map { sqlite3_backup_remaining($0) } ?? 0
    }

    private let targetConnection: Connection
    private let sourceConnection: Connection

    private var handle: OpaquePointer?

    /// Initializes a new SQLite backup.
    ///
    /// - Parameters:
    ///
    ///   - sourceConnection: The connection to the database to backup.
    ///   - sourceName: The name of the database to backup.
    ///     Default: `.main`.
    ///
    ///   - targetConnection: The connection to the database to save backup into.
    ///   - targetName: The name of the database to save backup into.
    ///     Default: `.main`.
    ///
    /// - Returns: A new database backup.
    ///
    /// See: <https://www.sqlite.org/c3ref/backup_finish.html#sqlite3backupinit>
    public init(sourceConnection: Connection,
                sourceName: DatabaseName = .main,
                targetConnection: Connection,
                targetName: DatabaseName = .main) throws {

        self.targetConnection = targetConnection
        self.sourceConnection = sourceConnection

        self.handle = sqlite3_backup_init(targetConnection.handle,
                                          targetName.name,
                                          sourceConnection.handle,
                                          sourceName.name)

        if handle == nil, let error = Result(errorCode: sqlite3_errcode(targetConnection.handle),
                                             connection: targetConnection) {
            throw error
        }
    }

    /// Performs a backup step.
    ///
    /// - Parameter pagesToCopy: The maximal number of pages to copy in one step
    ///
    /// - Throws: `Result.Error` if step fails.
    ///
    /// See: <https://www.sqlite.org/c3ref/backup_finish.html#sqlite3backupstep>
    public func step(pagesToCopy pages: Pages = .all) throws {
        let status = sqlite3_backup_step(handle, pages.number)

        guard status != SQLITE_DONE else {
            finish()
            return
        }

        if let error = Result(errorCode: status, connection: targetConnection) {
            throw error
        }
    }

    /// Finalizes backup.
    ///
    /// See: <https://www.sqlite.org/c3ref/backup_finish.html#sqlite3backupfinish>
    public func finish() {
        guard let handle = self.handle else {
            return
        }

        sqlite3_backup_finish(handle)
        self.handle = nil
    }

    deinit {
        finish()
    }
}
