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

#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

/// A single SQL statement.
public final class Statement {

    fileprivate var handle: OpaquePointer?

    fileprivate let connection: Connection

    init(_ connection: Connection, _ SQL: String) throws {
        self.connection = connection
        try connection.check(sqlite3_prepare_v2(connection.handle, SQL, -1, &handle, nil))
    }

    deinit {
        sqlite3_finalize(handle)
    }

    public lazy var columnCount: Int = Int(sqlite3_column_count(handle))

    public lazy var columnNames: [String] = (0..<Int32(columnCount)).map {
        String(cString: sqlite3_column_name(handle, $0))
    }

    /// A cursor pointing to the current row.
    public lazy var row: Cursor = Cursor(self)

    /// Binds a list of parameters to a statement.
    ///
    /// - Parameter values: A list of parameters to bind to the statement.
    ///
    /// - Returns: The statement object (useful for chaining).
    public func bind(_ values: Binding?...) -> Statement {
        bind(values)
    }

    /// Binds a list of parameters to a statement.
    ///
    /// - Parameter values: A list of parameters to bind to the statement.
    ///
    /// - Returns: The statement object (useful for chaining).
    public func bind(_ values: [Binding?]) -> Statement {
        if values.isEmpty { return self }
        reset()
        guard values.count == Int(sqlite3_bind_parameter_count(handle)) else {
            fatalError("\(sqlite3_bind_parameter_count(handle)) values expected, \(values.count) passed")
        }
        for idx in 1...values.count { bind(values[idx - 1], atIndex: idx) }
        return self
    }

    /// Binds a dictionary of named parameters to a statement.
    ///
    /// - Parameter values: A dictionary of named parameters to bind to the
    ///   statement.
    ///
    /// - Returns: The statement object (useful for chaining).
    public func bind(_ values: [String: Binding?]) -> Statement {
        reset()
        for (name, value) in values {
            let idx = sqlite3_bind_parameter_index(handle, name)
            guard idx > 0 else {
                fatalError("parameter not found: \(name)")
            }
            bind(value, atIndex: Int(idx))
        }
        return self
    }

    fileprivate func bind(_ value: Binding?, atIndex idx: Int) {
        switch value {
        case .none:
            sqlite3_bind_null(handle, Int32(idx))
        case let value as Blob where value.bytes.count == 0:
            sqlite3_bind_zeroblob(handle, Int32(idx), 0)
        case let value as Blob:
            sqlite3_bind_blob(handle, Int32(idx), value.bytes, Int32(value.bytes.count), SQLITE_TRANSIENT)
        case let value as Double:
            sqlite3_bind_double(handle, Int32(idx), value)
        case let value as Int64:
            sqlite3_bind_int64(handle, Int32(idx), value)
        case let value as String:
            sqlite3_bind_text(handle, Int32(idx), value, -1, SQLITE_TRANSIENT)
        case let value as Int:
            self.bind(value.datatypeValue, atIndex: idx)
        case let value as Bool:
            self.bind(value.datatypeValue, atIndex: idx)
        case .some(let value):
            fatalError("tried to bind unexpected value \(value)")
        }
    }

    /// - Parameter bindings: A list of parameters to bind to the statement.
    ///
    /// - Throws: `Result.Error` if query execution fails.
    ///
    /// - Returns: The statement object (useful for chaining).
    @discardableResult public func run(_ bindings: Binding?...) throws -> Statement {
        guard bindings.isEmpty else {
            return try run(bindings)
        }

        reset(clearBindings: false)
        repeat {} while try step()
        return self
    }

    /// - Parameter bindings: A list of parameters to bind to the statement.
    ///
    /// - Throws: `Result.Error` if query execution fails.
    ///
    /// - Returns: The statement object (useful for chaining).
    @discardableResult public func run(_ bindings: [Binding?]) throws -> Statement {
        try bind(bindings).run()
    }

    /// - Parameter bindings: A dictionary of named parameters to bind to the
    ///   statement.
    ///
    /// - Throws: `Result.Error` if query execution fails.
    ///
    /// - Returns: The statement object (useful for chaining).
    @discardableResult public func run(_ bindings: [String: Binding?]) throws -> Statement {
        try bind(bindings).run()
    }

    /// - Parameter bindings: A list of parameters to bind to the statement.
    ///
    /// - Returns: The first value of the first row returned.
    public func scalar(_ bindings: Binding?...) throws -> Binding? {
        guard bindings.isEmpty else {
            return try scalar(bindings)
        }

        reset(clearBindings: false)
        _ = try step()
        return row[0]
    }

    /// - Parameter bindings: A list of parameters to bind to the statement.
    ///
    /// - Returns: The first value of the first row returned.
    public func scalar(_ bindings: [Binding?]) throws -> Binding? {
        try bind(bindings).scalar()
    }

    /// - Parameter bindings: A dictionary of named parameters to bind to the
    ///   statement.
    ///
    /// - Returns: The first value of the first row returned.
    public func scalar(_ bindings: [String: Binding?]) throws -> Binding? {
        try bind(bindings).scalar()
    }

    public func step() throws -> Bool {
        try connection.sync { try connection.check(sqlite3_step(handle)) == SQLITE_ROW }
    }

    public func reset() {
        reset(clearBindings: true)
    }

    fileprivate func reset(clearBindings shouldClear: Bool) {
        sqlite3_reset(handle)
        if shouldClear { sqlite3_clear_bindings(handle) }
    }

}

extension Statement: Sequence {

    public func makeIterator() -> Statement {
        reset(clearBindings: false)
        return self
    }

}

public protocol FailableIterator: IteratorProtocol {
    func failableNext() throws -> Self.Element?
}

extension FailableIterator {
    public func next() -> Element? {
        // swiftlint:disable:next force_try
        try! failableNext()
    }
}

extension Array {
    public init<I: FailableIterator>(_ failableIterator: I) throws where I.Element == Element {
        self.init()
        while let row = try failableIterator.failableNext() {
            append(row)
        }
    }
}

extension Statement: FailableIterator {
    public typealias Element = [Binding?]
    public func failableNext() throws -> [Binding?]? {
        try step() ? Array(row) : nil
    }
}

extension Statement {
    func prepareRowIterator() -> RowIterator {
        RowIterator(statement: self, columnNames: columnNameMap)
    }

    var columnNameMap: [String: Int] {
        var result = [String: Int]()
        for (index, name) in self.columnNames.enumerated() {
            result[name.quote()] = index
        }

        return result
    }
}

extension Statement: CustomStringConvertible {

    public var description: String {
        String(cString: sqlite3_sql(handle))
    }

}

public struct Cursor {

    fileprivate let handle: OpaquePointer

    fileprivate let columnCount: Int

    fileprivate init(_ statement: Statement) {
        handle = statement.handle!
        columnCount = statement.columnCount
    }

    public subscript(idx: Int) -> Double {
        sqlite3_column_double(handle, Int32(idx))
    }

    public subscript(idx: Int) -> Int64 {
        sqlite3_column_int64(handle, Int32(idx))
    }

    public subscript(idx: Int) -> String {
        String(cString: UnsafePointer(sqlite3_column_text(handle, Int32(idx))))
    }

    public subscript(idx: Int) -> Blob {
        if let pointer = sqlite3_column_blob(handle, Int32(idx)) {
            let length = Int(sqlite3_column_bytes(handle, Int32(idx)))
            return Blob(bytes: pointer, length: length)
        } else {
            // The return value from sqlite3_column_blob() for a zero-length BLOB is a NULL pointer.
            // https://www.sqlite.org/c3ref/column_blob.html
            return Blob(bytes: [])
        }
    }

    // MARK: -

    public subscript(idx: Int) -> Bool {
        Bool.fromDatatypeValue(self[idx])
    }

    public subscript(idx: Int) -> Int {
        Int.fromDatatypeValue(self[idx])
    }

}

/// Cursors provide direct access to a statement’s current row.
extension Cursor: Sequence {

    public subscript(idx: Int) -> Binding? {
        switch sqlite3_column_type(handle, Int32(idx)) {
        case SQLITE_BLOB:
            return self[idx] as Blob
        case SQLITE_FLOAT:
            return self[idx] as Double
        case SQLITE_INTEGER:
            return self[idx] as Int64
        case SQLITE_NULL:
            return nil
        case SQLITE_TEXT:
            return self[idx] as String
        case let type:
            fatalError("unsupported column type: \(type)")
        }
    }

    public func makeIterator() -> AnyIterator<Binding?> {
        var idx = 0
        return AnyIterator {
            if idx >= columnCount {
                return .none
            } else {
                idx += 1
                return self[idx - 1]
            }
        }
    }

}
