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

extension QueryType {
    /// Creates an `INSERT` statement by encoding the given object
    /// This method converts any custom nested types to JSON data and does not handle any sort
    /// of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Encodable implementations that encode the correct ids.
    ///
    /// - Parameters:
    ///
    ///   - encodable: An encodable object to insert
    ///
    ///   - userInfo: User info to be passed to encoder
    ///
    ///   - otherSetters: Any other setters to include in the insert
    ///
    /// - Returns: An `INSERT` statement for the encodable object
    public func insert(_ encodable: Encodable, userInfo: [CodingUserInfoKey: Any] = [:], otherSetters: [Setter] = []) throws -> Insert {
        let encoder = SQLiteEncoder(userInfo: userInfo)
        try encodable.encode(to: encoder)
        return self.insert(encoder.setters + otherSetters)
    }

    /// Creates an `INSERT` statement by encoding the given object
    /// This method converts any custom nested types to JSON data and does not handle any sort
    /// of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Encodable implementations that encode the correct ids.
    /// The onConflict will be passed to the actual insert function to define what should happen
    /// when an error occurs during the insert operation.
    ///
    /// - Parameters:
    ///
    ///   - onConlict: Define what happens when an insert operation fails
    ///
    ///   - encodable: An encodable object to insert
    ///
    ///   - userInfo: User info to be passed to encoder
    ///
    ///   - otherSetters: Any other setters to include in the insert
    ///
    /// - Returns: An `INSERT` statement fort the encodable object
    public func insert(or onConflict: OnConflict, encodable: Encodable, userInfo: [CodingUserInfoKey: Any] = [:],
                       otherSetters: [Setter] = []) throws -> Insert {
        let encoder = SQLiteEncoder(userInfo: userInfo)
        try encodable.encode(to: encoder)
        return self.insert(or: onConflict, encoder.setters + otherSetters)
    }

    /// Creates a batch `INSERT` statement by encoding the array of given objects
    /// This method converts any custom nested types to JSON data and does not handle any sort
    /// of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Encodable implementations that encode the correct ids.
    ///
    /// - Parameters:
    ///
    ///   - encodables: Encodable objects to insert
    ///
    ///   - userInfo: User info to be passed to encoder
    ///
    ///   - otherSetters: Any other setters to include in the inserts, per row/object.
    ///
    /// - Returns: An `INSERT` statement for the encodable objects
    public func insertMany(_ encodables: [Encodable], userInfo: [CodingUserInfoKey: Any] = [:],
                           otherSetters: [Setter] = []) throws -> Insert {
        let combinedSettersWithoutNils = try encodables.map { encodable -> [Setter] in
            let encoder = SQLiteEncoder(userInfo: userInfo, forcingNilValueSetters: false)
            try encodable.encode(to: encoder)
            return encoder.setters + otherSetters
        }
        // requires the same number of setters per encodable
        guard Set(combinedSettersWithoutNils.map(\.count)).count == 1 else {
            // asymmetric sets of value insertions (some nil, some not), requires NULL value to satisfy INSERT query
            let combinedSymmetricSetters = try encodables.map { encodable -> [Setter] in
                let encoder = SQLiteEncoder(userInfo: userInfo, forcingNilValueSetters: true)
                try encodable.encode(to: encoder)
                return encoder.setters + otherSetters
            }
            return self.insertMany(combinedSymmetricSetters)
        }
        return self.insertMany(combinedSettersWithoutNils)
    }

    /// Creates an `INSERT ON CONFLICT DO UPDATE` statement, aka upsert, by encoding the given object
    /// This method converts any custom nested types to JSON data and does not handle any sort
    /// of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Encodable implementations that encode the correct ids.
    ///
    /// - Parameters:
    ///
    ///   - encodable: An encodable object to insert
    ///
    ///   - userInfo: User info to be passed to encoder
    ///
    ///   - otherSetters: Any other setters to include in the insert
    ///
    ///   - onConflictOf: The column that if conflicts should trigger an update instead of insert.
    ///
    /// - Returns: An `INSERT` statement fort the encodable object
    public func upsert(_ encodable: Encodable, userInfo: [CodingUserInfoKey: Any] = [:],
                       otherSetters: [Setter] = [], onConflictOf conflicting: Expressible) throws -> Insert {
        let encoder = SQLiteEncoder(userInfo: userInfo)
        try encodable.encode(to: encoder)
        return self.upsert(encoder.setters + otherSetters, onConflictOf: conflicting)
    }

    /// Creates an `UPDATE` statement by encoding the given object
    /// This method converts any custom nested types to JSON data and does not handle any sort
    /// of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Encodable implementations that encode the correct ids.
    ///
    /// - Parameters:
    ///
    ///   - encodable: An encodable object to insert
    ///
    ///   - userInfo: User info to be passed to encoder
    ///
    ///   - otherSetters: Any other setters to include in the insert
    ///
    /// - Returns: An `UPDATE` statement fort the encodable object
    public func update(_ encodable: Encodable, userInfo: [CodingUserInfoKey: Any] = [:],
                       otherSetters: [Setter] = []) throws -> Update {
        let encoder = SQLiteEncoder(userInfo: userInfo)
        try encodable.encode(to: encoder)
        return self.update(encoder.setters + otherSetters)
    }
}

extension Row {
    /// Decode an object from this row
    /// This method expects any custom nested types to be in the form of JSON data and does not handle
    /// any sort of object relationships. If you want to support relationships between objects you will
    /// have to provide your own Decodable implementations that decodes the correct columns.
    ///
    /// - Parameter: userInfo
    ///
    /// - Returns: a decoded object from this row
    public func decode<V: Decodable>(userInfo: [CodingUserInfoKey: Any] = [:]) throws -> V {
        try V(from: decoder(userInfo: userInfo))
    }

    public func decoder(userInfo: [CodingUserInfoKey: Any] = [:]) -> Decoder {
        SQLiteDecoder(row: self, userInfo: userInfo)
    }
}

/// Generates a list of settings for an Encodable object
private class SQLiteEncoder: Encoder {
    class SQLiteKeyedEncodingContainer<MyKey: CodingKey>: KeyedEncodingContainerProtocol {
        // swiftlint:disable nesting
        typealias Key = MyKey

        let encoder: SQLiteEncoder
        let codingPath: [CodingKey] = []
        let forcingNilValueSetters: Bool

        init(encoder: SQLiteEncoder, forcingNilValueSetters: Bool = false) {
            self.encoder = encoder
            self.forcingNilValueSetters = forcingNilValueSetters
        }

        func superEncoder() -> Swift.Encoder {
            fatalError("SQLiteEncoding does not support super encoders")
        }

        func superEncoder(forKey key: Key) -> Swift.Encoder {
            fatalError("SQLiteEncoding does not support super encoders")
        }

        func encodeNil(forKey key: SQLiteEncoder.SQLiteKeyedEncodingContainer<Key>.Key) throws {
            encoder.setters.append(Expression<String?>(key.stringValue) <- nil)
        }

        func encode(_ value: Int, forKey key: SQLiteEncoder.SQLiteKeyedEncodingContainer<Key>.Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- value)
        }

        func encode(_ value: Bool, forKey key: Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- value)
        }

        func encode(_ value: Float, forKey key: Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- Double(value))
        }

        func encode(_ value: Double, forKey key: Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- value)
        }

        func encode(_ value: String, forKey key: Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- value)
        }

        func encodeIfPresent(_ value: Int?, forKey key: SQLiteEncoder.SQLiteKeyedEncodingContainer<Key>.Key) throws {
            if let value {
                try encode(value, forKey: key)
            } else if forcingNilValueSetters {
                encoder.setters.append(Expression<Int?>(key.stringValue) <- nil)
            }
        }

        func encodeIfPresent(_ value: Bool?, forKey key: Key) throws {
            if let value {
                try encode(value, forKey: key)
            } else if forcingNilValueSetters {
                encoder.setters.append(Expression<Bool?>(key.stringValue) <- nil)
            }
        }

        func encodeIfPresent(_ value: Float?, forKey key: Key) throws {
            if let value {
                try encode(value, forKey: key)
            } else if forcingNilValueSetters {
                encoder.setters.append(Expression<Double?>(key.stringValue) <- nil)
            }
        }

        func encodeIfPresent(_ value: Double?, forKey key: Key) throws {
            if let value {
                try encode(value, forKey: key)
            } else if forcingNilValueSetters {
                encoder.setters.append(Expression<Double?>(key.stringValue) <- nil)
            }
        }

        func encodeIfPresent(_ value: String?, forKey key: MyKey) throws {
            if let value {
                try encode(value, forKey: key)
            } else if forcingNilValueSetters {
                encoder.setters.append(Expression<String?>(key.stringValue) <- nil)
            }
        }

        func encode<T>(_ value: T, forKey key: Key) throws where T: Swift.Encodable {
            switch value {
            case let data as Data:
                encoder.setters.append(Expression(key.stringValue) <- data)
            case let date as Date:
                encoder.setters.append(Expression(key.stringValue) <- date.datatypeValue)
            case let uuid as UUID:
                encoder.setters.append(Expression(key.stringValue) <- uuid.datatypeValue)
            default:
                let encoded = try JSONEncoder().encode(value)
                let string = String(data: encoded, encoding: .utf8)
                encoder.setters.append(Expression(key.stringValue) <- string)
            }
        }

        func encodeIfPresent<T>(_ value: T?, forKey key: Key) throws where T: Swift.Encodable {
            guard let value else {
                guard forcingNilValueSetters else {
                    return
                }
                encoder.setters.append(Expression<String?>(key.stringValue) <- nil)
                return
            }
            try encode(value, forKey: key)
        }

        func encode(_ value: Int8, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an Int8 is not supported"))
        }

        func encode(_ value: Int16, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an Int16 is not supported"))
        }

        func encode(_ value: Int32, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an Int32 is not supported"))
        }

        func encode(_ value: Int64, forKey key: Key) throws {
            encoder.setters.append(Expression(key.stringValue) <- value)
        }

        func encode(_ value: UInt, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an UInt is not supported"))
        }

        func encode(_ value: UInt8, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an UInt8 is not supported"))
        }

        func encode(_ value: UInt16, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an UInt16 is not supported"))
        }

        func encode(_ value: UInt32, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an UInt32 is not supported"))
        }

        func encode(_ value: UInt64, forKey key: Key) throws {
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: codingPath,
                                                                          debugDescription: "encoding an UInt64 is not supported"))
        }

        func nestedContainer<NestedKey>(keyedBy keyType: NestedKey.Type, forKey key: Key)
        -> KeyedEncodingContainer<NestedKey> where NestedKey: CodingKey {
            fatalError("encoding a nested container is not supported")
        }

        func nestedUnkeyedContainer(forKey key: Key) -> UnkeyedEncodingContainer {
            fatalError("encoding nested values is not supported")
        }
    }

    fileprivate var setters: [Setter] = []
    let codingPath: [CodingKey] = []
    let userInfo: [CodingUserInfoKey: Any]
    let forcingNilValueSetters: Bool

    init(userInfo: [CodingUserInfoKey: Any], forcingNilValueSetters: Bool = false) {
        self.userInfo = userInfo
        self.forcingNilValueSetters = forcingNilValueSetters
    }

    func singleValueContainer() -> SingleValueEncodingContainer {
        fatalError("not supported")
    }

    func unkeyedContainer() -> UnkeyedEncodingContainer {
        fatalError("not supported")
    }

    func container<Key>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> where Key: CodingKey {
        KeyedEncodingContainer(SQLiteKeyedEncodingContainer(encoder: self, forcingNilValueSetters: forcingNilValueSetters))
    }
}

private class SQLiteDecoder: Decoder {
    class SQLiteKeyedDecodingContainer<MyKey: CodingKey>: KeyedDecodingContainerProtocol {
        typealias Key = MyKey

        let codingPath: [CodingKey] = []
        let row: Row

        init(row: Row) {
            self.row = row
        }

        var allKeys: [Key] {
            row.columnNames.keys.compactMap({ Key(stringValue: $0) })
        }

        func contains(_ key: Key) -> Bool {
            row.hasValue(for: key.stringValue)
        }

        func decodeNil(forKey key: Key) throws -> Bool {
            !contains(key)
        }

        func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool {
            try row.get(Expression(key.stringValue))
        }

        func decode(_ type: Int.Type, forKey key: Key) throws -> Int {
            try row.get(Expression(key.stringValue))
        }

        func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an Int8 is not supported"))
        }

        func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an Int16 is not supported"))
        }

        func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an Int32 is not supported"))
        }

        func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 {
            try row.get(Expression(key.stringValue))
        }

        func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an UInt is not supported"))

        }

        func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an UInt8 is not supported"))
        }

        func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an UInt16 is not supported"))
        }

        func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an UInt32 is not supported"))
        }

        func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 {
            throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                         debugDescription: "decoding an UInt64 is not supported"))
        }

        func decode(_ type: Float.Type, forKey key: Key) throws -> Float {
            Float(try row.get(Expression<Double>(key.stringValue)))
        }

        func decode(_ type: Double.Type, forKey key: Key) throws -> Double {
            try row.get(Expression(key.stringValue))
        }

        func decode(_ type: String.Type, forKey key: Key) throws -> String {
            try row.get(Expression(key.stringValue))
        }

        func decode<T>(_ type: T.Type, forKey key: Key) throws -> T where T: Swift.Decodable {
            // swiftlint:disable force_cast
            switch type {
            case is Data.Type:
                let data = try row.get(Expression<Data>(key.stringValue))
                return data as! T
            case is Date.Type:
                let date = try row.get(Expression<Date>(key.stringValue))
                return date as! T
            case is UUID.Type:
                let uuid = try row.get(Expression<UUID>(key.stringValue))
                return uuid as! T
            default:
                // swiftlint:enable force_cast
                guard let JSONString = try row.get(Expression<String?>(key.stringValue)) else {
                    throw DecodingError.typeMismatch(type, DecodingError.Context(codingPath: codingPath,
                                                                                 debugDescription: "an unsupported type was found"))
                }
                guard let data = JSONString.data(using: .utf8) else {
                    throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                            debugDescription: "invalid utf8 data found"))
                }
                return try JSONDecoder().decode(type, from: data)
            }
        }

        func decodeIfPresent(_ type: Bool.Type, forKey key: Key) throws -> Bool? {
            try? row.get(Expression(key.stringValue))
        }

        func decodeIfPresent(_ type: Int.Type, forKey key: Key) throws -> Int? {
            try? row.get(Expression(key.stringValue))
        }

        func decodeIfPresent(_ type: Int64.Type, forKey key: Key) throws -> Int64? {
            try? row.get(Expression(key.stringValue))
        }

        func decodeIfPresent(_ type: Float.Type, forKey key: Key) throws -> Float? {
            try? Float(row.get(Expression<Double>(key.stringValue)))
        }

        func decodeIfPresent(_ type: Double.Type, forKey key: Key) throws -> Double? {
            try? row.get(Expression(key.stringValue))
        }

        func decodeIfPresent(_ type: String.Type, forKey key: Key) throws -> String? {
            try? row.get(Expression(key.stringValue))
        }

        func decodeIfPresent<T>(_ type: T.Type, forKey key: Key) throws -> T? where T: Swift.Decodable {
            switch type {
            case is Data.Type:
                return try? row.get(Expression<Data>(key.stringValue)) as? T
            case is Date.Type:
                return try? row.get(Expression<Date>(key.stringValue)) as? T
            case is UUID.Type:
                return try? row.get(Expression<UUID>(key.stringValue)) as? T
            default:
                guard let JSONString = try row.get(Expression<String?>(key.stringValue)) else {
                    return nil
                }
                guard let data = JSONString.data(using: .utf8) else {
                    throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                            debugDescription: "invalid utf8 data found"))
                }
                return try JSONDecoder().decode(type, from: data)
            }
        }

        func nestedContainer<NestedKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws
        -> KeyedDecodingContainer<NestedKey> where NestedKey: CodingKey {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                    debugDescription: "decoding nested containers is not supported"))
        }

        func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                    debugDescription: "decoding unkeyed containers is not supported"))
        }

        func superDecoder() throws -> Swift.Decoder {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                    debugDescription: "decoding super encoders containers is not supported"))
        }

        func superDecoder(forKey key: Key) throws -> Swift.Decoder {
            throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                    debugDescription: "decoding super decoders is not supported"))
        }
    }

    let row: Row
    let codingPath: [CodingKey] = []
    let userInfo: [CodingUserInfoKey: Any]

    init(row: Row, userInfo: [CodingUserInfoKey: Any]) {
        self.row = row
        self.userInfo = userInfo
    }

    func container<Key>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> where Key: CodingKey {
        KeyedDecodingContainer(SQLiteKeyedDecodingContainer(row: row))
    }

    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                debugDescription: "decoding an unkeyed container is not supported"))
    }

    func singleValueContainer() throws -> SingleValueDecodingContainer {
        throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: codingPath,
                                                                debugDescription: "decoding a single value container is not supported"))
    }
}
