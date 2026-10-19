//
//  Config.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.

import Foundation

// MARK: - Configuration files with dynamic lookup

@dynamicMemberLookup
public struct Config: Hashable, Sendable,
    ExpressibleByStringLiteral,
    ExpressibleByIntegerLiteral,
    ExpressibleByBooleanLiteral,
    ExpressibleByFloatLiteral,
    ExpressibleByDictionaryLiteral,
    ExpressibleByArrayLiteral,
    ExpressibleByExtendedGraphemeClusterLiteral,
    CustomStringConvertible
{
    public typealias Key = BinaryDistinctString
    public typealias Value = Config

    private let value: Data

    public enum Data: Sendable {
        case null
        case string(BinaryDistinctString)
        case integer(Int)
        case boolean(Bool)
        case floating(Float)
        case dictionary([BinaryDistinctString: Config])
        case array([Config])
        case token((UInt, BinaryDistinctString))

        public static func == (lhs: Data, rhs: Data) -> Bool {
            switch (lhs, rhs) {
            case (.null, .null):
                return true
            case let (.string(lhs), _):
                if let rhs = rhs.string() {
                    return lhs == BinaryDistinctString(rhs)
                }
            case let (.integer(lhs), _):
                if let rhs = rhs.integer() {
                    return lhs == rhs
                }
            case let (.boolean(lhs), _):
                if let rhs = rhs.boolean() {
                    return lhs == rhs
                }
            case let (.floating(lhs), _):
                if let rhs = rhs.floating() {
                    return lhs == rhs
                }
            case let (.dictionary(lhs), .dictionary(rhs)):
                return lhs == rhs
            case let (.array(lhs), .array(rhs)):
                return lhs == rhs
            case let (.token(lhs), .token(rhs)):
                return lhs == rhs
            default:
                return false
            }

            // right hand side might be a super set of left hand side
            switch rhs {
            case let .string(rhs):
                if let lhs = lhs.string() {
                    return BinaryDistinctString(lhs) == rhs
                }
            case let .integer(rhs):
                if let lhs = lhs.integer() {
                    return lhs == rhs
                }
            case let .boolean(rhs):
                if let lhs = lhs.boolean() {
                    return lhs == rhs
                }
            case let .floating(rhs):
                if let lhs = lhs.floating() {
                    return lhs == rhs
                }
            default:
                return false
            }

            return false
        }

        public var description: String {
            switch self {
            case .null:
                "null"
            case let .string(value):
                "\"\(value)\""
            case let .integer(value):
                "\(value)"
            case let .boolean(value):
                "\(value)"
            case let .floating(value):
                "\(value)"
            case let .array(arr):
                "[\(arr)]"
            case let .dictionary(val):
                "{\(val)}"
            case let .token(val):
                "(\(val.0), \(val.1))"
            }
        }

        public func string() -> String? {
            if case let .string(val) = self {
                return val.string
            }
            return nil
        }

        public func boolean() -> Bool? {
            if case let .boolean(val) = self {
                return val
            }
            if case let .integer(val) = self {
                return val == 1
            }
            if case let .string(val) = self {
                switch val.string.lowercased() {
                case "true", "t", "1":
                    return true
                case "false", "f", "0":
                    return false
                default:
                    return nil
                }
            }
            return nil
        }

        public func integer() -> Int? {
            if case let .integer(val) = self {
                return val
            }
            return nil
        }

        public func floating() -> Float? {
            if case let .floating(val) = self {
                return val
            }
            if case let .integer(val) = self {
                return Float(val)
            }
            return nil
        }
    }

    init() {
        self.value = .null
    }

    public init(_ value: BinaryDistinctString) {
        self.value = .string(value)
    }

    public init(_ value: String) {
        self.init(stringLiteral: value)
    }

    public init(_ value: Int) {
        self.init(integerLiteral: value)
    }

    public init(_ value: Bool) {
        self.init(booleanLiteral: value)
    }

    public init(_ value: Float) {
        self.init(floatLiteral: value)
    }

    public init(_ value: [Config]) {
        self.value = .array(value)
    }

    public init(_ values: (BinaryDistinctString, Config)...) {
        var dict = [BinaryDistinctString: Config]()
        for (key, value) in values {
            dict[key] = value
        }
        self.value = .dictionary(dict)
    }

    public init(_ value: [BinaryDistinctString: Config]) {
        self.value = .dictionary(value)
    }

    public init(_ dictionary: [NSString: Any]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    public init(_ dictionary: [String: Config]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    public init(_ dictionary: [NSString: Config]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    public init(_ token: (UInt, BinaryDistinctString)) {
        self.value = .token(token)
    }

    private static func convertToBinaryDistinctKeys(_ object: Any) -> Config {
        if let dict = object as? [NSString: Any] {
            Config(Dictionary(uniqueKeysWithValues: dict.map { (BinaryDistinctString($0.key), convertToBinaryDistinctKeys($0.value)) }))
        } else if let array = object as? [Any] {
            Config(array.map { convertToBinaryDistinctKeys($0) })
        } else {
            switch object {
            case let obj as String:
                Config(obj)
            case let obj as Int:
                Config(obj)
            case let obj as Float:
                Config(obj)
            case let obj as Bool:
                Config(obj)
            case let obj as NSNumber:
                if CFNumberIsFloatType(obj) {
                    Config(obj.floatValue)
                } else {
                    Config(obj.intValue)
                }
            case _ as NSNull:
                Config()
            case let obj as Config:
                obj
            case let obj as (UInt, String):
                Config((obj.0, BinaryDistinctString(obj.1)))
            default:
                fatalError("unknown type: \(type(of: object)) \(object)")
            }
        }
    }

    // MARK: constructors

    /// Conformance to ExpressibleByStringLiteral
    public init(stringLiteral value: String) {
        self.value = .string(.init(value))
    }

    /// Conformance to ExpressibleByIntegerLiteral
    public init(integerLiteral value: Int) {
        self.value = .integer(value)
    }

    /// Conformance to ExpressibleByBooleanLiteral
    public init(booleanLiteral value: Bool) {
        self.value = .boolean(value)
    }

    /// Conformance to ExpressibleByFloatLiteral
    public init(floatLiteral value: Float) {
        self.value = .floating(value)
    }

    public init(dictionaryLiteral elements: (BinaryDistinctString, Config)...) {
        let dict = elements.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[element.0] = element.1
        }

        self.value = .dictionary(dict)
    }

    public init(arrayLiteral elements: Config...) {
        self.value = .array(elements)
    }

    public func isNull() -> Bool {
        if case .null = self.value {
            return true
        }
        return false
    }

    // MARK: getters - string

    public func get() -> String? {
        self.string()
    }

    public func get(or: String) -> String? {
        self.string(or: or)
    }

    public func string() -> String? {
        self.value.string()
    }

    public func string(or: String) -> String {
        if let val: String = self.string() {
            return val
        }
        return or
    }

    public func get() -> BinaryDistinctString? {
        self.binaryDistinctString()
    }

    public func get(or: BinaryDistinctString) -> BinaryDistinctString? {
        self.binaryDistinctString(or: or)
    }

    public func binaryDistinctString() -> BinaryDistinctString? {
        if case let .string(val) = self.value {
            return val
        }
        return nil
    }

    public func binaryDistinctString(or: BinaryDistinctString) -> BinaryDistinctString {
        if let val: BinaryDistinctString = self.binaryDistinctString() {
            return val
        }
        return or
    }

    // MARK: getters - boolean

    public func get() -> Bool? {
        self.boolean()
    }

    public func get(or: Bool) -> Bool? {
        self.boolean(or: or)
    }

    public func boolean() -> Bool? {
        self.value.boolean()
    }

    public func boolean(or: Bool) -> Bool {
        if let val = self.boolean() {
            return val
        }
        return or
    }

    // MARK: getters - integer

    public func get() -> Int? {
        self.integer()
    }

    public func get(or: Int) -> Int? {
        self.integer(or: or)
    }

    public func integer() -> Int? {
        self.value.integer()
    }

    public func integer(or: Int) -> Int {
        if let val = self.integer() {
            return val
        }
        return or
    }

    // MARK: getters/operators - floating

    public func get() -> Float? {
        self.value.floating()
    }

    public func get(or: Float) -> Float? {
        self.floating(or: or)
    }

    public func floating() -> Float? {
        self.value.floating()
    }

    public func floating(or: Float) -> Float {
        if let val = self.value.floating() {
            return val
        }
        return or
    }

    // MARK: getters - dictionary

    public func get() -> [BinaryDistinctString: Int]? {
        if let dict = self.dictionary() {
            return dict.reduce(into: [:]) { result, element in
                if let val = element.value.value.integer() {
                    result[element.key] = val
                }
            }
        }

        return nil
    }

    public func get() -> [BinaryDistinctString: Config]? {
        self.dictionary()
    }

    public func get(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] {
        self.dictionary(or: or)
    }

    public func toJinjaCompatible() -> Any? {
        switch self.value {
        case let .array(val):
            return val.map { $0.toJinjaCompatible() }
        case let .dictionary(val):
            var result: [String: Any?] = [:]
            for (key, config) in val {
                result[key.string] = config.toJinjaCompatible()
            }
            return result
        case let .boolean(val):
            return val
        case let .floating(val):
            return val
        case let .integer(val):
            return val
        case let .string(val):
            return val.string
        case let .token(val):
            return [String(val.0): val.1.string] as [String: String]
        case .null:
            return nil
        }
    }

    public func dictionary() -> [BinaryDistinctString: Config]? {
        if case let .dictionary(val) = self.value {
            return val
        }
        return nil
    }

    public func dictionary(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] {
        if let val = self.dictionary() {
            return val
        }
        return or
    }

    // MARK: getters - array

    public func get() -> [String]? {
        if let arr = self.array() {
            return arr.reduce(into: []) { result, element in
                if let val: String = element.value.string() {
                    result.append(val)
                }
            }
        }

        return nil
    }

    public func get(or: [String]) -> [String] {
        if let arr: [String] = self.get() {
            return arr
        }

        return or
    }

    public func get() -> [BinaryDistinctString]? {
        if let arr = self.array() {
            return arr.reduce(into: []) { result, element in
                if let val: BinaryDistinctString = element.binaryDistinctString() {
                    result.append(val)
                }
            }
        }

        return nil
    }

    public func get(or: [BinaryDistinctString]) -> [BinaryDistinctString] {
        if let arr: [BinaryDistinctString] = self.get() {
            return arr
        }

        return or
    }

    public func get() -> [Config]? {
        self.array()
    }

    public func get(or: [Config]) -> [Config] {
        self.array(or: or)
    }

    public func array() -> [Config]? {
        if case let .array(val) = self.value {
            return val
        }
        return nil
    }

    public func array(or: [Config]) -> [Config] {
        if let val = self.array() {
            return val
        }
        return or
    }

    // MARK: getters - token

    public func get() -> (UInt, String)? {
        self.token()
    }

    public func get(or: (UInt, String)) -> (UInt, String) {
        self.token(or: or)
    }

    public func token() -> (UInt, String)? {
        if case let .token(val) = self.value {
            return (val.0, val.1.string)
        }

        if case let .array(arr) = self.value {
            guard arr.count == 2 else {
                return nil
            }
            guard let token = arr[0].string() else {
                return nil
            }
            guard let id = arr[1].integer() else {
                return nil
            }

            return (UInt(id), token)
        }

        return nil
    }

    public func token(or: (UInt, String)) -> (UInt, String) {
        if let val = self.token() {
            return val
        }
        return or
    }

    // MARK: subscript

    public subscript(index: BinaryDistinctString) -> Config {
        if let dict = self.dictionary() {
            return dict[index] ?? dict[self.uncamelCase(index)] ?? Config()
        }

        return Config()
    }

    public subscript(index: Int) -> Config {
        if let arr = self.array(), index >= 0, index < arr.count {
            return arr[index]
        }

        return Config()
    }

    public subscript(dynamicMember member: String) -> Config? {
        if let dict = self.dictionary() {
            return dict[BinaryDistinctString(member)] ?? dict[self.uncamelCase(BinaryDistinctString(member))] ?? Config()
        }

        return nil // backward compatibility
    }

    public subscript(dynamicMember member: String) -> Config {
        if let dict = self.dictionary() {
            return dict[BinaryDistinctString(member)] ?? dict[self.uncamelCase(BinaryDistinctString(member))] ?? Config()
        }

        return Config()
    }

    func uncamelCase(_ string: BinaryDistinctString) -> BinaryDistinctString {
        let scalars = string.string.unicodeScalars
        var result = ""

        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }

        return BinaryDistinctString(result)
    }

    public var description: String {
        "\(self.value.description)"
    }
}

/// Old style, deprecated getters
public extension Config {
    @available(*, deprecated, message: "Use string() instead")
    var stringValue: String? { string() }

    @available(*, deprecated, message: "Use integer() instead")
    var intValue: Int? { integer() }

    @available(*, deprecated, message: "Use boolean() instead")
    var boolValue: Bool? { boolean() }

    @available(*, deprecated, message: "Use array() instead")
    var arrayValue: [Config]? { array() }

    @available(*, deprecated, message: "Use token() instead")
    var tokenValue: (UInt, String)? { token() }
}

extension Config: Codable {
    public init(from decoder: any Decoder) throws {
        // Try decoding as a single value first (for scalars and null)
        let singleValueContainer = try? decoder.singleValueContainer()
        if let container = singleValueContainer {
            if container.decodeNil() {
                self.value = .null
                return
            }
            do {
                let intValue = try container.decode(Int.self)
                self.value = .integer(intValue)
                return
            } catch { }
            do {
                let floatValue = try container.decode(Float.self)
                self.value = .floating(floatValue)
                return
            } catch { }
            do {
                let boolValue = try container.decode(Bool.self)
                self.value = .boolean(boolValue)
                return
            } catch { }
            do {
                let stringValue = try container.decode(String.self)
                self.value = .string(.init(stringValue))
                return
            } catch { }
        }

        if let tupple = Self.decodeTuple(decoder) {
            self.value = tupple
            return
        }
        if let array = Self.decodeArray(decoder) {
            self.value = array
            return
        }

        if let dict = Self.decodeDictionary(decoder) {
            self.value = dict
            return
        }

        self.value = .null
    }

    private static func decodeTuple(_ decoder: Decoder) -> Data? {
        let unkeyedContainer = try? decoder.unkeyedContainer()
        if var container = unkeyedContainer {
            if container.count == 2 {
                do {
                    let intValue = try container.decode(UInt.self)
                    let stringValue = try container.decode(String.self)
                    return .token((intValue, .init(stringValue)))
                } catch { }
            }
        }
        return nil
    }

    private static func decodeArray(_ decoder: Decoder) -> Data? {
        do {
            if var container = try? decoder.unkeyedContainer() {
                var elements: [Config] = []
                while !container.isAtEnd {
                    let element = try container.decode(Config.self)
                    elements.append(element)
                }
                return .array(elements)
            }
        } catch { }
        return nil
    }

    private static func decodeDictionary(_ decoder: Decoder) -> Data? {
        do {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            var dictionaryValues: [BinaryDistinctString: Config] = [:]
            for key in container.allKeys {
                let value = try container.decode(Config.self, forKey: key)
                dictionaryValues[BinaryDistinctString(key.stringValue)] = value
            }

            return .dictionary(dictionaryValues)
        } catch {
            return nil
        }
    }

    public func encode(to encoder: any Encoder) throws {
        switch self.value {
        case .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        case let .integer(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .floating(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .boolean(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .string(val):
            var container = encoder.singleValueContainer()
            try container.encode(val.string)
        case let .dictionary(val):
            var container = encoder.container(keyedBy: CodingKeys.self)
            for (key, value) in val {
                try container.encode(value, forKey: CodingKeys(stringValue: key.string)!)
            }
        case let .array(val):
            var container = encoder.unkeyedContainer()
            try container.encode(contentsOf: val)
        case let .token(val):
            var tupple = encoder.unkeyedContainer()
            try tupple.encode(val.0)
            try tupple.encode(val.1.string)
        }
    }

    private struct CodingKeys: CodingKey {
        var stringValue: String
        init?(stringValue: String) {
            self.stringValue = stringValue
        }

        var intValue: Int? { nil }
        init?(intValue: Int) { nil }
    }
}

extension Config: Equatable {
    public static func == (lhs: Config, rhs: Config) -> Bool {
        lhs.value == rhs.value
    }
}

extension Config.Data: Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .null:
            hasher.combine(0) // Discriminator for null
        case let .string(s):
            hasher.combine(1) // Discriminator for string
            hasher.combine(s)
        case let .integer(i):
            hasher.combine(2) // Discriminator for integer
            hasher.combine(i)
        case let .boolean(b):
            hasher.combine(3) // Discriminator for boolean
            hasher.combine(b)
        case let .floating(f):
            hasher.combine(4) // Discriminator for floating
            hasher.combine(f)
        case let .dictionary(d):
            hasher.combine(5) // Discriminator for dict
            d.hash(into: &hasher)
        case let .array(a):
            hasher.combine(6) // Discriminator for array
            for e in a {
                e.hash(into: &hasher)
            }
        case let .token(a):
            hasher.combine(7) // Discriminator for token
            a.0.hash(into: &hasher)
            a.1.hash(into: &hasher)
        }
    }
}

public enum ConfigError: Error {
    case typeMismatch(expected: Config.Data, actual: Config.Data)
    case typeConversionFailed(value: Sendable, targetType: Sendable.Type)
}
