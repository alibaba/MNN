//
//  BinaryDistinct.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.
//

import Foundation

/// BinaryDistinctString helps to overcome limitations of both String and NSString types. Where the prior is performing unicode normalization and the following is not Sendable. For more reference [Modifying-and-Comparing-Strings](https://developer.apple.com/documentation/swift/string#Modifying-and-Comparing-Strings).
public struct BinaryDistinctString: Equatable, Hashable, Sendable, Comparable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let value: [UInt16]

    public var nsString: NSString {
        String(utf16CodeUnits: value, count: value.count) as NSString
    }

    public var string: String {
        String(nsString)
    }

    public var count: Int {
        string.count
    }

    /// Satisfies ``CustomStringConvertible`` protocol.
    public var description: String {
        string
    }

    public init(_ bytes: [UInt16]) {
        value = bytes
    }

    public init(_ str: NSString) {
        value = Array(str as String).flatMap { $0.utf16 }
    }

    public init(_ str: String) {
        self.init(str as NSString)
    }

    public init(_ character: BinaryDistinctCharacter) {
        value = character.bytes
    }

    public init(_ characters: [BinaryDistinctCharacter]) {
        var data: [UInt16] = []
        for character in characters {
            data.append(contentsOf: character.bytes)
        }
        value = data
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    public static func == (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        lhs.value == rhs.value
    }

    public static func < (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        lhs.value.lexicographicallyPrecedes(rhs.value)
    }

    public static func + (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> BinaryDistinctString {
        BinaryDistinctString(lhs.value + rhs.value)
    }

    public func hasPrefix(_ prefix: BinaryDistinctString) -> Bool {
        guard prefix.value.count <= value.count else { return false }
        return value.starts(with: prefix.value)
    }

    public func hasSuffix(_ suffix: BinaryDistinctString) -> Bool {
        guard suffix.value.count <= value.count else { return false }
        return value.suffix(suffix.value.count) == suffix.value
    }

    public func lowercased() -> BinaryDistinctString {
        .init(string.lowercased())
    }

    public func replacingOccurrences(of: Self, with: Self) -> BinaryDistinctString {
        BinaryDistinctString(string.replacingOccurrences(of: of.string, with: with.string))
    }
}

public extension BinaryDistinctString {
    typealias Index = Int // Treat indices as integers

    var startIndex: Index { 0 }
    var endIndex: Index { count }

    func index(_ i: Index, offsetBy distance: Int) -> Index {
        let newIndex = i + distance
        guard newIndex >= 0, newIndex <= count else {
            fatalError("Index out of bounds")
        }
        return newIndex
    }

    func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
        let newIndex = i + distance
        return newIndex <= limit ? newIndex : nil
    }
}

extension BinaryDistinctString: Sequence {
    public func makeIterator() -> AnyIterator<BinaryDistinctCharacter> {
        var iterator = string.makeIterator() // Use native Swift String iterator

        return AnyIterator {
            guard let char = iterator.next() else { return nil }
            return BinaryDistinctCharacter(char)
        }
    }
}

public extension BinaryDistinctString {
    subscript(bounds: PartialRangeFrom<Int>) -> BinaryDistinctString {
        let validRange = bounds.lowerBound..<value.count // Convert to Range<Int>
        return self[validRange]
    }

    /// Returns a slice of the `BinaryDistinctString` while ensuring correct rune (grapheme cluster) boundaries.
    subscript(bounds: Range<Int>) -> BinaryDistinctString {
        guard bounds.lowerBound >= 0, bounds.upperBound <= count else {
            fatalError("Index out of bounds")
        }

        let utf8Bytes = value
        var byteIndices: [Int] = []

        // Decode UTF-8 manually to find rune start positions
        var currentByteIndex = 0
        for (index, scalar) in string.unicodeScalars.enumerated() {
            if index == bounds.lowerBound {
                byteIndices.append(currentByteIndex)
            }
            currentByteIndex += scalar.utf8.count
            if index == bounds.upperBound - 1 {
                byteIndices.append(currentByteIndex)
                break
            }
        }

        // Extract the byte range
        let startByteIndex = byteIndices.first ?? 0
        let endByteIndex = byteIndices.last ?? utf8Bytes.count

        let slicedBytes = Array(utf8Bytes[startByteIndex..<endByteIndex])
        return BinaryDistinctString(slicedBytes)
    }
}

public extension Dictionary where Key == BinaryDistinctString {
    /// Merges another `BinaryDistinctDictionary` into this one
    mutating func merge(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        merge(other, uniquingKeysWith: strategy)
    }

    /// Merges a `[String: Value]` dictionary into this one
    mutating func merge(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        merge(converted, uniquingKeysWith: strategy)
    }

    /// Merges a `[NSString: Value]` dictionary into this one
    mutating func merge(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) {
        let converted = Dictionary(uniqueKeysWithValues: other.map { (BinaryDistinctString($0.key), $0.value) })
        merge(converted, uniquingKeysWith: strategy)
    }

    func merging(_ other: [String: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    func merging(_ other: [BinaryDistinctString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }

    func merging(_ other: [NSString: Value], strategy: (Value, Value) -> Value = { _, new in new }) -> Self {
        var newDict = self
        newDict.merge(other, strategy: strategy)
        return newDict
    }
}

public protocol StringConvertible: ExpressibleByStringLiteral { }

extension BinaryDistinctString: StringConvertible { }
extension String: StringConvertible { }
extension NSString: StringConvertible { }

public struct BinaryDistinctCharacter: Equatable, Hashable, CustomStringConvertible, ExpressibleByStringLiteral {
    let bytes: [UInt16]

    public init(_ character: Character) {
        bytes = Array(character.utf16)
    }

    public init(_ string: String) {
        bytes = Array(string.utf16)
    }

    public init(_ nsString: NSString) {
        let swiftString = nsString as String
        bytes = Array(swiftString.utf16)
    }

    public init(bytes: [UInt16]) {
        self.bytes = bytes
    }

    /// Satisfies ``ExpressibleByStringLiteral`` protocol.
    public init(stringLiteral value: String) {
        self.init(value)
    }

    var stringValue: String? {
        String(utf16CodeUnits: bytes, count: bytes.count)
    }

    public var description: String {
        if let str = stringValue {
            "BinaryDistinctCharacter('\(str)', bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        } else {
            "BinaryDistinctCharacter(invalid UTF-8, bytes: \(bytes.map { String(format: "0x%02X", $0) }))"
        }
    }

    public static func == (lhs: BinaryDistinctCharacter, rhs: BinaryDistinctCharacter) -> Bool {
        lhs.bytes == rhs.bytes
    }
}
