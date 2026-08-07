//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

public class MinimalDecoder {
  public typealias Value = MinimalEncoder.Value
  typealias _Key = MinimalEncoder._Key

  public private(set) var userInfo: [CodingUserInfoKey: Any]

  public init(userInfo: [CodingUserInfoKey: Any] = [:]) {
    self.userInfo = userInfo
  }

  public func decode<T: Decodable>(_ input: Value, as type: T.Type) throws -> T {
    let decoder = Decoder(base: self, parent: nil, input: input, key: nil)
    defer {
      decoder.finalize()
    }
    return try T(from: decoder)
  }

  public static func decode<T: Decodable>(_ input: Value, as type: T.Type) throws -> T {
    let decoder = MinimalDecoder()
    return try decoder.decode(input, as: T.self)
  }
}

protocol _MinimalDecoderContainer: AnyObject {
  var base: MinimalDecoder { get }
  var parent: _MinimalDecoderContainer? { get }
  var key: CodingKey? { get }
  var isValid: Bool { get }
  func finalize()
}

extension _MinimalDecoderContainer {
  var codingPath: [CodingKey] {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    var path: [CodingKey] = []
    var encoder: _MinimalDecoderContainer? = self
    while let e = encoder {
      if let key = e.key { path.append(key) }
      encoder = e.parent
    }
    path.reverse()
    return path
  }

  var userInfo: [CodingUserInfoKey: Any] {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    return base.userInfo
  }
}

extension DecodingError {
  internal static func _typeMismatch(
    at path: [CodingKey],
    expectation: Any.Type,
    reality: MinimalDecoder.Value
  ) -> DecodingError {
    let description = "Expected to decode \(expectation) but found `\(reality.typeDescription))`."
    return .typeMismatch(expectation, Context(codingPath: path, debugDescription: description))
  }
}

extension MinimalDecoder {
  final class Decoder {
    typealias Value = MinimalEncoder.Value

    unowned let base: MinimalDecoder
    unowned let parent: _MinimalDecoderContainer?
    let key: CodingKey?
    var input: Value
    var isValid = true
    var pendingContainer: _MinimalDecoderContainer?

    init(base: MinimalDecoder, parent: _MinimalDecoderContainer?, input: Value, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
      self.input = input
    }
  }
}

extension MinimalDecoder.Decoder: _MinimalDecoderContainer {
  func finalize() {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    isValid = false
  }
}

extension MinimalDecoder.Decoder: Decoder {
  func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(pendingContainer, "Cannot extract multiple containers", trapping: true)
    guard case let .dictionary(d) = input else {
      throw DecodingError._typeMismatch(at: codingPath, expectation: [String: Value].self, reality: input)
    }
    let keyed = MinimalDecoder.KeyedContainer<Key>(base: base, parent: parent, input: d, key: key)
    pendingContainer = keyed
    return .init(keyed)
  }

  func unkeyedContainer() throws -> UnkeyedDecodingContainer {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(pendingContainer, "Cannot extract multiple containers", trapping: true)
    guard case let .array(a) = input else {
      throw DecodingError._typeMismatch(at: codingPath, expectation: [Value].self, reality: input)
    }
    let unkeyed = MinimalDecoder.UnkeyedContainer(base: base, parent: parent, input: a, key: nil)
    pendingContainer = unkeyed
    return unkeyed
  }

  func singleValueContainer() throws -> SingleValueDecodingContainer {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(pendingContainer, "Cannot extract multiple containers", trapping: true)
    let single = MinimalDecoder.SingleValueContainer(base: base, parent: parent, input: input, key: nil)
    pendingContainer = single
    return single
  }
}

extension MinimalDecoder {
  final class KeyedContainer<Key: CodingKey> {
    typealias Value = MinimalEncoder.Value

    unowned let base: MinimalDecoder
    unowned let parent: _MinimalDecoderContainer?
    let key: CodingKey?
    var input: [String: Value]
    var isValid = true
    var pendingContainer: AnyObject?

    init(base: MinimalDecoder, parent: _MinimalDecoderContainer?, input: [String: Value], key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
      self.input = input
    }

    func commitPendingContainer() {
      expectTrue(self.isValid, "Container isn't valid", trapping: true)
      guard let c = pendingContainer else { return }
      self.pendingContainer = nil
      let container = c as! _MinimalDecoderContainer
      container.finalize()
    }
  }
}

extension MinimalDecoder.KeyedContainer: _MinimalDecoderContainer {
  func finalize() {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    isValid = false
  }
}

extension MinimalDecoder.KeyedContainer: KeyedDecodingContainerProtocol {
  var allKeys: [Key] {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    return input.keys.compactMap { Key(stringValue: $0) }
  }

  func contains(_ key: Key) -> Bool {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    return input[key.stringValue] != nil
  }

  func _decode<K: CodingKey>(key: K) throws -> Value {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    guard let value = input[key.stringValue] else {
      let context = DecodingError.Context(
        codingPath: codingPath + [key],
        debugDescription: "Key not found (\"\(key.stringValue)\")")
      throw DecodingError.keyNotFound(key, context)
    }
    return value
  }

  func _decode<T>(_ type: T.Type, forKey key: Key, _ extract: (Value) -> T?) throws -> T {
    let value = try _decode(key: key)
    guard let v = extract(value) else {
      throw DecodingError._typeMismatch(at: codingPath + [key], expectation: T.self, reality: value)
    }
    return v
  }

  func decodeNil(forKey key: Key) throws -> Bool {
    let value = try _decode(key: key)
    return value == .null
  }

  func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool {
    try _decode(type, forKey: key) { input in
      guard case let .bool(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Int.Type, forKey key: Key) throws -> Int {
    try _decode(type, forKey: key) { input in
      guard case let .int(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 {
    try _decode(type, forKey: key) { input in
      guard case let .int8(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 {
    try _decode(type, forKey: key) { input in
      guard case let .int16(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 {
    try _decode(type, forKey: key) { input in
      guard case let .int32(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 {
    try _decode(type, forKey: key) { input in
      guard case let .int64(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt {
    try _decode(type, forKey: key) { input in
      guard case let .uint(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 {
    try _decode(type, forKey: key) { input in
      guard case let .uint8(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 {
    try _decode(type, forKey: key) { input in
      guard case let .uint16(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 {
    try _decode(type, forKey: key) { input in
      guard case let .uint32(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 {
    try _decode(type, forKey: key) { input in
      guard case let .uint64(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: String.Type, forKey key: Key) throws -> String {
    try _decode(type, forKey: key) { input in
      guard case let .string(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Double.Type, forKey key: Key) throws -> Double {
    try _decode(type, forKey: key) { input in
      guard case let .double(v) = input else { return nil }
      return v
    }
  }

  func decode(_ type: Float.Type, forKey key: Key) throws -> Float {
    try _decode(type, forKey: key) { input in
      guard case let .float(v) = input else { return nil }
      return v
    }
  }

  func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
    let value = try _decode(key: key)
    let decoder = MinimalDecoder.Decoder(base: base, parent: self, input: value, key: key)
    pendingContainer = decoder
    defer {
      expectIdentical(pendingContainer, decoder, trapping: true)
      commitPendingContainer()
    }
    return try T(from: decoder)
  }

  func nestedContainer<NestedKey: CodingKey>(
    keyedBy type: NestedKey.Type,
    forKey key: Key
  ) throws -> KeyedDecodingContainer<NestedKey> {
    let value = try _decode(key: key)
    guard case let .dictionary(v) = value else {
      throw DecodingError._typeMismatch(at: codingPath + [key],
                                        expectation: [String: Value].self,
                                        reality: value)
    }
    let nested = MinimalDecoder.KeyedContainer<NestedKey>(base: base, parent: self, input: v, key: key)
    pendingContainer = nested
    return .init(nested)
  }

  func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
    let value = try _decode(key: key)
    guard case let .array(v) = value else {
      throw DecodingError._typeMismatch(at: codingPath + [key],
                                        expectation: [String: Value].self,
                                        reality: value)
    }
    let nested = MinimalDecoder.UnkeyedContainer(base: base, parent: self, input: v, key: key)
    pendingContainer = nested
    return nested
  }

  func superDecoder() throws -> Decoder {
    let value = try _decode(key: MinimalDecoder._Key.super)
    let nested = MinimalDecoder.Decoder(base: base, parent: self, input: value, key: key)
    pendingContainer = nested
    return nested
  }

  func superDecoder(forKey key: Key) throws -> Decoder {
    let value = try _decode(key: key)
    let nested = MinimalDecoder.Decoder(base: base, parent: self, input: value, key: key)
    pendingContainer = nested
    return nested
  }
}

extension MinimalDecoder {
  class UnkeyedContainer {
    typealias Value = MinimalEncoder.Value

    unowned let base: MinimalDecoder
    unowned let parent: _MinimalDecoderContainer?
    let key: CodingKey?
    var input: [Value]
    var isValid = true
    var pendingContainer: AnyObject?
    var index: Int = 0

    init(base: MinimalDecoder, parent: _MinimalDecoderContainer?, input: [Value], key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
      self.input = input
    }

    func commitPendingContainer() {
      expectTrue(self.isValid, "Container isn't valid", trapping: true)
      guard let c = pendingContainer else { return }
      self.pendingContainer = nil
      let container = c as! _MinimalDecoderContainer
      container.finalize()
    }
  }
}

extension MinimalDecoder.UnkeyedContainer: _MinimalDecoderContainer {
  func finalize() {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    isValid = false
  }
}

extension MinimalDecoder.UnkeyedContainer: UnkeyedDecodingContainer {
  typealias _Key = MinimalDecoder._Key

  var count: Int? {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    return input.count
  }

  var isAtEnd: Bool {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    return index == input.count
  }

  var currentIndex: Int {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    return index
  }

  func _decodeNext<T>(_ type: T.Type) throws -> Value {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    guard index < input.count else {
      let context = DecodingError.Context(
        codingPath: codingPath + [_Key(index)],
        debugDescription: "Unkeyed container is at end")
        throw DecodingError.valueNotFound(T.self, context)
    }
    let value = input[index]
    index += 1
    return value
  }

  func _decodeNext<T>(_ type: T.Type = T.self, _ extract: (Value) -> T?) throws -> T {
    let value = try _decodeNext(type)
    guard let v = extract(value) else {
      throw DecodingError._typeMismatch(at: codingPath + [_Key(index - 1)],
                                        expectation: T.self,
                                        reality: value)
    }
    return v
  }


  func decodeNil() throws -> Bool {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    guard index < input.count else {
      let context = DecodingError.Context(
        codingPath: codingPath + [_Key(index)],
        debugDescription: "Unkeyed container is at end")
        throw DecodingError.valueNotFound(Any?.self, context)
    }
    let value = input[index]
    guard value == .null else { return false }
    index += 1
    return true
  }

  func decode(_ type: Bool.Type) throws -> Bool {
    try _decodeNext() { value in
      guard case let .bool(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Float.Type) throws -> Float {
    try _decodeNext() { value in
      guard case let .float(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Double.Type) throws -> Double {
    try _decodeNext() { value in
      guard case let .double(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int.Type) throws -> Int {
    try _decodeNext() { value in
      guard case let .int(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int8.Type) throws -> Int8 {
    try _decodeNext() { value in
      guard case let .int8(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int16.Type) throws -> Int16 {
    try _decodeNext() { value in
      guard case let .int16(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int32.Type) throws -> Int32 {
    try _decodeNext() { value in
      guard case let .int32(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int64.Type) throws -> Int64 {
    try _decodeNext() { value in
      guard case let .int64(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt.Type) throws -> UInt {
    try _decodeNext() { value in
      guard case let .uint(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt8.Type) throws -> UInt8 {
    try _decodeNext() { value in
      guard case let .uint8(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt16.Type) throws -> UInt16 {
    try _decodeNext() { value in
      guard case let .uint16(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt32.Type) throws -> UInt32 {
    try _decodeNext() { value in
      guard case let .uint32(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt64.Type) throws -> UInt64 {
    try _decodeNext() { value in
      guard case let .uint64(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: String.Type) throws -> String {
    try _decodeNext() { value in
      guard case let .string(v) = value else { return nil }
      return v
    }
  }

  func decode<T: Decodable>(_ type: T.Type) throws -> T {
    let key = _Key(index)
    let value = try _decodeNext(T.self)
    let decoder = MinimalDecoder.Decoder(base: base, parent: self, input: value, key: key)
    pendingContainer = decoder
    defer {
      expectIdentical(pendingContainer, decoder, trapping: true)
      commitPendingContainer()
    }
    return try T(from: decoder)
  }

  func nestedContainer<NestedKey: CodingKey>(
    keyedBy type: NestedKey.Type
  ) throws -> KeyedDecodingContainer<NestedKey> {
    let key = _Key(index)
    let value = try _decodeNext([String: Value].self)
    guard case let .dictionary(v) = value else {
      throw DecodingError._typeMismatch(at: codingPath + [key],
                                        expectation: [String: Value].self,
                                        reality: value)
    }
    let nested = MinimalDecoder.KeyedContainer<NestedKey>(base: base, parent: parent, input: v, key: key)
    pendingContainer = nested
    return .init(nested)
  }

  func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer {
    let key = _Key(index)
    let value = try _decodeNext([Value].self)
    guard case let .array(v) = value else {
      throw DecodingError._typeMismatch(at: codingPath + [key],
                                        expectation: [Value].self,
                                        reality: value)
    }
    let nested = MinimalDecoder.UnkeyedContainer(base: base, parent: parent, input: v, key: key)
    pendingContainer = nested
    return nested
  }

  func superDecoder() throws -> Decoder {
    let key = _Key(index)
    let value = try _decodeNext(Any.self)
    let nested = MinimalDecoder.Decoder(base: base, parent: self, input: value, key: key)
    pendingContainer = nested
    return nested
  }
}

extension MinimalDecoder {
  class SingleValueContainer {
    typealias Value = MinimalEncoder.Value

    unowned let base: MinimalDecoder
    unowned let parent: _MinimalDecoderContainer?
    let key: CodingKey?
    var input: Value
    var isValid = true
    var done = false

    init(base: MinimalDecoder, parent: _MinimalDecoderContainer?, input: Value, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
      self.input = input
    }
  }
}

extension MinimalDecoder.SingleValueContainer: _MinimalDecoderContainer {
  func finalize() {
    expectTrue(isValid, "Container isn't valid")
    isValid = false
  }
}

extension MinimalDecoder.SingleValueContainer: SingleValueDecodingContainer {
  func _decodeNext<T>(_ type: T.Type = T.self, _ extract: (Value) -> T?) throws -> T {
    expectTrue(isValid, "Container isn't valid")
    expectFalse(done, "Cannot decode multiple values from a single-value container")
    guard let v = extract(input) else {
      throw DecodingError._typeMismatch(at: codingPath,
                                        expectation: T.self,
                                        reality: input)
    }
    done = true
    return v
  }

  func decodeNil() -> Bool {
    expectTrue(isValid, "Container isn't valid")
    expectFalse(done, "Cannot decode multiple values from a single-value container")
    guard input == .null else { return false }
    done = true
    return true
  }

  func decode(_ type: Bool.Type) throws -> Bool {
    try _decodeNext { value in
      guard case let .bool(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int.Type) throws -> Int {
    try _decodeNext { value in
      guard case let .int(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int8.Type) throws -> Int8 {
    try _decodeNext { value in
      guard case let .int8(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int16.Type) throws -> Int16 {
    try _decodeNext { value in
      guard case let .int16(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int32.Type) throws -> Int32 {
    try _decodeNext { value in
      guard case let .int32(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Int64.Type) throws -> Int64 {
    try _decodeNext { value in
      guard case let .int64(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt.Type) throws -> UInt {
    try _decodeNext { value in
      guard case let .uint(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt8.Type) throws -> UInt8 {
    try _decodeNext { value in
      guard case let .uint8(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt16.Type) throws -> UInt16 {
    try _decodeNext { value in
      guard case let .uint16(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt32.Type) throws -> UInt32 {
    try _decodeNext { value in
      guard case let .uint32(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: UInt64.Type) throws -> UInt64 {
    try _decodeNext { value in
      guard case let .uint64(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Float.Type) throws -> Float {
    try _decodeNext { value in
      guard case let .float(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: Double.Type) throws -> Double {
    try _decodeNext { value in
      guard case let .double(v) = value else { return nil }
      return v
    }
  }

  func decode(_ type: String.Type) throws -> String {
    try _decodeNext { value in
      guard case let .string(v) = value else { return nil }
      return v
    }
  }

  func decode<T: Decodable>(_ type: T.Type) throws -> T {
    expectTrue(isValid, "Container isn't valid")
    expectFalse(done, "Cannot decode multiple values from a single-value container")
    let decoder = MinimalDecoder.Decoder(base: base, parent: self, input: input, key: nil)
    defer {
      done = true
      decoder.finalize()
    }
    return try T(from: decoder)
  }
}
