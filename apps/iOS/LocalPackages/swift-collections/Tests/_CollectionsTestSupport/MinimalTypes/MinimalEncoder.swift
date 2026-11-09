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

/// A Plist-style encoder that attempts to enforce a strict interpretation of
/// the `Encodable` protocol requirements. The encoded value is of an enum type
/// that exactly reflects the values encoded, allowing easy matching of
/// actual vs expected outputs.
///
/// This encoder verifies that `encoder(to:)` implementations do not keep
/// containers alive beyond their expected lifetime.
///
/// FIXME: Fix error reporting. We ought to throw errors instead of trapping
/// on validation problems, and we should include context information to help
/// isolating the problem.
public class MinimalEncoder {
  public enum Value: Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case int8(Int8)
    case int16(Int16)
    case int32(Int32)
    case int64(Int64)
    case uint(UInt)
    case uint8(UInt8)
    case uint16(UInt16)
    case uint32(UInt32)
    case uint64(UInt64)
    case float(Float)
    case double(Double)
    case string(String)
    case array([Value])
    case dictionary([String: Value])
  }

  var userInfo: [CodingUserInfoKey: Any]

  public init(userInfo: [CodingUserInfoKey: Any] = [:]) {
    self.userInfo = userInfo
  }

  public func encode<T: Encodable>(_ value: T) throws -> Value {
    let encoder = Encoder(base: self, parent: nil, key: nil)
    return try encoder._encode(value, key: nil)
  }

  public static func encode<T: Encodable>(_ value: T) throws -> Value {
    let base = MinimalEncoder()
    return try base.encode(value)
  }
}

extension MinimalEncoder {
  struct _Key: CodingKey {
    var stringValue: String
    var intValue: Int?

    static let `super` = Self("super")

    init(_ string: String) {
      self.stringValue = string
      self.intValue = nil
    }

    init(_ index: Int) {
      self.stringValue = "Index \(index)"
      self.intValue = index
    }

    init?(stringValue: String) {
      self.init(stringValue)
    }

    init?(intValue: Int) {
      self.init(intValue)
    }
  }
}

extension MinimalEncoder.Value: CustomStringConvertible {
  func _description(prefix: String, indent: String) -> String {
    switch self {
    case .null: return ".null"
    case .bool(let v): return ".bool(\(v))"
    case .int(let v): return ".int(\(v))"
    case .int8(let v): return ".int8(\(v))"
    case .int16(let v): return ".int16(\(v))"
    case .int32(let v): return ".int32(\(v))"
    case .int64(let v): return ".int64(\(v))"
    case .uint(let v): return ".uint(\(v))"
    case .uint8(let v): return ".uint8(\(v))"
    case .uint16(let v): return ".uint16(\(v))"
    case .uint32(let v): return ".uint32(\(v))"
    case .uint64(let v): return ".uint64(\(v))"
    case .float(let v): return ".float(\(v))"
    case .double(let v): return ".double(\(v))"
    case .string(let v): return ".string(\(String(reflecting: v)))"
    case .array(let value):
      if value.count == 0 { return ".array([])" }
      var result = ".array([\n"
      for v in value {
        result += prefix
        result += indent
        result += v._description(prefix: prefix + indent, indent: indent)
        result += ",\n"
      }
      result += prefix + "])"
      return result
    case .dictionary(let value):
      if value.count == 0 { return ".dictionary([:])" }
      var result = ".dictionary([\n"
      for (k, v) in value.sorted(by: { $0.key < $1.key }) {
        result += prefix
        result += indent
        result += String(reflecting: k)
        result += ": "
        result += v._description(prefix: prefix + indent, indent: indent)
        result += ",\n"
      }
      result += prefix + "])"
      return result
    }
  }

  public var description: String {
    self._description(prefix: "", indent: "  ")
  }
}

extension MinimalDecoder.Value {
  public var typeDescription: String {
    switch self {
    case .null: return ".null"
    case .bool(_): return ".bool"
    case .int(_): return ".int"
    case .int8(_): return ".int8"
    case .int16(_): return ".int16"
    case .int32(_): return ".int32"
    case .int64(_): return ".int64"
    case .uint(_): return ".uint"
    case .uint8(_): return ".uint8"
    case .uint16(_): return ".uint16"
    case .uint32(_): return ".uint32"
    case .uint64(_): return ".uint64"
    case .float(_): return ".float"
    case .double(_): return ".double"
    case .string(_): return ".string"
    case .array(_): return ".array"
    case .dictionary(_): return ".dictionary"
    }
  }
}

protocol _MinimalEncoderContainer: AnyObject {
  var base: MinimalEncoder { get }
  var parent: _MinimalEncoderContainer? { get }
  var key: CodingKey? { get }
  var isValid: Bool { get }
  func finalize() -> MinimalEncoder.Value?
}

extension _MinimalEncoderContainer {
  var codingPath: [CodingKey] {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    var path: [CodingKey] = []
    var encoder: _MinimalEncoderContainer? = self
    while let e = encoder {
      if let key = e.key { path.append(key) }
      encoder = e.parent
    }
    path.reverse()
    return path
  }

  func _encode<T: Encodable>(_ value: T, key: CodingKey?) throws -> MinimalEncoder.Value {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    var encoder = MinimalEncoder.Encoder(base: base, parent: self, key: key)
    try value.encode(to: encoder)
    expectTrue(isKnownUniquelyReferenced(&encoder), "\(T.self).encode(to:) retained its encoder", trapping: true)
    guard let result = encoder.finalize() else {
      preconditionFailure("\(T.self).encode(to:) failed to encode anything")
    }
    return result
  }
}

extension MinimalEncoder {
  final class Encoder {
    unowned let base: MinimalEncoder
    unowned let parent: _MinimalEncoderContainer?
    let key: CodingKey?
    var container: _MinimalEncoderContainer?
    var isValid = true

    init(base: MinimalEncoder, parent: _MinimalEncoderContainer?, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
    }
  }
}

extension MinimalEncoder.Encoder: _MinimalEncoderContainer {
  func finalize() -> MinimalEncoder.Value? {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    let value = container?.finalize()
    isValid = false
    return value
  }
}

extension MinimalEncoder.Encoder: Encoder {
  var userInfo: [CodingUserInfoKey: Any] { base.userInfo }

  func container<Key: CodingKey>(
    keyedBy type: Key.Type
  ) -> KeyedEncodingContainer<Key> {
    expectTrue(isValid, "Encoder isn't valid", trapping: true)
    expectNil(container, "Cannot extract multiple containers", trapping: true)
    let keyed = MinimalEncoder.KeyedContainer<Key>(base: base, parent: self, key: nil)
    self.container = keyed
    return .init(keyed)
  }

  func unkeyedContainer() -> UnkeyedEncodingContainer {
    expectTrue(isValid, "Encoder isn't valid", trapping: true)
    expectNil(container, "Cannot extract multiple containers", trapping: true)
    let unkeyed = MinimalEncoder.UnkeyedContainer(base: base, parent: self, key: nil)
    self.container = unkeyed
    return unkeyed
  }

  func singleValueContainer() -> SingleValueEncodingContainer {
    expectTrue(isValid, "Encoder isn't valid", trapping: true)
    expectNil(container, "Cannot extract multiple containers", trapping: true)
    let single = MinimalEncoder.SingleValueContainer(base: base, parent: self, key: nil)
    self.container = single
    return single
  }
}


extension MinimalEncoder {
  final class KeyedContainer<Key: CodingKey> {
    unowned let base: MinimalEncoder
    unowned let parent: _MinimalEncoderContainer?
    var key: CodingKey?
    var values: [String: Value] = [:]
    var isValid = true
    var pendingContainer: AnyObject?

    init(base: MinimalEncoder, parent: _MinimalEncoderContainer?, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
    }

    func commitPendingContainer() {
      expectTrue(self.isValid, "Container isn't valid", trapping: true)
      guard var c = pendingContainer else { return }
      self.pendingContainer = nil
      expectTrue(isKnownUniquelyReferenced(&c), "Container was retained", trapping: true)
      let container = c as! _MinimalEncoderContainer
      let key = container.key!
      expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
      guard let v = container.finalize() else { preconditionFailure("Container remained empty")}
      values[key.stringValue] = v
    }
  }
}

extension MinimalEncoder.KeyedContainer: _MinimalEncoderContainer {
  func finalize() -> MinimalEncoder.Value? {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    isValid = false
    return .dictionary(values)
  }
}

extension MinimalEncoder.KeyedContainer: KeyedEncodingContainerProtocol {

  func encodeNil(forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .null
  }
  func encode(_ value: Bool, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .bool(value)
  }
  func encode(_ value: Int, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .int(value)
  }
  func encode(_ value: Int8, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .int8(value)
  }
  func encode(_ value: Int16, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .int16(value)
  }
  func encode(_ value: Int32, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .int32(value)
  }
  func encode(_ value: Int64, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .int64(value)
  }
  func encode(_ value: UInt, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .uint(value)
  }
  func encode(_ value: UInt8, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .uint8(value)
  }
  func encode(_ value: UInt16, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .uint16(value)
  }
  func encode(_ value: UInt32, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .uint32(value)
  }
  func encode(_ value: UInt64, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .uint64(value)
  }
  func encode(_ value: Float, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .float(value)
  }
  func encode(_ value: Double, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .double(value)
  }
  func encode(_ value: String, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = .string(value)
  }
  func encode<T: Encodable>(_ value: T, forKey key: Key) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    values[key.stringValue] = try _encode(value, key: key)
  }
  func nestedContainer<NestedKey: CodingKey>(
    keyedBy keyType: NestedKey.Type,
    forKey key: Key
  ) -> KeyedEncodingContainer<NestedKey> {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    let keyed = MinimalEncoder.KeyedContainer<NestedKey>(base: base, parent: self, key: key)
    pendingContainer = keyed
    return .init(keyed)
  }

  func nestedUnkeyedContainer(forKey key: Key) -> UnkeyedEncodingContainer {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key.stringValue) has already been encoded", trapping: true)
    let unkeyed = MinimalEncoder.UnkeyedContainer(base: base, parent: self, key: key)
    pendingContainer = unkeyed
    return unkeyed
  }

  func superEncoder() -> Encoder {
    let key = MinimalEncoder._Key.super
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key) has already been encoded", trapping: true)
    let sup = MinimalEncoder.Encoder(base: base, parent: self, key: key)
    pendingContainer = sup
    return sup
  }

  func superEncoder(forKey key: Key) -> Encoder {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    expectNil(values[key.stringValue], "Key \(key) has already been encoded", trapping: true)
    let sup = MinimalEncoder.Encoder(base: base, parent: self, key: key)
    pendingContainer = sup
    return sup
  }
}


extension MinimalEncoder {
  final class UnkeyedContainer {
    unowned let base: MinimalEncoder
    unowned let parent: _MinimalEncoderContainer?
    let key: CodingKey?
    var values: [Value] = []
    var isValid = true
    var pendingContainer: AnyObject?

    init(base: MinimalEncoder, parent: _MinimalEncoderContainer?, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
    }

    func commitPendingContainer() {
      expectTrue(self.isValid, "Container isn't valid", trapping: true)
      guard var c = pendingContainer else { return }
      self.pendingContainer = nil
      expectTrue(isKnownUniquelyReferenced(&c), "Container was retained", trapping: true)
      let container = c as! _MinimalEncoderContainer
      guard let v = container.finalize() else { preconditionFailure("Container remained empty")}
      values.append(v)
    }
  }
}

extension MinimalEncoder.UnkeyedContainer: _MinimalEncoderContainer {
  func finalize() -> MinimalEncoder.Value? {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    isValid = false
    return .array(values)
  }
}

extension MinimalEncoder.UnkeyedContainer: UnkeyedEncodingContainer {
  var count: Int {
    values.count
  }

  func encodeNil() throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.null)
  }

  func encode(_ value: Bool) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.bool(value))
  }

  func encode(_ value: Int) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.int(value))
  }

  func encode(_ value: Int8) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.int8(value))
  }

  func encode(_ value: Int16) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.int16(value))
  }

  func encode(_ value: Int32) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.int32(value))
  }

  func encode(_ value: Int64) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.int64(value))
  }

  func encode(_ value: UInt) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.uint(value))
  }

  func encode(_ value: UInt8) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.uint8(value))
  }

  func encode(_ value: UInt16) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.uint16(value))
  }

  func encode(_ value: UInt32) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.uint32(value))
  }

  func encode(_ value: UInt64) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.uint64(value))
  }

  func encode(_ value: Float) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.float(value))
  }

  func encode(_ value: Double) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.double(value))
  }

  func encode(_ value: String) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(.string(value))
  }

  func encode<T: Encodable>(_ value: T) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    values.append(try _encode(value, key: key))
  }

  func nestedContainer<NestedKey: CodingKey>(
    keyedBy keyType: NestedKey.Type
  ) -> KeyedEncodingContainer<NestedKey> {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    let keyed = MinimalEncoder.KeyedContainer<NestedKey>(
      base: base,
      parent: self,
      key: MinimalEncoder._Key(count))
    pendingContainer = keyed
    return .init(keyed)
  }

  func nestedUnkeyedContainer() -> UnkeyedEncodingContainer {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    let unkeyed = MinimalEncoder.UnkeyedContainer(base: base, parent: self, key: MinimalEncoder._Key(count))
    pendingContainer = unkeyed
    return unkeyed
  }

  func superEncoder() -> Encoder {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    commitPendingContainer()
    let sup = MinimalEncoder.Encoder(base: base, parent: self, key: MinimalEncoder._Key(count))
    pendingContainer = sup
    return sup
  }
}

extension MinimalEncoder {
  final class SingleValueContainer {
    unowned let base: MinimalEncoder
    unowned let parent: _MinimalEncoderContainer?
    let key: CodingKey?
    var value: Value?
    var isValid = true

    init(base: MinimalEncoder, parent: _MinimalEncoderContainer?, key: CodingKey?) {
      self.base = base
      self.parent = parent
      self.key = key
    }
  }
}

extension MinimalEncoder.SingleValueContainer: _MinimalEncoderContainer {
  func finalize() -> MinimalEncoder.Value? {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    isValid = false
    return value
  }
}

extension MinimalEncoder.SingleValueContainer: SingleValueEncodingContainer {
  func encodeNil() throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .null
  }

  func encode(_ value: Bool) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .bool(value)
  }

  func encode(_ value: Int) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .int(value)
  }

  func encode(_ value: Int8) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .int8(value)
  }

  func encode(_ value: Int16) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .int16(value)
  }

  func encode(_ value: Int32) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .int32(value)
  }

  func encode(_ value: Int64) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .int64(value)
  }

  func encode(_ value: UInt) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .uint(value)
  }

  func encode(_ value: UInt8) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .uint8(value)
  }

  func encode(_ value: UInt16) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .uint16(value)
  }

  func encode(_ value: UInt32) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .uint32(value)
  }

  func encode(_ value: UInt64) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .uint64(value)
  }

  func encode(_ value: Float) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .float(value)
  }

  func encode(_ value: Double) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .double(value)
  }

  func encode(_ value: String) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = .string(value)
  }

  func encode<T: Encodable>(_ value: T) throws {
    expectTrue(isValid, "Container isn't valid", trapping: true)
    expectNil(self.value, "Cannot encode multiple values into a single-value container", trapping: true)
    self.value = try _encode(value, key: nil)
  }
}
