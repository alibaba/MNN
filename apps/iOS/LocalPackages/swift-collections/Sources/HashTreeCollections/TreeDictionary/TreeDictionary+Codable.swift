//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

// Code in this file is a slightly adapted variant of `Dictionary`'s `Codable`
// implementation in the Standard Library as of Swift 5.7.
// `TreeDictionary` therefore encodes/decodes itself exactly the same as
// `Dictionary`, and the two types can each decode data encoded by the other.

/// A wrapper for dictionary keys which are Strings or Ints.
internal struct _DictionaryCodingKey: CodingKey {
  internal let stringValue: String
  internal let intValue: Int?

  internal init(stringValue: String) {
    self.stringValue = stringValue
    self.intValue = Int(stringValue)
  }

  internal init(intValue: Int) {
    self.stringValue = "\(intValue)"
    self.intValue = intValue
  }

  fileprivate init(codingKey: CodingKey) {
    self.stringValue = codingKey.stringValue
    self.intValue = codingKey.intValue
  }
}

extension TreeDictionary: Encodable
where Key: Encodable, Value: Encodable
{
  /// Encodes the elements of this dictionary into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  public func encode(to encoder: Encoder) throws {
    if Key.self == String.self {
      // Since the keys are already Strings, we can use them as keys directly.
      var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
      for (key, value) in self {
        let codingKey = _DictionaryCodingKey(stringValue: key as! String)
        try container.encode(value, forKey: codingKey)
      }
      return
    }
    if Key.self == Int.self {
      // Since the keys are already Ints, we can use them as keys directly.
      var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
      for (key, value) in self {
        let codingKey = _DictionaryCodingKey(intValue: key as! Int)
        try container.encode(value, forKey: codingKey)
      }
      return
    }
    if #available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *),
            Key.self is CodingKeyRepresentable.Type {
      // Since the keys are CodingKeyRepresentable, we can use the `codingKey`
      // to create `_DictionaryCodingKey` instances.
      var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
      for (key, value) in self {
        let codingKey = (key as! CodingKeyRepresentable).codingKey
        let dictionaryCodingKey = _DictionaryCodingKey(codingKey: codingKey)
        try container.encode(value, forKey: dictionaryCodingKey)
      }
      return
    }
    // Keys are Encodable but not Strings or Ints, so we cannot arbitrarily
    // convert to keys. We can encode as an array of alternating key-value
    // pairs, though.
    var container = encoder.unkeyedContainer()
    for (key, value) in self {
      try container.encode(key)
      try container.encode(value)
    }
  }
}

extension TreeDictionary: Decodable
where Key: Decodable, Value: Decodable
{
  /// Creates a new dictionary by decoding from the given decoder.
  ///
  /// This initializer throws an error if reading from the decoder fails, or
  /// if the data read is corrupted or otherwise invalid.
  ///
  /// - Parameter decoder: The decoder to read data from.
  public init(from decoder: Decoder) throws {
    self.init()

    if Key.self == String.self {
      // The keys are Strings, so we should be able to expect a keyed container.
      let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
      for key in container.allKeys {
        let value = try container.decode(Value.self, forKey: key)
        self[key.stringValue as! Key] = value
      }
      return
    }
    if Key.self == Int.self {
      // The keys are Ints, so we should be able to expect a keyed container.
      let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
      for key in container.allKeys {
        guard key.intValue != nil else {
          // We provide stringValues for Int keys; if an encoder chooses not to
          // use the actual intValues, we've encoded string keys.
          // So on init, _DictionaryCodingKey tries to parse string keys as
          // Ints. If that succeeds, then we would have had an intValue here.
          // We don't, so this isn't a valid Int key.
          var codingPath = decoder.codingPath
          codingPath.append(key)
          throw DecodingError.typeMismatch(
            Int.self,
            DecodingError.Context(
              codingPath: codingPath,
              debugDescription: "Expected Int key but found String key instead."
            )
          )
        }

        let value = try container.decode(Value.self, forKey: key)
        self[key.intValue! as! Key] = value
      }
      return
    }

    if #available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *),
       let keyType = Key.self as? CodingKeyRepresentable.Type {
      // The keys are CodingKeyRepresentable, so we should be able to expect
      // a keyed container.
      let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
      for codingKey in container.allKeys {
        guard let key: Key = keyType.init(codingKey: codingKey) as? Key else {
          throw DecodingError.dataCorruptedError(
            forKey: codingKey,
            in: container,
            debugDescription: "Could not convert key to type \(Key.self)"
          )
        }
        let value: Value = try container.decode(Value.self, forKey: codingKey)
        self[key] = value
      }
      return
    }

    // We should have encoded as an array of alternating key-value pairs.
    var container = try decoder.unkeyedContainer()

    // We're expecting to get pairs. If the container has a known count, it
    // had better be even; no point in doing work if not.
    if let count = container.count {
      guard count % 2 == 0 else {
        throw DecodingError.dataCorrupted(
          DecodingError.Context(
            codingPath: decoder.codingPath,
            debugDescription: "Expected collection of key-value pairs; encountered odd-length array instead."
          )
        )
      }
    }

    while !container.isAtEnd {
      let key = try container.decode(Key.self)

      guard !container.isAtEnd else {
        throw DecodingError.dataCorrupted(
          DecodingError.Context(
            codingPath: decoder.codingPath,
            debugDescription: "Unkeyed container reached end before value in key-value pair."
          )
        )
      }

      let value = try container.decode(Value.self)
      self[key] = value
    }
  }
}

