//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

/// A previously decoded parsable arguments type.
///
/// Because arguments are consumed and decoded the first time they're
/// encountered, we save the decoded instances for using later in the
/// command/subcommand hierarchy.
struct DecodedArguments {
  var type: ParsableArguments.Type
  var value: ParsableArguments

  var commandType: ParsableCommand.Type? {
    type as? ParsableCommand.Type
  }

  var command: ParsableCommand? {
    value as? ParsableCommand
  }
}

/// A decoder that decodes from parsed command-line arguments.
final class ArgumentDecoder: Decoder {
  init(values: ParsedValues, previouslyDecoded: [DecodedArguments] = []) {
    self.values = values
    self.previouslyDecoded = previouslyDecoded
    self.usedOrigins = InputOrigin()
  }
  
  let values: ParsedValues
  var usedOrigins: InputOrigin
  var nextCommandIndex = 0
  var previouslyDecoded: [DecodedArguments] = []
  
  var codingPath: [CodingKey] = []
  
  var userInfo: [CodingUserInfoKey : Any] = [:]
  
  func container<K>(keyedBy type: K.Type) throws -> KeyedDecodingContainer<K> where K: CodingKey {
    let container = ParsedArgumentsContainer(for: self, keyType: K.self, codingPath: codingPath)
    return KeyedDecodingContainer(container)
  }
  
  func unkeyedContainer() throws -> UnkeyedDecodingContainer {
    throw Error.topLevelHasNoUnkeyedContainer
  }
  
  func singleValueContainer() throws -> SingleValueDecodingContainer {
    throw Error.topLevelHasNoSingleValueContainer
  }
}

extension ArgumentDecoder {
  fileprivate func element(forKey key: InputKey) -> ParsedValues.Element? {
    guard let element = values.element(forKey: key) else { return nil }
    usedOrigins.formUnion(element.inputOrigin)
    return element
  }
}

extension ArgumentDecoder {
  enum Error: Swift.Error {
    case topLevelHasNoUnkeyedContainer
    case topLevelHasNoSingleValueContainer
    case singleValueDecoderHasNoContainer
    case wrongKeyType(CodingKey.Type, CodingKey.Type)
  }
}

final class ParsedArgumentsContainer<K>: KeyedDecodingContainerProtocol where K : CodingKey {
  var codingPath: [CodingKey]
  
  let decoder: ArgumentDecoder
  
  init(for decoder: ArgumentDecoder, keyType: K.Type, codingPath: [CodingKey]) {
    self.codingPath = codingPath
    self.decoder = decoder
  }
  
  var allKeys: [K] {
    fatalError()
  }
  
  fileprivate func element(forKey key: K) -> ParsedValues.Element? {
    let k = InputKey(codingKey: key, path: codingPath)
    return decoder.element(forKey: k)
  }
  
  func contains(_ key: K) -> Bool {
    return element(forKey: key) != nil
  }
  
  func decodeNil(forKey key: K) throws -> Bool {
    return element(forKey: key)?.value == nil
  }
  
  func decode<T>(_ type: T.Type, forKey key: K) throws -> T where T : Decodable {
    let parsedElement = element(forKey: key)
    if parsedElement?.inputOrigin.isDefaultValue ?? false, let rawValue = parsedElement?.value {
      guard let value = rawValue as? T else {
        throw InternalParseError.wrongType(valueRepresentation: "\(rawValue)", forKey: parsedElement!.key)
      }
      return value
    }
    let subDecoder = SingleValueDecoder(userInfo: decoder.userInfo, underlying: decoder, codingPath: codingPath + [key], key: InputKey(codingKey: key, path: codingPath), parsedElement: element(forKey: key))
    return try type.init(from: subDecoder)
  }
  
  func decodeIfPresent<T>(_ type: T.Type, forKey key: KeyedDecodingContainer<K>.Key) throws -> T? where T : Decodable {
    let parsedElement = element(forKey: key)
    if let parsedElement = parsedElement, parsedElement.inputOrigin.isDefaultValue {
      return parsedElement.value as? T
    }
    let subDecoder = SingleValueDecoder(userInfo: decoder.userInfo, underlying: decoder, codingPath: codingPath + [key], key: InputKey(codingKey: key, path: codingPath), parsedElement: parsedElement)
    do {
      return try type.init(from: subDecoder)
    } catch let error as ParserError {
      if case .noValue = error {
        return nil
      } else {
        throw error
      }
    }
  }
  
  func nestedContainer<NestedKey>(keyedBy type: NestedKey.Type, forKey key: K) throws -> KeyedDecodingContainer<NestedKey> where NestedKey : CodingKey {
    fatalError()
  }
  
  func nestedUnkeyedContainer(forKey key: K) throws -> UnkeyedDecodingContainer {
    fatalError()
  }
  
  func superDecoder() throws -> Decoder {
    fatalError()
  }
  
  func superDecoder(forKey key: K) throws -> Decoder {
    fatalError()
  }
}

struct SingleValueDecoder: Decoder {
  var userInfo: [CodingUserInfoKey : Any]
  var underlying: ArgumentDecoder
  var codingPath: [CodingKey]
  var key: InputKey
  var parsedElement: ParsedValues.Element?
  
  func container<K>(keyedBy type: K.Type) throws -> KeyedDecodingContainer<K> where K: CodingKey {
    return KeyedDecodingContainer(ParsedArgumentsContainer(for: underlying, keyType: type, codingPath: codingPath))
  }
  
  func unkeyedContainer() throws -> UnkeyedDecodingContainer {
    guard let e = parsedElement else {
      var errorPath = codingPath
      let last = errorPath.popLast()!
      throw ParserError.noValue(forKey: InputKey(codingKey: last, path: errorPath))
    }
    guard let a = e.value as? [Any] else {
      throw ParserError.invalidState
    }
    return UnkeyedContainer(codingPath: codingPath, parsedElement: e, array: ArrayWrapper(a))
  }
  
  func singleValueContainer() throws -> SingleValueDecodingContainer {
    return SingleValueContainer(underlying: self, codingPath: codingPath, parsedElement: parsedElement)
  }
  
  func previousValue<T>(_ type: T.Type) throws -> T {
    guard let previous = underlying.previouslyDecoded.first(where: { type == $0.type })
      else { throw ParserError.invalidState }
    return previous.value as! T
  }

  func saveValue<T: ParsableArguments>(_ value: T, type: T.Type = T.self) {
    underlying.previouslyDecoded.append(DecodedArguments(type: type, value: value))
  }
  
  struct SingleValueContainer: SingleValueDecodingContainer {
    var underlying: SingleValueDecoder
    var codingPath: [CodingKey]
    var parsedElement: ParsedValues.Element?
    
    func decodeNil() -> Bool {
      return parsedElement == nil
    }
    
    func decode<T>(_ type: T.Type) throws -> T where T : Decodable {
      guard let e = parsedElement else {
        var errorPath = codingPath
        let last = errorPath.popLast()!
        throw ParserError.noValue(forKey: InputKey(codingKey: last, path: errorPath))
      }
      guard let s = e.value as? T else {
        throw InternalParseError.wrongType(valueRepresentation: "nil", forKey: e.key)
      }
      return s
    }
  }
  
  struct UnkeyedContainer: UnkeyedDecodingContainer {
    var codingPath: [CodingKey]
    var parsedElement: ParsedValues.Element
    var array: ArrayWrapperProtocol
    
    var count: Int? {
      return array.count
    }
    
    var isAtEnd: Bool {
      return array.isAtEnd
    }
    
    var currentIndex: Int {
      return array.currentIndex
    }
    
    mutating func decodeNil() throws -> Bool {
      return false
    }
    
    mutating func decode<T>(_ type: T.Type) throws -> T where T : Decodable {
      guard let next = array.getNext() else { fatalError() }
      guard let t = next as? T else {
        throw InternalParseError.wrongType(valueRepresentation: "\(next)", forKey: parsedElement.key)
      }
      return t
    }
    
    mutating func nestedContainer<NestedKey>(keyedBy type: NestedKey.Type) throws -> KeyedDecodingContainer<NestedKey> where NestedKey : CodingKey {
      fatalError()
    }
    
    mutating func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer {
      fatalError()
    }
    
    mutating func superDecoder() throws -> Decoder {
      fatalError()
    }
  }
}

/// A type-erasing wrapper for consuming elements of an array.
protocol ArrayWrapperProtocol {
  var count: Int? { get }
  var isAtEnd: Bool { get }
  var currentIndex: Int { get }
  mutating func getNext() -> Any?
}

struct ArrayWrapper<A>: ArrayWrapperProtocol {
  var base: [A]
  var currentIndex: Int

  init(_ a: [A]) {
    self.base = a
    self.currentIndex = a.startIndex
  }
  
  var count: Int? {
    return base.count
  }
  
  var isAtEnd: Bool {
    return base.endIndex <= currentIndex
  }
  
  mutating func getNext() -> Any? {
    guard currentIndex < base.endIndex else { return nil }
    let next = base[currentIndex]
    currentIndex += 1
    return next
  }
}
