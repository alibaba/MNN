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

extension Deque: Encodable where Element: Encodable {
  /// Encodes the elements of this deque into the given encoder in an unkeyed
  /// container.
  ///
  /// This function throws an error if any values are invalid for the given
  /// encoder's format.
  ///
  /// - Parameter encoder: The encoder to write data to.
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.unkeyedContainer()
    for element in self {
      try container.encode(element)
    }
  }
}

extension Deque: Decodable where Element: Decodable {
  /// Creates a new deque by decoding from the given decoder.
  ///
  /// This initializer throws an error if reading from the decoder fails, or
  /// if the data read is corrupted or otherwise invalid.
  ///
  /// - Parameter decoder: The decoder to read data from.
  @inlinable
  public init(from decoder: Decoder) throws {
    self.init()

    var container = try decoder.unkeyedContainer()
    if let count = container.count {
      self.reserveCapacity(count)
    }
    while !container.isAtEnd {
      let element = try container.decode(Element.self)
      self.append(element)
    }
  }
}
