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

extension TreeSet: Encodable where Element: Encodable {
  /// Encodes the elements of this set into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.unkeyedContainer()
    try container.encode(contentsOf: self)
  }
}

extension TreeSet: Decodable where Element: Decodable {
  /// Creates a new set by decoding from the given decoder.
  ///
  /// This initializer throws an error if reading from the decoder fails, or
  /// if the decoded contents contain duplicate values.
  ///
  /// - Parameter decoder: The decoder to read data from.
  @inlinable
  public init(from decoder: Decoder) throws {
    self.init()

    var container = try decoder.unkeyedContainer()
    while !container.isAtEnd {
      let element = try container.decode(Element.self)
      let inserted = self._insert(element)
      guard inserted else {
        let context = DecodingError.Context(
          codingPath: container.codingPath,
          debugDescription: "Decoded elements aren't unique (first duplicate at offset \(self.count))")
        throw DecodingError.dataCorrupted(context)
      }
    }
  }
}
