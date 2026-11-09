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

extension OrderedSet: Encodable where Element: Encodable {
  /// Encodes the elements of this ordered set into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  @inlinable
  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(_elements)
  }
}

extension OrderedSet: Decodable where Element: Decodable {
  /// Creates a new ordered set by decoding from the given decoder.
  ///
  /// This initializer throws an error if reading from the decoder fails, or
  /// if the decoded contents contain duplicate values.
  ///
  /// - Parameter decoder: The decoder to read data from.
  @inlinable
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let elements = try container.decode(ContiguousArray<Element>.self)

    let (table, end) = _HashTable.create(untilFirstDuplicateIn: elements)
    guard end == elements.endIndex else {
      let context = DecodingError.Context(
        codingPath: container.codingPath,
        debugDescription: "Decoded elements aren't unique (first duplicate at offset \(end))")
      throw DecodingError.dataCorrupted(context)
    }
    self.init(
      _uniqueElements: elements,
      elements.count > _HashTable.maximumUnhashedCount ? table : nil)
  }
}
