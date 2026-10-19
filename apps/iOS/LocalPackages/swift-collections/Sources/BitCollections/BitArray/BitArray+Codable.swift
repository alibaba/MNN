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

extension BitArray: Codable {
  /// Encodes this bit array into the given encoder.
  ///
  /// Bit arrays are encoded as an unkeyed container of `UInt64` values,
  /// representing the total number of bits in the array, followed by
  /// UInt64-sized pieces of the underlying bitmap.
  ///
  /// - Parameter encoder: The encoder to write data to.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.unkeyedContainer()
    try container.encode(UInt64(truncatingIfNeeded: _count))
    try _storage._encodeAsUInt64(to: &container)
  }

  /// Creates a new bit array by decoding from the given decoder.
  ///
  /// Bit arrays are encoded as an unkeyed container of `UInt64` values,
  /// representing the total number of bits in the array, followed by
  /// UInt64-sized pieces of the underlying bitmap.
  ///
  /// - Parameter decoder: The decoder to read data from.
  public init(from decoder: Decoder) throws {
    var container = try decoder.unkeyedContainer()
    guard let count = UInt(exactly: try container.decode(UInt64.self)) else {
      let context = DecodingError.Context(
        codingPath: container.codingPath,
        debugDescription: "Bit Array too long")
      throw DecodingError.dataCorrupted(context)
    }
    var words = try [_Word](
      _fromUInt64: &container,
      reservingCount: container.count.map { Swift.max(1, $0) - 1 })
    if _Word.capacity < UInt64.bitWidth,
        count <= (words.count - 1) * _Word.capacity
    {
      let last = words.removeLast()
      guard last.isEmpty else {
        let context = DecodingError.Context(
          codingPath: container.codingPath,
          debugDescription: "Unexpected bits after end")
        throw DecodingError.dataCorrupted(context)
      }
    }
    guard
      count <= words.count * _Word.capacity,
      count > (words.count - 1) * _Word.capacity
    else {
      let context = DecodingError.Context(
        codingPath: container.codingPath,
        debugDescription: "Decoded words don't match expected count")
      throw DecodingError.dataCorrupted(context)
    }
    let bit = _BitPosition(count).bit
    if bit > 0, !words.last!.subtracting(_Word(upTo: bit)).isEmpty {
      let context = DecodingError.Context(
        codingPath: container.codingPath,
        debugDescription: "Unexpected bits after end")
      throw DecodingError.dataCorrupted(context)
    }
    self.init(_storage: words, count: count)
  }
}
