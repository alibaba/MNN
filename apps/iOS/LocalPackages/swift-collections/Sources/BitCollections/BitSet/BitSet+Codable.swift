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

extension BitSet: Codable {
  /// Encodes this bit set into the given encoder.
  ///
  /// Bit sets are encoded as an unkeyed container of `UInt64` values,
  /// representing pieces of the underlying bitmap.
  ///
  /// - Parameter encoder: The encoder to write data to.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.unkeyedContainer()
    try _storage._encodeAsUInt64(to: &container)
  }

  /// Creates a new bit set by decoding from the given decoder.
  ///
  /// Bit sets are encoded as an unkeyed container of `UInt64` values,
  /// representing pieces of the underlying bitmap.
  ///
  /// - Parameter decoder: The decoder to read data from.
  public init(from decoder: Decoder) throws {
    var container = try decoder.unkeyedContainer()
    let words = try [_Word](
      _fromUInt64: &container, reservingCount: container.count)
    self.init(_words: words)
  }
}
