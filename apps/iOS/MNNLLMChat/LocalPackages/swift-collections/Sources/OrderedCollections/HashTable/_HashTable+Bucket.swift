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

extension _HashTable {
  /// Identifies a particular bucket within a hash table by its offset.
  /// Having a dedicated wrapper type for this prevents passing a bucket number
  /// to a function that expects a word index, or vice versa.
  @usableFromInline
  @frozen
  internal struct Bucket {
    /// The distance of this bucket from the first bucket in the hash table.
    @usableFromInline
    internal var offset: Int

    @inlinable
    @inline(__always)
    internal init(offset: Int) {
      assert(offset >= 0)
      self.offset = offset
    }
  }
}

extension _HashTable.Bucket: Equatable {
  @_transparent
  public static func == (left: Self, right: Self) -> Bool {
    left.offset == right.offset
  }
}
