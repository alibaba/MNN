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
  /// The storage header for hash table buffers.
  ///
  /// Note that we don't store the number of items currently in the table;
  /// that information can be easily retrieved from the element storage.
  @usableFromInline
  internal struct Header {
    /// We are packing the scale data into the lower bits of the seed & bias
    /// to saves a bit of space that would be otherwise taken up by padding.
    ///
    /// Layout:
    ///
    ///     63                                           6 5      0
    ///    ├──────────────────────────────────────────────┼────────┤
    ///    │                    seed                      │ scale  │
    ///    └──────────────────────────────────────────────┴────────┘
    ///     63                                           6 5      0
    ///    ├──────────────────────────────────────────────┼────────┤
    ///    │                    bias                      │ rsvd   │
    ///    └──────────────────────────────────────────────┴────────┘
    @usableFromInline
    var _scaleAndSeed: UInt64
    @usableFromInline
    var _reservedScaleAndBias: UInt64

    init(scale: Int, reservedScale: Int, seed: Int) {
      assert(scale >= _HashTable.minimumScale && scale <= _HashTable.maximumScale)
      assert(reservedScale >= 0 && reservedScale <= _HashTable.maximumScale)
      _scaleAndSeed = UInt64(truncatingIfNeeded: seed) << (Swift.max(UInt64.bitWidth - Int.bitWidth, 6))
      _scaleAndSeed &= ~0x3F
      _scaleAndSeed |= UInt64(truncatingIfNeeded: scale)
      _reservedScaleAndBias = UInt64(truncatingIfNeeded: reservedScale)
      assert(self.scale == scale)
      assert(self.reservedScale == reservedScale)
      assert(self.bias == 0)
    }

    /// The scale of the hash table. A table of scale *n* holds 2^*n* buckets,
    /// each of which contain an *n*-bit value.
    @inlinable
    @inline(__always)
    var scale: Int { Int(_scaleAndSeed & 0x3F) }

    /// The scale corresponding to the last call to `reserveCapacity`.
    /// We remember this here to make sure we don't shrink the table below its reserved size.
    @inlinable
    var reservedScale: Int {
      @inline(__always)
      get { Int(_reservedScaleAndBias & 0x3F) }
      set {
        assert(newValue >= 0 && newValue < 64)
        _reservedScaleAndBias &= ~0x3F
        _reservedScaleAndBias |= UInt64(truncatingIfNeeded: newValue) & 0x3F
      }
    }

    /// The hasher seed to use within this hash table.
    @inlinable
    @inline(__always)
    var seed: Int {
      Int(truncatingIfNeeded: _scaleAndSeed)
    }

    /// A bias value that needs to be added to buckets to convert them into offsets
    /// into element storage. (This allows O(1) insertions at the front when the
    /// underlying storage supports it.)
    @inlinable
    var bias: Int {
      @inline(__always)
      get { Int(truncatingIfNeeded: _reservedScaleAndBias) &>> 6 }
      set {
        let limit = (1 &<< scale) - 1
        var bias = newValue
        if bias < 0 { bias += limit }
        if bias >= limit { bias -= limit }
        assert(bias >= 0 && bias < limit)
        _reservedScaleAndBias &= 0x3F
        _reservedScaleAndBias |= UInt64(truncatingIfNeeded: bias) &<< 6
        assert(self.bias >= 0 && self.bias < limit)
      }
    }

    /// The maximum number of items that can fit into this table.
    @inlinable
    @inline(__always)
    var capacity: Int { _HashTable.maximumCapacity(forScale: scale) }
  }
}
