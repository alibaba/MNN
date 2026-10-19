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

struct DictionaryStatistics {
  /// The sum of all storage within the hash table that is available for
  /// item storage, measured in bytes. This does account for the maximum
  /// load factor.
  var capacityBytes: Int = 0

  /// The number of bytes of storage currently used for storing items.
  var itemBytes: Int = 0

  /// The number of bytes currently available in storage for storing items.
  var freeBytes: Int = 0

  /// An estimate of the actual memory occupied by this hash table.
  /// This includes not only storage space available for items,
  /// but also the memory taken up by the object header and the hash table
  /// occupation bitmap.
  var grossBytes: Int = 0

  /// An estimate of how efficiently this data structure manages memory.
  /// This is a value between 0 and 1 -- the ratio between how much space
  /// the actual stored data occupies and the overall number of bytes allocated
  /// for the entire data structure. (`itemBytes / grossBytes`)
  var memoryEfficiency: Double {
    guard grossBytes > 0 else { return 1 }
    return Double(itemBytes) / Double(grossBytes)
  }
}

extension Dictionary {
  var statistics: DictionaryStatistics {
    // Note: This logic is based on the Dictionary ABI. It may be off by a few
    // bytes due to not accounting for padding bytes between storage regions.
    // The gross bytes reported also do not include extra memory that was
    // allocated by malloc but not actually used for Dictionary storage.
    var stats = DictionaryStatistics()
    let keyStride = MemoryLayout<Key>.stride
    let valueStride = MemoryLayout<Value>.stride
    stats.capacityBytes = self.capacity * (keyStride + valueStride)
    stats.itemBytes = self.count * (keyStride + valueStride)
    stats.freeBytes = stats.capacityBytes - stats.itemBytes

    let bucketCount = self.capacity._roundUpToPowerOfTwo()
    let bitmapBitcount = (bucketCount + UInt.bitWidth - 1)

    let objectHeaderBits = 2 * Int.bitWidth
    let ivarBits = 5 * Int.bitWidth + 64
    stats.grossBytes = (objectHeaderBits + ivarBits + bitmapBitcount) / 8
    stats.grossBytes += bucketCount * keyStride + bucketCount * valueStride
    return stats
  }
}
