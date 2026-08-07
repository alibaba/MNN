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

public struct _HashTreeStatistics {
  /// The number of nodes in the tree.
  public internal(set) var nodeCount: Int = 0

  /// The number of collision nodes in the tree.
  public internal(set) var collisionNodeCount: Int = 0

  /// The number of elements within this tree.
  public internal(set) var itemCount: Int = 0

  /// The number of elements whose keys have colliding hashes in the tree.
  public internal(set) var collisionCount: Int = 0

  /// The number of key comparisons that need to be done due to hash collisions
  /// when finding every key in the tree.
  public internal(set) var _collisionChainCount: Int = 0

  /// The maximum depth of the tree.
  public internal(set) var maxItemDepth: Int = 0

  internal var _sumItemDepth: Int = 0

  /// The sum of all storage within the tree that is available for item storage,
  /// measured in bytes. (This is storage is shared between actual
  /// items and child references. Depending on alignment issues, not all of
  /// this may be actually usable.)
  public internal(set) var capacityBytes: Int = 0

  /// The number of bytes of storage currently used for storing items.
  public internal(set) var itemBytes: Int = 0

  /// The number of bytes of storage currently used for storing child
  /// references.
  public internal(set) var childBytes: Int = 0

  /// The number of bytes currently available for storage, summed over all
  /// nodes in the tree.
  public internal(set) var freeBytes: Int = 0

  /// An estimate of the actual memory occupied by this tree. This includes
  /// not only storage space for items & children, but also the memory taken up
  /// by node headers and Swift's object headers.
  public internal(set) var grossBytes: Int = 0

  /// The average level of an item within this tree.
  public var averageItemDepth: Double {
    guard nodeCount > 0 else { return 0 }
    return Double(_sumItemDepth) / Double(itemCount)
  }
  /// An estimate of how efficiently this data structure manages memory.
  /// This is a value between 0 and 1 -- the ratio between how much space
  /// the actual stored data occupies and the overall number of bytes allocated
  /// for the entire data structure. (`itemBytes / grossBytes`)
  public var memoryEfficiency: Double {
    guard grossBytes > 0 else { return 1 }
    return Double(itemBytes) / Double(grossBytes)
  }

  public var averageNodeSize: Double {
    guard nodeCount > 0 else { return 0 }
    return Double(capacityBytes) / Double(nodeCount)
  }

  /// The average number of keys that need to be compared within the tree
  /// to find a member item. This is exactly 1 unless the tree contains hash
  /// collisions.
  public var averageLookupChainLength: Double {
    guard itemCount > 0 else { return 1 }
    return Double(itemCount + _collisionChainCount) / Double(itemCount)
  }

  internal init() {
    // Nothing to do
  }
}


extension _HashNode {
  internal func gatherStatistics(
    _ level: _HashLevel, _ stats: inout _HashTreeStatistics
  ) {
    // The empty singleton does not count as a node and occupies no space.
    if self.raw.storage === _emptySingleton { return }

    read {
      stats.nodeCount += 1
      stats.itemCount += $0.itemCount

      if isCollisionNode {
        stats.collisionNodeCount += 1
        stats.collisionCount += $0.itemCount
        stats._collisionChainCount += $0.itemCount * ($0.itemCount - 1) / 2
      }

      let keyStride = MemoryLayout<Key>.stride
      let valueStride = MemoryLayout<Value>.stride

      stats.maxItemDepth = Swift.max(stats.maxItemDepth, level.depth)
      stats._sumItemDepth += (level.depth + 1) * $0.itemCount
      stats.capacityBytes += $0.byteCapacity
      stats.freeBytes += $0.bytesFree
      stats.itemBytes += $0.itemCount * (keyStride + valueStride)
      stats.childBytes += $0.childCount * MemoryLayout<_RawHashNode>.stride

      let objectHeaderSize = 2 * MemoryLayout<Int>.stride

      // Note: for simplicity, we assume that there is no padding between
      // the object header and the storage header.
      let start = _getUnsafePointerToStoredProperties(self.raw.storage)
      let end = $0._memory + $0.byteCapacity
      stats.grossBytes += objectHeaderSize + (end - start)

      for child in $0.children {
        child.gatherStatistics(level.descend(), &stats)
      }
    }
  }
}
