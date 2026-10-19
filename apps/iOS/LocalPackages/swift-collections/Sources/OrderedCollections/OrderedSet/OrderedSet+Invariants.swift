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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension OrderedSet {
  /// True if consistency checking is enabled in the implementation of this
  /// type, false otherwise.
  ///
  /// Documented performance promises are null and void when this property
  /// returns true -- for example, operations that are documented to take
  /// O(1) time might take O(*n*) time, or worse.
  public static var _isConsistencyCheckingEnabled: Bool {
    _isCollectionsInternalCheckingEnabled
  }

  #if COLLECTIONS_INTERNAL_CHECKS
  @inlinable
  @inline(never) @_effects(releasenone)
  public func _checkInvariants() {
    if _table == nil {
      precondition(_elements.count <= _HashTable.maximumUnhashedCount,
                   "Oversized set without a hash table")
      precondition(Set(_elements).count == _elements.count,
                   "Duplicate items in set")
      return
    }
    // Check that each element in _elements can be found in the hash table.
    for index in _elements.indices {
      let item = _elements[index]
      let i = _find(item).index
      precondition(i != nil,
                   "Index \(index) not found in hash table (element: \(item))")
      precondition(
        i == index,
        "Offset of element '\(item)' in hash table differs from its position")
    }
    // Check that the hash table has exactly as many entries as there are elements.
    _table!.read { hashTable in
      var it = hashTable.bucketIterator(startingAt: _Bucket(offset: 0))
      var c = 0
      repeat {
        it.advance()
        if it.isOccupied { c += 1 }
      } while it.currentBucket.offset != 0
      precondition(
        c == _elements.count,
        """
        Number of entries in hash table (\(c)) differs
        from number of elements (\(_elements.count))
        """)
    }
  }
  #else
  @inline(__always) @inlinable
  public func _checkInvariants() {}
  #endif // COLLECTIONS_INTERNAL_CHECKS
}
