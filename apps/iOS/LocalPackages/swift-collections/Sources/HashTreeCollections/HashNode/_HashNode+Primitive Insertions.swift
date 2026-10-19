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

// MARK: Node-level insertion operations

extension _HashNode.UnsafeHandle {
  /// Make room for a new item at `slot` corresponding to `bucket`.
  /// There must be enough free space in the node to fit the new item.
  ///
  /// `itemMap` must not already reflect the insertion at the time this
  /// function is called. This method does not update `itemMap`.
  ///
  /// - Returns: an unsafe mutable pointer to uninitialized memory that is
  ///    ready to store the new item. It is the caller's responsibility to
  ///    initialize this memory.
  @inlinable
  internal func _makeRoomForNewItem(
    at slot: _HashSlot, _ bucket: _Bucket
  ) -> UnsafeMutablePointer<Element> {
    assertMutable()
    let c = itemCount
    assert(slot.value <= c)

    let stride = MemoryLayout<Element>.stride
    assert(bytesFree >= stride)
    bytesFree &-= stride

    let start = _memory
      .advanced(by: byteCapacity &- (c &+ 1) &* stride)
      .bindMemory(to: Element.self, capacity: 1)

    let prefix = c &- slot.value
    start.moveInitialize(from: start + 1, count: prefix)

    if bucket.isInvalid {
      assert(isCollisionNode)
      collisionCount &+= 1
    } else {
      assert(!itemMap.contains(bucket))
      assert(!childMap.contains(bucket))
      itemMap.insert(bucket)
      assert(itemMap.slot(of: bucket) == slot)
    }

    return start + prefix
  }

  /// Insert `child` at `slot`. There must be enough free space in the node
  /// to fit the new child.
  ///
  /// `childMap` must not yet reflect the insertion at the time this
  /// function is called. This method does not update `childMap`.
  @inlinable
  internal func _insertChild(_ child: __owned _HashNode, at slot: _HashSlot) {
    assertMutable()
    assert(!isCollisionNode)

    let c = childMap.count
    assert(slot.value <= c)

    let stride = MemoryLayout<_HashNode>.stride
    assert(bytesFree >= stride)
    bytesFree &-= stride

    _memory.bindMemory(to: _HashNode.self, capacity: c &+ 1)
    let q = _childrenStart + slot.value
    (q + 1).moveInitialize(from: q, count: c &- slot.value)
    q.initialize(to: child)
  }
}

extension _HashNode {
  @inlinable @inline(__always)
  internal mutating func insertItem(
    _ item: __owned Element, at bucket: _Bucket
  ) {
    let slot = read { $0.itemMap.slot(of: bucket) }
    self.insertItem(item, at: slot, bucket)
  }

  @inlinable @inline(__always)
  internal mutating func insertItem(
    _ item: __owned Element, at slot: _HashSlot, _ bucket: _Bucket
  ) {
    self.count &+= 1
    update {
      let p = $0._makeRoomForNewItem(at: slot, bucket)
      p.initialize(to: item)
    }
  }

  /// Insert `child` in `bucket`. There must be enough free space in the
  /// node to fit the new child.
  @inlinable
  internal mutating func insertChild(
    _ child: __owned _HashNode, _ bucket: _Bucket
  ) {
    count &+= child.count
    update {
      assert(!$0.isCollisionNode)
      assert(!$0.itemMap.contains(bucket))
      assert(!$0.childMap.contains(bucket))

      let slot = $0.childMap.slot(of: bucket)
      $0._insertChild(child, at: slot)
      $0.childMap.insert(bucket)
    }
  }
}
