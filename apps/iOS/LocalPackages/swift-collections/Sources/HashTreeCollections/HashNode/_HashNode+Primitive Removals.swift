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

// MARK: Node-level removal operations

extension _HashNode.UnsafeHandle {
  /// Remove and return the item at `slot`, increasing the amount of free
  /// space available in the node.
  ///
  /// `itemMap` must not yet reflect the removal at the time this
  /// function is called. This method does not update `itemMap`.
  @inlinable
  internal func _removeItem<R>(
    at slot: _HashSlot,
    by remover: (UnsafeMutablePointer<Element>) -> R
  ) -> R {
    assertMutable()
    let c = itemCount
    assert(slot.value < c)
    let stride = MemoryLayout<Element>.stride
    bytesFree &+= stride

    let start = _memory
      .advanced(by: byteCapacity &- stride &* c)
      .assumingMemoryBound(to: Element.self)

    let prefix = c &- 1 &- slot.value
    let q = start + prefix
    defer {
      (start + 1).moveInitialize(from: start, count: prefix)
    }
    return remover(q)
  }

  /// Remove and return the child at `slot`, increasing the amount of free
  /// space available in the node.
  ///
  /// `childMap` must not yet reflect the removal at the time this
  /// function is called. This method does not update `childMap`.
  @inlinable
  internal func _removeChild(at slot: _HashSlot) -> _HashNode {
    assertMutable()
    assert(!isCollisionNode)
    let count = childCount
    assert(slot.value < count)

    bytesFree &+= MemoryLayout<_HashNode>.stride

    let q = _childrenStart + slot.value
    let child = q.move()
    q.moveInitialize(from: q + 1, count: count &- 1 &- slot.value)
    return child
  }
}

extension _HashNode {
  @inlinable
  internal mutating func removeItem(
    at bucket: _Bucket
  ) -> Element {
    let slot = read { $0.itemMap.slot(of: bucket) }
    return removeItem(at: bucket, slot, by: { $0.move() })
  }

  @inlinable
  internal mutating func removeItem(
    at bucket: _Bucket, _ slot: _HashSlot
  ) -> Element {
    removeItem(at: bucket, slot, by: { $0.move() })
  }

  /// Remove the item at `slot`, increasing the amount of free
  /// space available in the node.
  ///
  /// The closure `remove` is called to perform the deinitialization of the
  /// storage slot corresponding to the item to be removed.
  @inlinable
  internal mutating func removeItem<R>(
    at bucket: _Bucket, _ slot: _HashSlot,
    by remover: (UnsafeMutablePointer<Element>) -> R
  ) -> R {
    defer { _invariantCheck() }
    assert(count > 0)
    count &-= 1
    return update {
      let old = $0._removeItem(at: slot, by: remover)
      if $0.isCollisionNode {
        assert(slot.value < $0.collisionCount)
        $0.collisionCount &-= 1
      } else {
        assert($0.itemMap.contains(bucket))
        assert($0.itemMap.slot(of: bucket) == slot)
        $0.itemMap.remove(bucket)
      }
      return old
    }
  }

  @inlinable
  internal mutating func removeChild(
    at bucket: _Bucket, _ slot: _HashSlot
  ) -> _HashNode {
    assert(!isCollisionNode)
    let child: _HashNode = update {
      assert($0.childMap.contains(bucket))
      assert($0.childMap.slot(of: bucket) == slot)
      let child = $0._removeChild(at: slot)
      $0.childMap.remove(bucket)
      return child
    }
    assert(self.count >= child.count)
    self.count &-= child.count
    return child
  }

  @inlinable
  internal mutating func removeSingletonItem() -> Element {
    defer { _invariantCheck() }
    assert(count == 1)
    count = 0
    return update {
      assert($0.hasSingletonItem)
      let old = $0._removeItem(at: .zero) { $0.move() }
      $0.clear()
      return old
    }
  }

  @inlinable
  internal mutating func removeSingletonChild() -> _HashNode {
    defer { _invariantCheck() }
    let child: _HashNode = update {
      assert($0.hasSingletonChild)
      let child = $0._removeChild(at: .zero)
      $0.childMap = .empty
      return child
    }
    assert(self.count == child.count)
    self.count = 0
    return child
  }
}

