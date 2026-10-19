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

extension _HashNode {
  @inlinable
  internal mutating func replaceItem(
    at bucket: _Bucket, _ slot: _HashSlot, with item: __owned Element
  ) {
    update {
      assert($0.isCollisionNode || $0.itemMap.contains(bucket))
      assert($0.isCollisionNode || slot == $0.itemMap.slot(of: bucket))
      assert(!$0.isCollisionNode || slot.value < $0.collisionCount)
      $0[item: slot] = item
    }
  }

  @inlinable
  internal mutating func replaceChild(
    at bucket: _Bucket, with child: __owned _HashNode
  ) -> Int {
    let slot = read { $0.childMap.slot(of: bucket) }
    return replaceChild(at: bucket, slot, with: child)
  }

  @inlinable
  internal mutating func replaceChild(
    at bucket: _Bucket, _ slot: _HashSlot, with child: __owned _HashNode
  ) -> Int {
    let delta: Int = update {
      assert(!$0.isCollisionNode)
      assert($0.childMap.contains(bucket))
      assert($0.childMap.slot(of: bucket) == slot)
      let p = $0.childPtr(at: slot)
      let delta = child.count &- p.pointee.count
      p.pointee = child
      return delta
    }
    self.count &+= delta
    return delta
  }

  @inlinable
  internal func replacingChild(
    _ level: _HashLevel,
    at bucket: _Bucket,
    _ slot: _HashSlot,
    with child: __owned Builder
  ) -> Builder {
    assert(child.level == level.descend())
    read {
      assert(!$0.isCollisionNode)
      assert($0.childMap.contains(bucket))
      assert(slot == $0.childMap.slot(of: bucket))
    }
    switch child.kind {
    case .empty:
      return _removingChild(level, at: bucket, slot)
    case let .item(item, _):
      if hasSingletonChild {
        return .item(level, item, at: bucket)
      }
      var node = self.copy(withFreeSpace: _HashNode.spaceForInlinedChild)
      _ = node.removeChild(at: bucket, slot)
      node.insertItem(item, at: bucket)
      node._invariantCheck()
      return .node(level, node)
    case let .node(node):
      var copy = self.copy()
      _ = copy.replaceChild(at: bucket, slot, with: node)
      return .node(level, copy)
    case let .collisionNode(node):
      if hasSingletonChild {
        // Compression
        assert(!level.isAtBottom)
        return .collisionNode(level, node)
      }
      var copy = self.copy()
      _ = copy.replaceChild(at: bucket, slot, with: node)
      return .node(level, copy)
    }
  }
}
