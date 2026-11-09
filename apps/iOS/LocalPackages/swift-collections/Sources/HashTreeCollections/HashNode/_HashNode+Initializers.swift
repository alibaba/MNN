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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension _HashNode {
  @inlinable @inline(__always)
  internal static func _emptyNode() -> _HashNode {
    _HashNode(storage: _emptySingleton, count: 0)
  }

  @inlinable
  internal static func _collisionNode(
    _ hash: _Hash,
    _ item1: __owned Element,
    _ item2: __owned Element
  ) -> _HashNode {
    let node = _HashNode.allocateCollision(count: 2, hash) { items in
      items.initializeElement(at: 1, to: item1)
      items.initializeElement(at: 0, to: item2)
    }.node
    node._invariantCheck()
    return node
  }

  @inlinable
  internal static func _collisionNode(
    _ hash: _Hash,
    _ item1: __owned Element,
    _ inserter2: (UnsafeMutablePointer<Element>) -> Void
  ) -> _HashNode {
    let node = _HashNode.allocateCollision(count: 2, hash) { items in
      items.initializeElement(at: 1, to: item1)
      inserter2(items.baseAddress.unsafelyUnwrapped)
    }.node
    node._invariantCheck()
    return node
  }

  @inlinable
  internal static func _regularNode(
    _ item: __owned Element,
    _ bucket: _Bucket
  ) -> _HashNode {
    let r = _HashNode.allocate(
      itemMap: _Bitmap(bucket),
      childMap: .empty,
      count: 1
    ) { children, items in
      assert(items.count == 1 && children.count == 0)
      items.initializeElement(at: 0, to: item)
    }
    r.node._invariantCheck()
    return r.node
  }

  @inlinable
  internal static func _regularNode(
    _ item1: __owned Element,
    _ bucket1: _Bucket,
    _ item2: __owned Element,
    _ bucket2: _Bucket
  ) -> _HashNode {
    _regularNode(
      item1, bucket1,
      { $0.initialize(to: item2) }, bucket2).node
  }

  @inlinable
  internal static func _regularNode(
    _ item1: __owned Element,
    _ bucket1: _Bucket,
    _ inserter2: (UnsafeMutablePointer<Element>) -> Void,
    _ bucket2: _Bucket
  ) -> (node: _HashNode, slot1: _HashSlot, slot2: _HashSlot) {
    assert(bucket1 != bucket2)
    let r = _HashNode.allocate(
      itemMap: _Bitmap(bucket1, bucket2),
      childMap: .empty,
      count: 2
    ) { children, items -> (_HashSlot, _HashSlot) in
      assert(items.count == 2 && children.count == 0)
      let i1 = bucket1 < bucket2 ? 1 : 0
      let i2 = 1 &- i1
      items.initializeElement(at: i1, to: item1)
      inserter2(items.baseAddress.unsafelyUnwrapped + i2)
      return (_HashSlot(i2), _HashSlot(i1)) // Note: swapped
    }
    r.node._invariantCheck()
    return (r.node, r.result.0, r.result.1)
  }

  @inlinable
  internal static func _regularNode(
    _ child: __owned _HashNode,
    _ bucket: _Bucket
  ) -> _HashNode {
    let r = _HashNode.allocate(
      itemMap: .empty,
      childMap: _Bitmap(bucket),
      count: child.count
    ) { children, items in
      assert(items.count == 0 && children.count == 1)
      children.initializeElement(at: 0, to: child)
    }
    r.node._invariantCheck()
    return r.node
  }

  @inlinable
  internal static func _regularNode(
    _ item: __owned Element,
    _ itemBucket: _Bucket,
    _ child: __owned _HashNode,
    _ childBucket: _Bucket
  ) -> _HashNode {
    _regularNode(
      { $0.initialize(to: item) }, itemBucket,
      child, childBucket)
  }

  @inlinable
  internal static func _regularNode(
    _ inserter: (UnsafeMutablePointer<Element>) -> Void,
    _ itemBucket: _Bucket,
    _ child: __owned _HashNode,
    _ childBucket: _Bucket
  ) -> _HashNode {
    assert(itemBucket != childBucket)
    let r = _HashNode.allocate(
      itemMap: _Bitmap(itemBucket),
      childMap: _Bitmap(childBucket),
      count: child.count &+ 1
    ) { children, items in
      assert(items.count == 1 && children.count == 1)
      inserter(items.baseAddress.unsafelyUnwrapped)
      children.initializeElement(at: 0, to: child)
    }
    r.node._invariantCheck()
    return r.node
  }

  @inlinable
  internal static func _regularNode(
    _ child1: __owned _HashNode,
    _ child1Bucket: _Bucket,
    _ child2: __owned _HashNode,
    _ child2Bucket: _Bucket
  ) -> _HashNode {
    assert(child1Bucket != child2Bucket)
    let r = _HashNode.allocate(
      itemMap: .empty,
      childMap: _Bitmap(child1Bucket, child2Bucket),
      count: child1.count &+ child2.count
    ) { children, items in
      assert(items.count == 0 && children.count == 2)
      children.initializeElement(at: 0, to: child1)
      children.initializeElement(at: 1, to: child2)
    }
    r.node._invariantCheck()
    return r.node
  }
}

extension _HashNode {
  @inlinable
  internal static func build(
    level: _HashLevel,
    item1: __owned Element,
    _ hash1: _Hash,
    item2 inserter2: (UnsafeMutablePointer<Element>) -> Void,
    _ hash2: _Hash
  ) -> (top: _HashNode, leaf: _UnmanagedHashNode, slot1: _HashSlot, slot2: _HashSlot) {
    assert(hash1.isEqual(to: hash2, upTo: level.ascend()))
    if hash1 == hash2 {
      let top = _collisionNode(hash1, item1, inserter2)
      return (top, top.unmanaged, _HashSlot(0), _HashSlot(1))
    }
    let r = _build(
      level: level, item1: item1, hash1, item2: inserter2, hash2)
    return (r.top, r.leaf, r.slot1, r.slot2)
  }

  @inlinable
  internal static func _build(
    level: _HashLevel,
    item1: __owned Element,
    _ hash1: _Hash,
    item2 inserter2: (UnsafeMutablePointer<Element>) -> Void,
    _ hash2: _Hash
  ) -> (top: _HashNode, leaf: _UnmanagedHashNode, slot1: _HashSlot, slot2: _HashSlot) {
    assert(hash1 != hash2)
    let b1 = hash1[level]
    let b2 = hash2[level]
    guard b1 == b2 else {
      let r = _regularNode(item1, b1, inserter2, b2)
      return (r.node, r.node.unmanaged, r.slot1, r.slot2)
    }
    let r = _build(
      level: level.descend(),
      item1: item1, hash1,
      item2: inserter2, hash2)
    return (_regularNode(r.top, b1), r.leaf, r.slot1, r.slot2)
  }

  @inlinable
  internal static func build(
    level: _HashLevel,
    item1 inserter1: (UnsafeMutablePointer<Element>) -> Void,
    _ hash1: _Hash,
    child2: __owned _HashNode,
    _ hash2: _Hash
  ) -> (top: _HashNode, leaf: _UnmanagedHashNode, slot1: _HashSlot, slot2: _HashSlot) {
    assert(child2.isCollisionNode)
    assert(hash1 != hash2)
    let b1 = hash1[level]
    let b2 = hash2[level]
    if b1 == b2 {
      let node = build(
        level: level.descend(),
        item1: inserter1, hash1,
        child2: child2, hash2)
      return (_regularNode(node.top, b1), node.leaf, node.slot1, node.slot2)
    }
    let node = _regularNode(inserter1, hash1[level], child2, hash2[level])
    return (node, node.unmanaged, .zero, .zero)
  }

  @inlinable
  internal static func build(
    level: _HashLevel,
    child1: __owned _HashNode,
    _ hash1: _Hash,
    child2: __owned _HashNode,
    _ hash2: _Hash
  ) -> _HashNode {
    assert(child1.isCollisionNode)
    assert(child2.isCollisionNode)
    assert(hash1 != hash2)
    let b1 = hash1[level]
    let b2 = hash2[level]
    guard b1 == b2 else {
      return _regularNode(child1, b1, child2, b2)
    }
    let node = build(
      level: level.descend(),
      child1: child1, hash1,
      child2: child2, hash2)
    return _regularNode(node, b1)
  }
}
