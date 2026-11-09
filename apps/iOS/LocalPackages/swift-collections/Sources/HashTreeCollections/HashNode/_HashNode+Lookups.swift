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

// MARK: Node-level lookup operations

extension _HashNode {
  @inlinable
  internal func find(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> (descend: Bool, slot: _HashSlot)? {
    read { $0.find(level, key, hash) }
  }
}

extension _HashNode.UnsafeHandle {
  @inlinable
  internal func find(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> (descend: Bool, slot: _HashSlot)? {
    guard !isCollisionNode else {
      let r = _findInCollision(level, key, hash)
      guard r.code == 0 else { return nil }
      return (false, r.slot)
    }
    let bucket = hash[level]
    if itemMap.contains(bucket) {
      let slot = itemMap.slot(of: bucket)
      guard self[item: slot].key == key else { return nil }
      return (false, slot)
    }
    if childMap.contains(bucket) {
      let slot = childMap.slot(of: bucket)
      return (true, slot)
    }
    return nil
  }

  @inlinable @inline(never)
  internal func _findInCollision(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> (code: Int, slot: _HashSlot) {
    assert(isCollisionNode)
    if !level.isAtBottom {
      if hash != self.collisionHash { return (2, .zero) }
    }
    // Note: this searches the items in reverse insertion order.
    guard let slot = reverseItems.firstIndex(where: { $0.key == key })
    else { return (1, self.itemsEndSlot) }
    return (0, _HashSlot(itemCount &- 1 &- slot))
  }
}


/// Represents the results of a lookup operation within a single node of a hash
/// tree. This enumeration captures all of the different cases that need to be
/// covered if we wanted to insert a new item into the tree.
///
/// For simple read-only lookup operations (and removals) some of the cases are
/// equivalent: `.notFound`, .newCollision` and `expansion` all represent the
/// same logical outcome: the key we're looking for is not present in this
/// subtree.
@usableFromInline
@frozen
internal enum _FindResult {
  /// The item we're looking for is stored directly in this node, at the
  /// bucket / item slot identified in the payload.
  ///
  /// If the current node is a collision node, then the bucket value is
  /// set to `_Bucket.invalid`.
  case found(_Bucket, _HashSlot)

  /// The item we're looking for is not currently inside the subtree rooted at
  /// this node.
  ///
  /// If we wanted to insert it, then its correct slot is within this node
  /// at the specified bucket / item slot. (Which is currently empty.)
  ///
  /// When the node is a collision node, the `insertCollision` case is returned
  /// instead of this one.
  case insert(_Bucket, _HashSlot)

  /// The item we're looking for is not currently inside the subtree rooted at
  /// this collision node.
  ///
  /// If we wanted to insert it, then it needs to be appended to the items
  /// buffer.
  case appendCollision

  /// The item we're looking for is not currently inside the subtree rooted at
  /// this node.
  ///
  /// If we wanted to insert it, then it would need to be stored in this node
  /// at the specified bucket / item slot. However, that bucket is already
  /// occupied by another item, so the insertion would need to involve replacing
  /// it with a new child node.
  ///
  /// (This case is never returned if the current node is a collision node.)
  case spawnChild(_Bucket, _HashSlot)

  /// The item we're looking for is not in this subtree.
  ///
  /// However, the item doesn't belong in this subtree at all. This is an
  /// irregular case that can only happen with (compressed) hash collision nodes
  /// whose (otherwise empty) ancestors got eliminated, so they appear further
  /// up in the tree than what their (logical) level would indicate.
  ///
  /// If we wanted to insert a new item with this key, then we'd need to create
  /// (one or more) new parent nodes above this node, pushing this collision
  /// node further down the tree. (This undoes the compression by expanding
  /// the collision node's path, hence the name of the enum case.)
  ///
  /// (This case is never returned if the current node is a regular node.)
  case expansion

  /// The item we're looking for is not directly stored in this node, but it
  /// might be somewhere in the subtree rooted at the child at the given
  /// bucket & slot.
  ///
  /// (This case is never returned if the current node is a collision node.)
  case descend(_Bucket, _HashSlot)
}

extension _HashNode {
  @inlinable
  internal func findForInsertion(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> _FindResult {
    read { $0.findForInsertion(level, key, hash) }
  }
}

extension _HashNode.UnsafeHandle {
  @inlinable
  internal func findForInsertion(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> _FindResult {
    guard !isCollisionNode else {
      let r = _findInCollision(level, key, hash)
      if r.code == 0 {
        return .found(.invalid, r.slot)
      }
      if r.code == 1 {
        return .appendCollision
      }
      assert(r.code == 2)
      return .expansion
    }
    let bucket = hash[level]
    if itemMap.contains(bucket) {
      let slot = itemMap.slot(of: bucket)
      if self[item: slot].key == key {
        return .found(bucket, slot)
      }
      return .spawnChild(bucket, slot)
    }
    if childMap.contains(bucket) {
      let slot = childMap.slot(of: bucket)
      return .descend(bucket, slot)
    }
    let slot = itemMap.slot(of: bucket)
    return .insert(bucket, slot)
  }
}

// MARK: Subtree-level lookup operations

extension _HashNode {
  @inlinable
  internal func get(_ level: _HashLevel, _ key: Key, _ hash: _Hash) -> Value? {
    var node = unmanaged
    var level = level
    while true {
      let r = UnsafeHandle.read(node) { $0.find(level, key, hash) }
      guard let r = r else {
        return nil
      }
      guard r.descend else {
        return UnsafeHandle.read(node) { $0[item: r.slot].value }
      }
      node = node.unmanagedChild(at: r.slot)
      level = level.descend()
    }
  }
}

extension _HashNode {
  @inlinable
  internal func containsKey(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> Bool {
    var node = unmanaged
    var level = level
    while true {
      let r = UnsafeHandle.read(node) { $0.find(level, key, hash) }
      guard let r = r else { return false }
      guard r.descend else { return true }
      node = node.unmanagedChild(at: r.slot)
      level = level.descend()
    }
  }
}

extension _HashNode {
  @inlinable
  internal func lookup(
    _ level: _HashLevel, _ key: Key, _ hash: _Hash
  ) -> (node: _UnmanagedHashNode, slot: _HashSlot)? {
    var node = unmanaged
    var level = level
    while true {
      let r = UnsafeHandle.read(node) { $0.find(level, key, hash) }
      guard let r = r else {
        return nil
      }
      guard r.descend else {
        return (node, r.slot)
      }
      node = node.unmanagedChild(at: r.slot)
      level = level.descend()
    }
  }
}

extension _HashNode {
  @inlinable
  internal func position(
    forKey key: Key, _ level: _HashLevel, _ hash: _Hash
  ) -> Int? {
    guard let r = find(level, key, hash) else { return nil }
    guard r.descend else { return r.slot.value }
    return read { h in
      let children = h.children
      let p = children[r.slot.value]
        .position(forKey: key, level.descend(), hash)
      guard let p = p else { return nil }
      let c = h.itemCount &+ p
      return children[..<r.slot.value].reduce(into: c) { $0 &+= $1.count }
    }
  }

  @inlinable
  internal func item(position: Int) -> Element {
    assert(position >= 0 && position < self.count)
    return read {
      var itemsToSkip = position
      let itemCount = $0.itemCount
      if itemsToSkip < itemCount {
        return $0[item: _HashSlot(itemsToSkip)]
      }
      itemsToSkip -= itemCount
      let children = $0.children
      for i in children.indices {
        if itemsToSkip < children[i].count {
          return children[i].item(position: itemsToSkip)
        }
        itemsToSkip -= children[i].count
      }
      fatalError("Inconsistent tree")
    }
  }
}
