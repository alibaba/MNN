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
  @inlinable
  internal mutating func insert(
    _ level: _HashLevel,
    _ item: Element,
    _ hash: _Hash
  ) -> (inserted: Bool, leaf: _UnmanagedHashNode, slot: _HashSlot) {
    insert(level, item.key, hash) { $0.initialize(to: item) }
  }

  @inlinable
  internal mutating func insert(
    _ level: _HashLevel,
    _ key: Key,
    _ hash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (inserted: Bool, leaf: _UnmanagedHashNode, slot: _HashSlot) {
    defer { _invariantCheck() }
    let isUnique = self.isUnique()
    if !isUnique {
      let r = self.inserting(level, key, hash, inserter)
      self = r.node
      return (r.inserted, r.leaf, r.slot)
    }
    let r = findForInsertion(level, key, hash)
    switch r {
    case .found(_, let slot):
      return (false, unmanaged, slot)
    case .insert(let bucket, let slot):
      ensureUniqueAndInsertItem(
        isUnique: true, at: bucket, itemSlot: slot, inserter)
      return (true, unmanaged, slot)
    case .appendCollision:
      let slot = ensureUniqueAndAppendCollision(isUnique: true, inserter)
      return (true, unmanaged, slot)
    case .spawnChild(let bucket, let slot):
      let r = ensureUniqueAndSpawnChild(
        isUnique: true,
        level: level,
        replacing: bucket,
        itemSlot: slot,
        newHash: hash,
        inserter)
      return (true, r.leaf, r.slot)
    case .expansion:
      let r = _HashNode.build(
        level: level,
        item1: inserter, hash,
        child2: self, self.collisionHash)
      self = r.top
      return (true, r.leaf, r.slot1)
    case .descend(_, let slot):
      let r = update {
        $0[child: slot].insert(level.descend(), key, hash, inserter)
      }
      if r.inserted { count &+= 1 }
      return r
    }
  }

  @inlinable
  internal func inserting(
    _ level: _HashLevel,
    _ item: __owned Element,
    _ hash: _Hash
  ) -> (
    inserted: Bool, node: _HashNode, leaf: _UnmanagedHashNode, slot: _HashSlot
  ) {
    inserting(level, item.key, hash, { $0.initialize(to: item) })
  }

  @inlinable
  internal func inserting(
    _ level: _HashLevel,
    _ key: Key,
    _ hash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (
    inserted: Bool, node: _HashNode, leaf: _UnmanagedHashNode, slot: _HashSlot
  ) {
    defer { _invariantCheck() }
    let r = findForInsertion(level, key, hash)
    switch r {
    case .found(_, let slot):
      return (false, self, unmanaged, slot)
    case .insert(let bucket, let slot):
      let node = copyNodeAndInsertItem(at: bucket, itemSlot: slot, inserter)
      return (true, node, node.unmanaged, slot)
    case .appendCollision:
      let r = copyNodeAndAppendCollision(inserter)
      return (true, r.node, r.node.unmanaged, r.slot)
    case .spawnChild(let bucket, let slot):
      let existingHash = read { _Hash($0[item: slot].key) }
      let r = copyNodeAndSpawnChild(
        level: level,
        replacing: bucket,
        itemSlot: slot,
        existingHash: existingHash,
        newHash: hash,
        inserter)
      return (true, r.node, r.leaf, r.slot)
    case .expansion:
      let r = _HashNode.build(
        level: level,
        item1: inserter, hash,
        child2: self, self.collisionHash)
      return (true, r.top, r.leaf, r.slot1)
    case .descend(_, let slot):
      let r = read {
        $0[child: slot].inserting(level.descend(), key, hash, inserter)
      }
      guard r.inserted else {
        return (false, self, r.leaf, r.slot)
      }
      var copy = self.copy()
      copy.update { $0[child: slot] = r.node }
      copy.count &+= 1
      return (true, copy, r.leaf, r.slot)
    }
  }

  @inlinable
  internal mutating func updateValue(
    _ level: _HashLevel,
    forKey key: Key,
    _ hash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (inserted: Bool, leaf: _UnmanagedHashNode, slot: _HashSlot) {
    defer { _invariantCheck() }
    let isUnique = self.isUnique()
    let r = findForInsertion(level, key, hash)
    switch r {
    case .found(_, let slot):
      ensureUnique(isUnique: isUnique)
      return (false, unmanaged, slot)
    case .insert(let bucket, let slot):
      ensureUniqueAndInsertItem(
        isUnique: isUnique, at: bucket, itemSlot: slot, inserter)
      return (true, unmanaged, slot)
    case .appendCollision:
      let slot = ensureUniqueAndAppendCollision(isUnique: isUnique, inserter)
      return (true, unmanaged, slot)
    case .spawnChild(let bucket, let slot):
      let r = ensureUniqueAndSpawnChild(
        isUnique: isUnique,
        level: level,
        replacing: bucket,
        itemSlot: slot,
        newHash: hash,
        inserter)
      return (true, r.leaf, r.slot)
    case .expansion:
      let r = _HashNode.build(
        level: level,
        item1: inserter, hash,
        child2: self, self.collisionHash)
      self = r.top
      return (true, r.leaf, r.slot1)
    case .descend(_, let slot):
      ensureUnique(isUnique: isUnique)
      let r = update {
        $0[child: slot].updateValue(
          level.descend(), forKey: key, hash, inserter)
      }
      if r.inserted { count &+= 1 }
      return r
    }
  }
}

extension _HashNode {
  @inlinable
  internal mutating func ensureUniqueAndInsertItem(
    isUnique: Bool,
    _ item: Element,
    at bucket: _Bucket
  ) {
    let slot = self.read { $0.itemMap.slot(of: bucket) }
    ensureUniqueAndInsertItem(
      isUnique: isUnique,
      at: bucket,
      itemSlot: slot
    ) {
      $0.initialize(to: item)
    }
  }

  @inlinable
  internal mutating func ensureUniqueAndInsertItem(
    isUnique: Bool,
    at bucket: _Bucket,
    itemSlot slot: _HashSlot,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) {
    assert(!isCollisionNode)

    if !isUnique {
      self = copyNodeAndInsertItem(at: bucket, itemSlot: slot, inserter)
      return
    }
    if !hasFreeSpace(Self.spaceForNewItem) {
      resizeNodeAndInsertItem(at: bucket, itemSlot: slot, inserter)
      return
    }
    // In-place insert.
    update {
      let p = $0._makeRoomForNewItem(at: slot, bucket)
      inserter(p)
    }
    self.count &+= 1
  }

  @inlinable @inline(never)
  internal func copyNodeAndInsertItem(
    at bucket: _Bucket,
    itemSlot slot: _HashSlot,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> _HashNode {
    assert(!isCollisionNode)
    let c = self.count
    return read { src in
      assert(!src.itemMap.contains(bucket))
      assert(!src.childMap.contains(bucket))
      return Self.allocate(
        itemMap: src.itemMap.inserting(bucket),
        childMap: src.childMap,
        count: c &+ 1
      ) { dstChildren, dstItems in
        dstChildren.initializeAll(fromContentsOf: src.children)

        let srcItems = src.reverseItems
        assert(dstItems.count == srcItems.count + 1)
        dstItems.suffix(slot.value)
          .initializeAll(fromContentsOf: srcItems.suffix(slot.value))
        let rest = srcItems.count &- slot.value
        dstItems.prefix(rest)
          .initializeAll(fromContentsOf: srcItems.prefix(rest))

        inserter(dstItems.baseAddress! + rest)
      }.node
    }
  }

  @inlinable @inline(never)
  internal mutating func resizeNodeAndInsertItem(
    at bucket: _Bucket,
    itemSlot slot: _HashSlot,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) {
    assert(!isCollisionNode)
    let c = self.count
    self = update { src in
      assert(!src.itemMap.contains(bucket))
      assert(!src.childMap.contains(bucket))
      return Self.allocate(
        itemMap: src.itemMap.inserting(bucket),
        childMap: src.childMap,
        count: c &+ 1
      ) { dstChildren, dstItems in
        dstChildren.moveInitializeAll(fromContentsOf: src.children)

        let srcItems = src.reverseItems
        assert(dstItems.count == srcItems.count + 1)
        dstItems.suffix(slot.value)
          .moveInitializeAll(fromContentsOf: srcItems.suffix(slot.value))
        let rest = srcItems.count &- slot.value
        dstItems.prefix(rest)
          .moveInitializeAll(fromContentsOf: srcItems.prefix(rest))

        inserter(dstItems.baseAddress! + rest)

        src.clear()
      }.node
    }
  }
}

extension _HashNode {
  @inlinable
  internal mutating func ensureUniqueAndAppendCollision(
    isUnique: Bool,
    _ item: Element
  ) -> _HashSlot {
    ensureUniqueAndAppendCollision(isUnique: isUnique) {
      $0.initialize(to: item)
    }
  }

  @inlinable
  internal mutating func ensureUniqueAndAppendCollision(
    isUnique: Bool,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> _HashSlot {
    assert(isCollisionNode)
    if !isUnique {
      let r = copyNodeAndAppendCollision(inserter)
      self = r.node
      return r.slot
    }
    if !hasFreeSpace(Self.spaceForNewItem) {
      return resizeNodeAndAppendCollision(inserter)
    }
    // In-place insert.
    update {
      let p = $0._makeRoomForNewItem(at: $0.itemsEndSlot, .invalid)
      inserter(p)
    }
    self.count &+= 1
    return _HashSlot(self.count &- 1)
  }

  @inlinable @inline(never)
  internal func copyNodeAndAppendCollision(
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (node: _HashNode, slot: _HashSlot) {
    assert(isCollisionNode)
    assert(self.count == read { $0.collisionCount })
    let c = self.count
    let node = read { src in
      Self.allocateCollision(count: c &+ 1, src.collisionHash) { dstItems in
        let srcItems = src.reverseItems
        assert(dstItems.count == srcItems.count + 1)
        dstItems.dropFirst().initializeAll(fromContentsOf: srcItems)
        inserter(dstItems.baseAddress!)
      }.node
    }
    return (node, _HashSlot(c))
  }

  @inlinable @inline(never)
  internal mutating func resizeNodeAndAppendCollision(
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> _HashSlot {
    assert(isCollisionNode)
    assert(self.count == read { $0.collisionCount })
    let c = self.count
    self = update { src in
      Self.allocateCollision(count: c &+ 1, src.collisionHash) { dstItems in
        let srcItems = src.reverseItems
        assert(dstItems.count == srcItems.count + 1)
        dstItems.dropFirst().moveInitializeAll(fromContentsOf: srcItems)
        inserter(dstItems.baseAddress!)

        src.clear()
      }.node
    }
    return _HashSlot(c)
  }
}

extension _HashNode {
  @inlinable
  internal func _copyNodeAndReplaceItemWithNewChild(
    level: _HashLevel,
    _ newChild: __owned _HashNode,
    at bucket: _Bucket,
    itemSlot: _HashSlot
  ) -> _HashNode {
    let c = self.count
    return read { src in
      assert(!src.isCollisionNode)
      assert(src.itemMap.contains(bucket))
      assert(!src.childMap.contains(bucket))
      assert(src.itemMap.slot(of: bucket) == itemSlot)

      if src.hasSingletonItem && newChild.isCollisionNode {
        // Compression
        return newChild
      }

      let childSlot = src.childMap.slot(of: bucket)
      return Self.allocate(
        itemMap: src.itemMap.removing(bucket),
        childMap: src.childMap.inserting(bucket),
        count: c &+ newChild.count &- 1
      ) { dstChildren, dstItems in
        let srcChildren = src.children
        let srcItems = src.reverseItems

        // Initialize children.
        dstChildren.prefix(childSlot.value)
          .initializeAll(fromContentsOf: srcChildren.prefix(childSlot.value))
        let rest = srcChildren.count &- childSlot.value
        dstChildren.suffix(rest)
          .initializeAll(fromContentsOf: srcChildren.suffix(rest))

        dstChildren.initializeElement(at: childSlot.value, to: newChild)

        // Initialize items.
        dstItems.suffix(itemSlot.value)
          .initializeAll(fromContentsOf: srcItems.suffix(itemSlot.value))
        let rest2 = dstItems.count &- itemSlot.value
        dstItems.prefix(rest2)
          .initializeAll(fromContentsOf: srcItems.prefix(rest2))
      }.node
    }
  }

  /// The item at `itemSlot` must have already been deinitialized by the time
  /// this function is called.
  @inlinable
  internal mutating func _resizeNodeAndReplaceItemWithNewChild(
    level: _HashLevel,
    _ newChild: __owned _HashNode,
    at bucket: _Bucket,
    itemSlot: _HashSlot
  ) {
    let c = self.count
    let node: _HashNode = update { src in
      assert(!src.isCollisionNode)
      assert(src.itemMap.contains(bucket))
      assert(!src.childMap.contains(bucket))
      assert(src.itemMap.slot(of: bucket) == itemSlot)

      let childSlot = src.childMap.slot(of: bucket)
      return Self.allocate(
        itemMap: src.itemMap.removing(bucket),
        childMap: src.childMap.inserting(bucket),
        count: c &+ newChild.count &- 1
      ) { dstChildren, dstItems in
        let srcChildren = src.children
        let srcItems = src.reverseItems

        // Initialize children.
        dstChildren.prefix(childSlot.value)
          .moveInitializeAll(fromContentsOf: srcChildren.prefix(childSlot.value))
        let rest = srcChildren.count &- childSlot.value
        dstChildren.suffix(rest)
          .moveInitializeAll(fromContentsOf: srcChildren.suffix(rest))

        dstChildren.initializeElement(at: childSlot.value, to: newChild)

        // Initialize items.
        dstItems.suffix(itemSlot.value)
          .moveInitializeAll(fromContentsOf: srcItems.suffix(itemSlot.value))
        let rest2 = dstItems.count &- itemSlot.value
        dstItems.prefix(rest2)
          .moveInitializeAll(fromContentsOf: srcItems.prefix(rest2))

        src.clear()
      }.node
    }
    self = node
  }
}

extension _HashNode {
  @inlinable @inline(never)
  internal func copyNodeAndPushItemIntoNewChild(
    level: _HashLevel,
    _ newChild: __owned _HashNode,
    at bucket: _Bucket,
    itemSlot: _HashSlot
  ) -> _HashNode {
    assert(!isCollisionNode)
    let item = read { $0[item: itemSlot] }
    let hash = _Hash(item.key)
    let r = newChild.inserting(level, item, hash)
    return _copyNodeAndReplaceItemWithNewChild(
      level: level,
      r.node,
      at: bucket,
      itemSlot: itemSlot)
  }
}

extension _HashNode {
  @inlinable
  internal mutating func ensureUniqueAndSpawnChild(
    isUnique: Bool,
    level: _HashLevel,
    replacing bucket: _Bucket,
    itemSlot: _HashSlot,
    newHash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (leaf: _UnmanagedHashNode, slot: _HashSlot) {
    let existingHash = read { _Hash($0[item: itemSlot].key) }
    assert(existingHash.isEqual(to: newHash, upTo: level))
    if newHash == existingHash, hasSingletonItem {
      // Convert current node to a collision node.
      self = _HashNode._collisionNode(newHash, read { $0[item: .zero] }, inserter)
      return (unmanaged, _HashSlot(1))
    }

    if !isUnique {
      let r = copyNodeAndSpawnChild(
        level: level,
        replacing: bucket,
        itemSlot: itemSlot,
        existingHash: existingHash,
        newHash: newHash,
        inserter)
      self = r.node
      return (r.leaf, r.slot)
    }
    if !hasFreeSpace(Self.spaceForSpawningChild) {
      return resizeNodeAndSpawnChild(
        level: level,
        replacing: bucket,
        itemSlot: itemSlot,
        existingHash: existingHash,
        newHash: newHash,
        inserter)
    }

    let existing = removeItem(at: bucket, itemSlot)
    let r = _HashNode.build(
      level: level.descend(),
      item1: existing, existingHash,
      item2: inserter, newHash)
    insertChild(r.top, bucket)
    return (r.leaf, r.slot2)
  }

  @inlinable @inline(never)
  internal func copyNodeAndSpawnChild(
    level: _HashLevel,
    replacing bucket: _Bucket,
    itemSlot: _HashSlot,
    existingHash: _Hash,
    newHash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (node: _HashNode, leaf: _UnmanagedHashNode, slot: _HashSlot) {
    let r = read {
      _HashNode.build(
        level: level.descend(),
        item1: $0[item: itemSlot], existingHash,
        item2: inserter, newHash)
    }
    let node = _copyNodeAndReplaceItemWithNewChild(
      level: level,
      r.top,
      at: bucket,
      itemSlot: itemSlot)
    node._invariantCheck()
    return (node, r.leaf, r.slot2)
  }

  @inlinable @inline(never)
  internal mutating func resizeNodeAndSpawnChild(
    level: _HashLevel,
    replacing bucket: _Bucket,
    itemSlot: _HashSlot,
    existingHash: _Hash,
    newHash: _Hash,
    _ inserter: (UnsafeMutablePointer<Element>) -> Void
  ) -> (leaf: _UnmanagedHashNode, slot: _HashSlot) {
    let r = update {
      _HashNode.build(
        level: level.descend(),
        item1: $0.itemPtr(at: itemSlot).move(), existingHash,
        item2: inserter, newHash)
    }
    _resizeNodeAndReplaceItemWithNewChild(
      level: level,
      r.top,
      at: bucket,
      itemSlot: itemSlot)
    _invariantCheck()
    return (r.leaf, r.slot2)
  }
}

