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
  /// - Returns: The number of new items added to `self`.
  @inlinable
  internal mutating func merge(
  _ level: _HashLevel,
  _ other: _HashNode,
  _ combine: (Value, Value) throws -> Value
  ) rethrows -> Int {
    guard other.count > 0 else { return 0 }
    guard self.count > 0 else {
      self = other
      return self.count
    }
    if level.isAtRoot, self.hasSingletonItem {
      // In this special case, the root node may turn into a collision node
      // during the merge process. Prevent this from causing issues below by
      // handling it up front.
      var copy = other
      let delta = try self.read { l in
        let lp = l.itemPtr(at: .zero)
        let c = copy.count
        let res = copy.updateValue(
          level, forKey: lp.pointee.key, _Hash(lp.pointee.key)
        ) {
          $0.initialize(to: lp.pointee)
        }
        if !res.inserted {
          try UnsafeHandle.update(res.leaf) {
            let p = $0.itemPtr(at: res.slot)
            p.pointee.value = try combine(lp.pointee.value, p.pointee.value)
          }
        }
        return c - (res.inserted ? 0 : 1)
      }
      self = copy
      return delta
    }

    return try _merge(level, other, combine)
  }

  @inlinable
  internal mutating func _merge(
    _ level: _HashLevel,
    _ other: _HashNode,
    _ combine: (Value, Value) throws -> Value
  ) rethrows -> Int {
    // Note: don't check storage identities -- we do need to merge the contents
    // of identical nodes.

    if self.isCollisionNode || other.isCollisionNode {
      return try _merge_slow(level, other, combine)
    }

    return try other.read { r in
      var isUnique = self.isUnique()
      var delta = 0

      let (originalItems, originalChildren) = self.read {
        ($0.itemMap, $0.childMap)
      }

      for (bucket, _) in originalItems {
        assert(!isCollisionNode)
        if r.itemMap.contains(bucket) {
          let rslot = r.itemMap.slot(of: bucket)
          let rp = r.itemPtr(at: rslot)
          let lslot = self.read { $0.itemMap.slot(of: bucket) }
          let conflict = self.read { $0[item: lslot].key == rp.pointee.key }
          if conflict {
            self.ensureUnique(isUnique: isUnique)
            try self.update {
              let p = $0.itemPtr(at: lslot)
              p.pointee.value = try combine(p.pointee.value, rp.pointee.value)
            }
          } else {
            _ = self.ensureUniqueAndSpawnChild(
              isUnique: isUnique,
              level: level,
              replacing: bucket,
              itemSlot: lslot,
              newHash: _Hash(rp.pointee.key),
              { $0.initialize(to: rp.pointee) })
            // If we hadn't handled the singleton root node case above,
            // then this call would sometimes turn `self` into a collision
            // node on a compressed path, causing mischief.
            assert(!self.isCollisionNode)
            delta &+= 1
          }
          isUnique = true
        }
        else if r.childMap.contains(bucket) {
          let rslot = r.childMap.slot(of: bucket)
          let rp = r.childPtr(at: rslot)

          self.ensureUnique(
            isUnique: isUnique, withFreeSpace: _HashNode.spaceForSpawningChild)
          let item = self.removeItem(at: bucket)
          delta &-= 1
          var child = rp.pointee
          let r = child.updateValue(
            level.descend(), forKey: item.key, _Hash(item.key)
          ) {
            $0.initialize(to: item)
          }
          if !r.inserted {
            try UnsafeHandle.update(r.leaf) {
              let p = $0.itemPtr(at: r.slot)
              p.pointee.value = try combine(item.value, p.pointee.value)
            }
          }
          self.insertChild(child, bucket)
          isUnique = true
          delta &+= child.count
        }
      }

      for (bucket, _) in originalChildren {
        assert(!isCollisionNode)
        let lslot = self.read { $0.childMap.slot(of: bucket) }

        if r.itemMap.contains(bucket) {
          let rslot = r.itemMap.slot(of: bucket)
          let rp = r.itemPtr(at: rslot)
          self.ensureUnique(isUnique: isUnique)
          let h = _Hash(rp.pointee.key)
          let res = self.update { l in
            l[child: lslot].updateValue(
              level.descend(), forKey: rp.pointee.key, h
            ) {
              $0.initialize(to: rp.pointee)
            }
          }
          if res.inserted {
            self.count &+= 1
            delta &+= 1
          } else {
            try UnsafeHandle.update(res.leaf) {
              let p = $0.itemPtr(at: res.slot)
              p.pointee.value = try combine(p.pointee.value, rp.pointee.value)
            }
          }
          isUnique = true
        }
        else if r.childMap.contains(bucket) {
          let rslot = r.childMap.slot(of: bucket)
          self.ensureUnique(isUnique: isUnique)
          let d = try self.update { l in
            try l[child: lslot].merge(
              level.descend(),
              r[child: rslot],
              combine)
          }
          self.count &+= d
          delta &+= d
          isUnique = true
        }
      }

      assert(!self.isCollisionNode)

      /// Add buckets in `other` that we haven't processed above.
      let seen = self.read { l in l.itemMap.union(l.childMap) }
      for (bucket, _) in r.itemMap.subtracting(seen) {
        let rslot = r.itemMap.slot(of: bucket)
        self.ensureUniqueAndInsertItem(
          isUnique: isUnique, r[item: rslot], at: bucket)
        delta &+= 1
        isUnique = true
      }
      for (bucket, _) in r.childMap.subtracting(seen) {
        let rslot = r.childMap.slot(of: bucket)
        self.ensureUnique(
          isUnique: isUnique, withFreeSpace: _HashNode.spaceForNewChild)
        self.insertChild(r[child: rslot], bucket)
        delta &+= r[child: rslot].count
        isUnique = true
      }

      assert(isUnique)
      return delta
    }
  }

  @inlinable @inline(never)
  internal mutating func _merge_slow(
    _ level: _HashLevel,
    _ other: _HashNode,
    _ combine: (Value, Value) throws -> Value
  ) rethrows -> Int {
    let lc = self.isCollisionNode
    let rc = other.isCollisionNode
    if lc && rc {
      guard self.collisionHash == other.collisionHash else {
        self = _HashNode.build(
          level: level,
          child1: self, self.collisionHash,
          child2: other, other.collisionHash)
        return other.count
      }
      return try other.read { r in
        var isUnique = self.isUnique()
        var delta = 0
        let originalItemCount = self.count
        for rs: _HashSlot in stride(from: .zero, to: r.itemsEndSlot, by: 1) {
          let rp = r.itemPtr(at: rs)
          let lslot: _HashSlot? = self.read { l in
            let litems = l.reverseItems
            return litems
              .suffix(originalItemCount)
              .firstIndex { $0.key == rp.pointee.key }
              .map { _HashSlot(litems.count &- 1 &- $0) }
          }
          if let lslot = lslot {
            self.ensureUnique(isUnique: isUnique)
            try self.update {
              let p = $0.itemPtr(at: lslot)
              p.pointee.value = try combine(p.pointee.value, rp.pointee.value)
            }
          } else {
            _ = self.ensureUniqueAndAppendCollision(
              isUnique: isUnique, rp.pointee)
            delta &+= 1
          }
          isUnique = true
        }
        return delta
      }
    }

    // One of the nodes must be on a compressed path.
    assert(!level.isAtBottom)

    if lc {
      // `self` is a collision node on a compressed path. The other tree might
      // have the same set of collisions, just expanded a bit deeper.
      return try other.read { r in
        let bucket = self.collisionHash[level]
        if r.itemMap.contains(bucket) {
          let rslot = r.itemMap.slot(of: bucket)
          let rp = r.itemPtr(at: rslot)

          let h = _Hash(rp.pointee.key)
          let res = self.updateValue(
            level.descend(), forKey: rp.pointee.key, h
          ) {
            $0.initialize(to: rp.pointee)
          }
          if !res.inserted {
            try UnsafeHandle.update(res.leaf) {
              let p = $0.itemPtr(at: res.slot)
              p.pointee.value = try combine(p.pointee.value, rp.pointee.value)
            }
          }
          self = other._copyNodeAndReplaceItemWithNewChild(
            level: level, self, at: bucket, itemSlot: rslot)
          return other.count - (res.inserted ? 0 : 1)
        }

        if r.childMap.contains(bucket) {
          let originalCount = self.count
          let rslot = r.childMap.slot(of: bucket)
          _ = try self._merge(level.descend(), r[child: rslot], combine)
          var node = other.copy()
          _ = node.replaceChild(at: bucket, rslot, with: self)
          self = node
          return self.count - originalCount
        }

        var node = other.copy(withFreeSpace: _HashNode.spaceForNewChild)
        node.insertChild(self, bucket)
        self = node
        return other.count
      }
    }

    assert(rc)
    let isUnique = self.isUnique()
    // `other` is a collision node on a compressed path.
    return try other.read { r in
      let bucket = r.collisionHash[level]
      if self.read({ $0.itemMap.contains(bucket) }) {
        self.ensureUnique(
          isUnique: isUnique, withFreeSpace: _HashNode.spaceForSpawningChild)
        let item = self.removeItem(at: bucket)
        let h = _Hash(item.key)
        var copy = other
        let res = copy.updateValue(level.descend(), forKey: item.key, h) {
          $0.initialize(to: item)
        }
        if !res.inserted {
          try UnsafeHandle.update(res.leaf) {
            let p = $0.itemPtr(at: res.slot)
            p.pointee.value = try combine(item.value, p.pointee.value)
          }
        }
        assert(self.count > 0) // Singleton case handled up front above
        self.insertChild(copy, bucket)
        return other.count - (res.inserted ? 0 : 1)
      }
      if self.read({ $0.childMap.contains(bucket) }) {
        self.ensureUnique(isUnique: isUnique)
        let delta: Int = try self.update { l in
          let lslot = l.childMap.slot(of: bucket)
          let lchild = l.childPtr(at: lslot)
          return try lchild.pointee._merge(level.descend(), other, combine)
        }
        assert(delta >= 0)
        self.count &+= delta
        return delta
      }
      self.ensureUnique(
        isUnique: isUnique, withFreeSpace: _HashNode.spaceForNewChild)
      self.insertChild(other, bucket)
      return other.count
    }
  }
}
