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
  /// Returns true if `self` contains a disjoint set of keys than `other`.
  /// Otherwise, returns false.
  @inlinable @inline(never)
  internal func isDisjoint<Value2>(
    _ level: _HashLevel,
    with other: _HashNode<Key, Value2>
  ) -> Bool {
    if self.count == 0 || other.count == 0 { return true }
    if self.raw.storage === other.raw.storage { return false }

    if self.isCollisionNode {
      return _isDisjointCollision(level, with: other)
    }
    if other.isCollisionNode {
      return other._isDisjointCollision(level, with: self)
    }

    return self.read { l in
      other.read { r in
        let lmap = l.itemMap.union(l.childMap)
        let rmap = r.itemMap.union(r.childMap)
        if lmap.isDisjoint(with: rmap) { return true }

        for (bucket, _) in l.itemMap.intersection(r.itemMap) {
          let lslot = l.itemMap.slot(of: bucket)
          let rslot = r.itemMap.slot(of: bucket)
          guard l[item: lslot].key != r[item: rslot].key else { return false }
        }
        for (bucket, _) in l.itemMap.intersection(r.childMap) {
          let lslot = l.itemMap.slot(of: bucket)
          let hash = _Hash(l[item: lslot].key)
          let rslot = r.childMap.slot(of: bucket)
          let found = r[child: rslot].containsKey(
            level.descend(),
            l[item: lslot].key,
            hash)
          if found { return false }
        }
        for (bucket, _) in l.childMap.intersection(r.itemMap) {
          let lslot = l.childMap.slot(of: bucket)
          let rslot = r.itemMap.slot(of: bucket)
          let hash = _Hash(r[item: rslot].key)
          let found = l[child: lslot].containsKey(
            level.descend(),
            r[item: rslot].key,
            hash)
          if found { return false }
        }
        for (bucket, _) in l.childMap.intersection(r.childMap) {
          let lslot = l.childMap.slot(of: bucket)
          let rslot = r.childMap.slot(of: bucket)
          guard
            l[child: lslot].isDisjoint(level.descend(), with: r[child: rslot])
          else { return false }
        }
        return true
      }
    }
  }

  @inlinable @inline(never)
  internal func _isDisjointCollision<Value2>(
    _ level: _HashLevel,
    with other: _HashNode<Key, Value2>
  ) -> Bool {
    assert(isCollisionNode)
    if other.isCollisionNode {
      return read { l in
        other.read { r in
          guard l.collisionHash == r.collisionHash else { return true }
          let litems = l.reverseItems
          let ritems = r.reverseItems
          return litems.allSatisfy { li in
            !ritems.contains { ri in li.key == ri.key }
          }
        }
      }
    }
    // `self` is on a compressed path. Try descending down by one level.
    assert(!level.isAtBottom)
    let bucket = self.collisionHash[level]
    return other.read { r in
      if r.childMap.contains(bucket) {
        let slot = r.childMap.slot(of: bucket)
        return isDisjoint(level.descend(), with: r[child: slot])
      }
      if r.itemMap.contains(bucket) {
        let rslot = r.itemMap.slot(of: bucket)
        let p = r.itemPtr(at: rslot)
        let hash = _Hash(p.pointee.key)
        return read { l in
          guard hash == l.collisionHash else { return true }
          return !l.reverseItems.contains { $0.key == p.pointee.key }
        }
      }
      return true
    }
  }
}
