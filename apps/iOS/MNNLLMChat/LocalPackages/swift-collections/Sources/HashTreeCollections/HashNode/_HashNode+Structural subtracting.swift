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
  internal func subtracting<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode? {
    assert(level.isAtRoot)
    let builder = _subtracting(level, other)
    guard let builder = builder else { return nil }
    let root = builder.finalize(.top)
    root._fullInvariantCheck()
    return root
  }

  @inlinable
  internal func _subtracting<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> Builder? {
    if self.raw.storage === other.raw.storage { return .empty(level) }

    if self.isCollisionNode || other.isCollisionNode {
      return _subtracting_slow(level, other)
    }

    return self.read { l in
      other.read { r in
        var result: Builder = .empty(level)
        var removing = false

        for (bucket, lslot) in l.itemMap {
          let lp = l.itemPtr(at: lslot)
          let include: Bool
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            include = (lp.pointee.key != r[item: rslot].key)
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let h = _Hash(lp.pointee.key)
            include = !r[child: rslot]
              .containsKey(level.descend(), lp.pointee.key, h)
          }
          else {
            include = true
          }

          if include, removing {
            result.addNewItem(level, lp.pointee, at: bucket)
          }
          else if !include, !removing {
            removing = true
            result.copyItems(level, from: l, upTo: bucket)
          }
        }

        for (bucket, lslot) in l.childMap {
          var done = false
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            let rp = r.itemPtr(at: rslot)
            let h = _Hash(rp.pointee.key)
            let child = l[child: lslot]
                .removing(level.descend(), rp.pointee.key, h)?.replacement
            if let child = child {
              assert(child.count < self.count)
              if !removing {
                removing = true
                result.copyItemsAndChildren(level, from: l, upTo: bucket)
              }
              result.addNewChildBranch(level, child, at: bucket)
              done = true
            }
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let child = l[child: lslot]
              ._subtracting(level.descend(), r[child: rslot])
            if let child = child {
              assert(child.count < self.count)
              if !removing {
                removing = true
                result.copyItemsAndChildren(level, from: l, upTo: bucket)
              }
              result.addNewChildBranch(level, child, at: bucket)
              done = true
            }
          }
          if !done, removing {
            result.addNewChildNode(level, l[child: lslot], at: bucket)
          }
        }
        guard removing else { return nil }
        return result
      }
    }
  }

  @inlinable @inline(never)
  internal func _subtracting_slow<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> Builder? {
    let lc = self.isCollisionNode
    let rc = other.isCollisionNode
    if lc && rc {
      return read { l in
        other.read { r in
          guard l.collisionHash == r.collisionHash else {
            return nil
          }
          var result: Builder = .empty(level)
          var removing = false

          let ritems = r.reverseItems
          for lslot: _HashSlot in stride(from: .zero, to: l.itemsEndSlot, by: 1) {
            let lp = l.itemPtr(at: lslot)
            let include = !ritems.contains { $0.key == lp.pointee.key }
            if include, removing {
              result.addNewCollision(level, lp.pointee, l.collisionHash)
            }
            else if !include, !removing {
              removing = true
              result.copyCollisions(from: l, upTo: lslot)
            }
          }
          guard removing else { return nil }
          assert(result.count < self.count)
          return result
        }
      }
    }

    // One of the nodes must be on a compressed path.
    assert(!level.isAtBottom)

    if lc {
      // `self` is a collision node on a compressed path. The other tree might
      // have the same set of collisions, just expanded a bit deeper.
      return read { l in
        other.read { r in
          let bucket = l.collisionHash[level]
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            let ritem = r.itemPtr(at: rslot)
            let h = _Hash(ritem.pointee.key)
            let res = l.find(level, ritem.pointee.key, h)
            guard let res = res else { return nil }
            return self._removingItemFromLeaf(level, at: bucket, res.slot)
              .replacement
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            return _subtracting(level.descend(), r[child: rslot])
              .map { .childBranch(level, $0, at: bucket) }
          }
          return nil
        }
      }
    }

    assert(rc)
    // `other` is a collision node on a compressed path.
    return read { l in
      other.read { r in
        let bucket = r.collisionHash[level]
        if l.itemMap.contains(bucket) {
          let lslot = l.itemMap.slot(of: bucket)
          let litem = l.itemPtr(at: lslot)
          let h = _Hash(litem.pointee.key)
          let res = r.find(level, litem.pointee.key, h)
          if res == nil { return nil }
          return self._removingItemFromLeaf(level, at: bucket, lslot)
            .replacement
        }
        if l.childMap.contains(bucket) {
          let lslot = l.childMap.slot(of: bucket)
          let branch = l[child: lslot]._subtracting(level.descend(), other)
          guard let branch = branch else { return nil }
          var result = self._removingChild(level, at: bucket, lslot)
          result.addNewChildBranch(level, branch, at: bucket)
          return result
        }
        return nil
      }
    }
  }
}
