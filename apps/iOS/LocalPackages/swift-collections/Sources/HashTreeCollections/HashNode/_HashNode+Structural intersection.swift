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
  internal func intersection<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode? {
    assert(level.isAtRoot)
    let builder = _intersection(level, other)
    guard let builder = builder else { return nil }
    let root = builder.finalize(.top)
    root._fullInvariantCheck()
    return root
  }

  @inlinable
  internal func _intersection<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> Builder? {
    if self.raw.storage === other.raw.storage { return nil }

    if self.isCollisionNode || other.isCollisionNode {
      return _intersection_slow(level, other)
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
            include = (lp.pointee.key == r[item: rslot].key)
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let h = _Hash(lp.pointee.key)
            include = r[child: rslot]
              .containsKey(level.descend(), lp.pointee.key, h)
          }
          else { include = false}

          if include, removing {
            result.addNewItem(level, lp.pointee, at: bucket)
          }
          else if !include, !removing {
            removing = true
            result.copyItems(level, from: l, upTo: bucket)
          }
        }

        for (bucket, lslot) in l.childMap {
          if r.itemMap.contains(bucket) {
            if !removing {
              removing = true
              result.copyItemsAndChildren(level, from: l, upTo: bucket)
            }
            let rslot = r.itemMap.slot(of: bucket)
            let rp = r.itemPtr(at: rslot)
            let h = _Hash(rp.pointee.key)
            let res = l[child: lslot].lookup(level.descend(), rp.pointee.key, h)
            if let res = res {
              let item = UnsafeHandle.read(res.node) { $0[item: res.slot] }
              result.addNewItem(level, item, at: bucket)
            }
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let branch = l[child: lslot]
              ._intersection(level.descend(), r[child: rslot])
            if let branch = branch {
              assert(branch.count < self.count)
              if !removing {
                removing = true
                result.copyItemsAndChildren(level, from: l, upTo: bucket)
              }
              result.addNewChildBranch(level, branch, at: bucket)
            } else if removing {
              result.addNewChildNode(level, l[child: lslot], at: bucket)
            }
          }
          else if !removing {
            removing = true
            result.copyItemsAndChildren(level, from: l, upTo: bucket)
          }
        }
        guard removing else { return nil }
        return result
      }
    }
  }

  @inlinable @inline(never)
  internal func _intersection_slow<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> Builder? {
    let lc = self.isCollisionNode
    let rc = other.isCollisionNode
    if lc && rc {
      return read { l in
        other.read { r in
          var result: Builder = .empty(level)
          guard l.collisionHash == r.collisionHash else { return result }

          var removing = false
          let ritems = r.reverseItems
          for lslot: _HashSlot in stride(from: .zero, to: l.itemsEndSlot, by: 1) {
            let lp = l.itemPtr(at: lslot)
            let include = ritems.contains { $0.key == lp.pointee.key }
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
            let litems = l.reverseItems
            let i = litems.firstIndex { $0.key == ritem.pointee.key }
            guard let i = i else { return .empty(level) }
            return .item(level, litems[i], at: l.collisionHash[level])
          }
          if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            return _intersection(level.descend(), r[child: rslot])
              .map { .childBranch(level, $0, at: bucket) }
          }
          return .empty(level)
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
          let ritems = r.reverseItems
          let found = ritems.contains { $0.key == litem.pointee.key }
          guard found else { return .empty(level) }
          return .item(level, litem.pointee, at: bucket)
        }
        if l.childMap.contains(bucket) {
          let lslot = l.childMap.slot(of: bucket)
          let branch = l[child: lslot]._intersection(level.descend(), other)
          guard let branch = branch else {
            assert(l[child: lslot].isCollisionNode)
            assert(l[child: lslot].collisionHash == r.collisionHash)
            // Compression
            return .collisionNode(level, l[child: lslot])
          }
          return .childBranch(level, branch, at: bucket)
        }
        return .empty(level)
      }
    }
  }
}
