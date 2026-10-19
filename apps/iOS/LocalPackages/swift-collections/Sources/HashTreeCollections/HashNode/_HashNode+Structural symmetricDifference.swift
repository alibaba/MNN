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
  internal func symmetricDifference<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode<Key, Void>.Builder? {
    guard self.count > 0 else {
      return .init(level, other.mapValuesToVoid())
    }
    guard other.count > 0 else { return nil }
    return _symmetricDifference(level, other)
  }

  @inlinable
  internal func _symmetricDifference<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode<Key, Void>.Builder {
    typealias VoidNode = _HashNode<Key, Void>

    assert(self.count > 0 && other.count > 0)

    if self.raw.storage === other.raw.storage {
      return .empty(level)
    }
    if self.isCollisionNode || other.isCollisionNode {
      return _symmetricDifference_slow(level, other)
    }

    return self.read { l in
      other.read { r in
        var result: VoidNode.Builder = .empty(level)

        for (bucket, lslot) in l.itemMap {
          let lp = l.itemPtr(at: lslot)
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            let rp = r.itemPtr(at: rslot)
            if lp.pointee.key != rp.pointee.key {
              let h1 = _Hash(lp.pointee.key)
              let h2 = _Hash(rp.pointee.key)
              let child = VoidNode.build(
                level: level.descend(),
                item1: (lp.pointee.key, ()), h1,
                item2: { $0.initialize(to: (rp.pointee.key, ())) }, h2)
              result.addNewChildNode(level, child.top, at: bucket)
            }
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let rp = r.childPtr(at: rslot)
            let h = _Hash(lp.pointee.key)
            let child = rp.pointee
              .removing(level.descend(), lp.pointee.key, h)?.replacement
            if let child = child {
              let child = child.mapValuesToVoid()
              result.addNewChildBranch(level, child, at: bucket)
            }
            else {
              var child2 = rp.pointee
                .mapValuesToVoid(
                  copy: true, extraBytes: VoidNode.spaceForNewItem)
              let r = child2.insert(level.descend(), (lp.pointee.key, ()), h)
              assert(r.inserted)
              result.addNewChildNode(level, child2, at: bucket)
            }
          }
          else {
            result.addNewItem(level, (lp.pointee.key, ()), at: bucket)
          }
        }

        for (bucket, lslot) in l.childMap {
          let lp = l.childPtr(at: lslot)
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            let rp = r.itemPtr(at: rslot)
            let h = _Hash(rp.pointee.key)
            let child = lp.pointee
              .mapValuesToVoid()
              .removing(level.descend(), rp.pointee.key, h)?.replacement
            if let child = child {
              result.addNewChildBranch(level, child, at: bucket)
            }
            else {
              var child2 = lp.pointee.mapValuesToVoid(
                copy: true, extraBytes: VoidNode.spaceForNewItem)
              let r2 = child2.insert(level.descend(), (rp.pointee.key, ()), h)
              assert(r2.inserted)
              result.addNewChildNode(level, child2, at: bucket)
            }
          }
          else if r.childMap.contains(bucket) {
            let rslot = r.childMap.slot(of: bucket)
            let b = l[child: lslot]._symmetricDifference(
              level.descend(), r[child: rslot])
            result.addNewChildBranch(level, b, at: bucket)
          }
          else {
            result.addNewChildNode(
              level, lp.pointee.mapValuesToVoid(), at: bucket)
          }
        }

        let seen = l.itemMap.union(l.childMap)
        for (bucket, rslot) in r.itemMap {
          guard !seen.contains(bucket) else { continue }
          result.addNewItem(level, (r[item: rslot].key, ()), at: bucket)
        }
        for (bucket, rslot) in r.childMap {
          guard !seen.contains(bucket) else { continue }
          result.addNewChildNode(
            level, r[child: rslot].mapValuesToVoid(), at: bucket)
        }
        return result
      }
    }
  }

  @inlinable @inline(never)
  internal func _symmetricDifference_slow<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode<Key, Void>.Builder {
    switch (self.isCollisionNode, other.isCollisionNode) {
    case (true, true):
      return self._symmetricDifference_slow_both(level, other)
    case (true, false):
      return self._symmetricDifference_slow_left(level, other)
    case (false, _):
      return other._symmetricDifference_slow_left(level, self)
    }
  }

  @inlinable
  internal func _symmetricDifference_slow_both<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode<Key, Void>.Builder {
    typealias VoidNode = _HashNode<Key, Void>
    return read { l in
      other.read { r in
        guard l.collisionHash == r.collisionHash else {
          let node = VoidNode.build(
            level: level,
            child1: self.mapValuesToVoid(), l.collisionHash,
            child2: other.mapValuesToVoid(), r.collisionHash)
          return .node(level, node)
        }
        var result: VoidNode.Builder = .empty(level)
        let ritems = r.reverseItems
        for ls: _HashSlot in stride(from: .zero, to: l.itemsEndSlot, by: 1) {
          let lp = l.itemPtr(at: ls)
          let include = !ritems.contains(where: { $0.key == lp.pointee.key })
          if include {
            result.addNewCollision(level, (lp.pointee.key, ()), l.collisionHash)
          }
        }
        // FIXME: Consider remembering slots of shared items in r by
        // caching them in a bitset.
        let litems = l.reverseItems
        for rs: _HashSlot in stride(from: .zero, to: r.itemsEndSlot, by: 1) {
          let rp = r.itemPtr(at: rs)
          let include = !litems.contains(where: { $0.key == rp.pointee.key })
          if include {
            result.addNewCollision(level, (rp.pointee.key, ()), r.collisionHash)
          }
        }
        return result
      }
    }
  }

  @inlinable
  internal func _symmetricDifference_slow_left<Value2>(
    _ level: _HashLevel,
    _ other: _HashNode<Key, Value2>
  ) -> _HashNode<Key, Void>.Builder {
    typealias VoidNode = _HashNode<Key, Void>
    // `self` is a collision node on a compressed path. The other tree might
    // have the same set of collisions, just expanded a bit deeper.
    return read { l in
      other.read { r in
        assert(l.isCollisionNode && !r.isCollisionNode)
        let bucket = l.collisionHash[level]
        if r.itemMap.contains(bucket) {
          let rslot = r.itemMap.slot(of: bucket)
          let rp = r.itemPtr(at: rslot)
          let rh = _Hash(rp.pointee.key)
          guard rh == l.collisionHash else {
            var copy = other.mapValuesToVoid(
              copy: true, extraBytes: VoidNode.spaceForSpawningChild)
            let item = copy.removeItem(at: bucket, rslot)
            let child = VoidNode.build(
              level: level.descend(),
              item1: { $0.initialize(to: (item.key, ())) }, rh,
              child2: self.mapValuesToVoid(), l.collisionHash)
            copy.insertChild(child.top, bucket)
            return .node(level, copy)
          }
          let litems = l.reverseItems
          if let li = litems.firstIndex(where: { $0.key == rp.pointee.key }) {
            if l.itemCount == 2 {
              var node = other.mapValuesToVoid(copy: true)
              node.replaceItem(
                at: bucket, rslot,
                with: (litems[1 &- li].key, ()))
              return .node(level, node)
            }
            let lslot = _HashSlot(litems.count &- 1 &- li)
            var child = self.mapValuesToVoid(copy: true)
            _ = child.removeItem(at: .invalid, lslot)
            if other.hasSingletonItem {
              // Compression
              return .collisionNode(level, child)
            }
            var node = other.mapValuesToVoid(
              copy: true, extraBytes: VoidNode.spaceForSpawningChild)
            _ = node.removeItem(at: bucket, rslot)
            node.insertChild(child, bucket)
            return .node(level, node)
          }
          if other.hasSingletonItem {
            // Compression
            var copy = self.mapValuesToVoid(
              copy: true, extraBytes: VoidNode.spaceForNewItem)
            _ = copy.ensureUniqueAndAppendCollision(
              isUnique: true,
              (r[item: .zero].key, ()))
            return .collisionNode(level, copy)
          }
          var node = other.mapValuesToVoid(
            copy: true, extraBytes: VoidNode.spaceForSpawningChild)
          let item = node.removeItem(at: bucket, rslot)
          var child = self.mapValuesToVoid(
            copy: true, extraBytes: VoidNode.spaceForNewItem)
          _ = child.ensureUniqueAndAppendCollision(
            isUnique: true, (item.key, ()))
          node.insertChild(child, bucket)
          return .node(level, node)
        }
        if r.childMap.contains(bucket) {
          let rslot = r.childMap.slot(of: bucket)
          let rp = r.childPtr(at: rslot)
          let child = rp.pointee._symmetricDifference(level.descend(), self)
          return other
            .mapValuesToVoid()
            .replacingChild(level, at: bucket, rslot, with: child)
        }
        var node = other
          .mapValuesToVoid(copy: true, extraBytes: VoidNode.spaceForNewChild)
        node.insertChild(self.mapValuesToVoid(), bucket)
        return .node(level, node)
      }
    }
  }
}
