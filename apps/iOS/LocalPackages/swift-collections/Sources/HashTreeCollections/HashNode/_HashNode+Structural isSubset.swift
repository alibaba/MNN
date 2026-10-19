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
  /// Returns true if `self` contains a subset of the keys in `other`.
  /// Otherwise, returns false.
  @inlinable @inline(never)
  internal func isSubset<Value2>(
    _ level: _HashLevel,
    of other: _HashNode<Key, Value2>
  ) -> Bool {
    guard self.count > 0 else { return true }
    if self.raw.storage === other.raw.storage { return true }
    guard self.count <= other.count else { return false }

    if self.isCollisionNode {
      if other.isCollisionNode {
        guard self.collisionHash == other.collisionHash else { return false }
        return read { l in
          other.read { r in
            let li = l.reverseItems
            let ri = r.reverseItems
            return l.reverseItems.indices.allSatisfy { i in
              ri.contains { $0.key == li[i].key }
            }
          }
        }
      }
      // `self` is on a compressed path. Try to descend down by one level.
      assert(!level.isAtBottom)
      let bucket = self.collisionHash[level]
      return other.read {
        guard $0.childMap.contains(bucket) else { return false }
        let slot = $0.childMap.slot(of: bucket)
        return self.isSubset(level.descend(), of: $0[child: slot])
      }
    }
    if other.isCollisionNode {
      return read { l in
        guard level.isAtRoot, l.hasSingletonItem else { return false }
        // Annoying special case: the root node may contain a single item
        // that matches one in the collision node.
        return other.read { r in
          let hash = _Hash(l[item: .zero].key)
          return r.find(level, l[item: .zero].key, hash) != nil
        }
      }
    }

    return self.read { l in
      other.read { r in
        guard l.childMap.isSubset(of: r.childMap) else { return false }
        guard l.itemMap.isSubset(of: r.itemMap.union(r.childMap)) else {
          return false
        }
        for (bucket, lslot) in l.itemMap {
          if r.itemMap.contains(bucket) {
            let rslot = r.itemMap.slot(of: bucket)
            guard l[item: lslot].key == r[item: rslot].key else { return false }
          } else {
            let hash = _Hash(l[item: lslot].key)
            let rslot = r.childMap.slot(of: bucket)
            guard
              r[child: rslot].containsKey(
                level.descend(),
                l[item: lslot].key,
                hash)
            else { return false }
          }
        }

        for (bucket, lslot) in l.childMap {
          let rslot = r.childMap.slot(of: bucket)
          guard l[child: lslot].isSubset(level.descend(), of: r[child: rslot])
          else { return false }
        }
        return true
      }
    }
  }
}
