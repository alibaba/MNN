//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Rope {
  @inlinable
  public mutating func append(_ item: __owned Element) {
    _invalidateIndices()
    if _root == nil {
      _root = .createLeaf(_Item(item))
      return
    }
    if let spawn = root.append(_Item(item)) {
      _root = .createInner(children: root, spawn)
    }
  }
}

extension Rope._Node {
  @inlinable
  internal mutating func append(_ item: __owned _Item) -> Self? {
    var item = item
    if item.isUndersized, !self.isEmpty, self.lastItem.rebalance(nextNeighbor: &item) {
      return nil
    }
    ensureUnique()
    if height > 0 {
      var summary = self.summary
      let spawn = updateInner {
        let p = $0.mutableChildPtr(at: $0.childCount - 1)
        summary.subtract(p.pointee.summary)
        let spawn = p.pointee.append(item)
        summary.add(p.pointee.summary)
        return spawn
      }
      self.summary = summary
      guard let spawn = spawn else { return nil }
      
#if true // Compress existing nodes if possible.
      updateInner {
        let c = $0.mutableChildren
        let s = c[c.count - 2].childCount + c[c.count - 1].childCount
        if s <= Summary.maxNodeSize {
          Self.redistributeChildren(&c[c.count - 2], &c[c.count - 1], to: s)
          let removed = $0._removeChild(at: c.count - 1)
          assert(removed.childCount == 0)
        }
      }
#endif
      guard isFull else {
        _appendNode(spawn)
        return nil
      }
      
      var spawn2 = split(keeping: Summary.minNodeSize)
      spawn2._appendNode(spawn)
      return spawn2
    }
    guard isFull else {
      _appendItem(item)
      return nil
    }
    var spawn = split(keeping: Summary.minNodeSize)
    spawn._appendItem(item)
    return spawn
  }
}
