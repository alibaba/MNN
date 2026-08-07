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
  public mutating func builder(
    splittingAt position: Int,
    in metric: some RopeMetric<Element>
  ) -> Builder {
    _invalidateIndices()
    var builder = Builder()

    if self.isEmpty {
      precondition(position == 0, "Position out of bounds")
      return builder
    }
    var position = position
    var node = root
    _root = nil
    while !node.isLeaf {
      let r = node.readInner { $0.findSlot(at: position, in: metric) }
      position = r.remaining
      node._innerSplit(at: r.slot, into: &builder)
    }

    let r = node.readLeaf { $0.findSlot(at: position, in: metric) }
    var item = node._leafSplit(at: r.slot, into: &builder)
    let index = metric.index(at: r.remaining, in: item.value)
    let suffix = item.split(at: index)
    builder._insertAfterTip(suffix)
    builder._insertBeforeTip(item)
    return builder
  }

  @inlinable
  public mutating func split(at index: Index) -> (builder: Builder, item: Element) {
    validate(index)
    precondition(index < endIndex)
    var builder = Builder()
    
    var node = root
    _root = nil
    while !node.isLeaf {
      let slot = index._path[node.height]
      precondition(slot < node.childCount, "Index out of bounds")
      node._innerSplit(at: slot, into: &builder)
    }
    
    let slot = index._path[node.height]
    precondition(slot < node.childCount, "Index out of bounds")
    let item = node._leafSplit(at: slot, into: &builder)
    _invalidateIndices()
    return (builder, item.value)
  }

  @inlinable
  public mutating func split(at ropeIndex: Index, _ itemIndex: Element.Index) -> Builder {
    var (builder, item) = self.split(at: ropeIndex)
    let suffix = item.split(at: itemIndex)
    if !suffix.isEmpty {
      builder.insertAfterTip(suffix)
    }
    if !item.isEmpty {
      builder.insertBeforeTip(item)
    }
    return builder
  }
}

extension Rope._Node {
  @inlinable
  internal mutating func _innerSplit(
    at slot: Int,
    into builder: inout Rope.Builder
  ) {
    assert(!self.isLeaf)
    assert(slot >= 0 && slot < childCount)
    ensureUnique()
    
    var slot = slot
    if slot == childCount - 2 {
      builder._insertAfterTip(_removeNode(at: childCount - 1))
    }
    if slot == 1 {
      builder._insertBeforeTip(_removeNode(at: 0))
      slot -= 1
    }
    
    var n = _removeNode(at: slot)
    swap(&self, &n)
    
    guard n.childCount > 0 else { return }
    if slot == 0 {
      builder._insertAfterTip(n)
      return
    }
    if slot == n.childCount {
      builder._insertBeforeTip(n)
      return
    }
    let suffix = n.split(keeping: slot)
    builder._insertBeforeTip(n)
    builder._insertAfterTip(suffix)
  }

  @inlinable
  internal __consuming func _leafSplit(
    at slot: Int,
    into builder: inout Rope.Builder
  ) -> _Item {
    var n = self
    n.ensureUnique()
    
    assert(n.isLeaf)
    assert(slot >= 0 && slot < n.childCount)
    
    var slot = slot
    if slot == n.childCount - 2 {
      builder._insertAfterTip(n._removeItem(at: childCount - 1).removed)
    }
    if slot == 1 {
      builder.insertBeforeTip(n._removeItem(at: 0).removed.value)
      slot -= 1
    }
    
    let item = n._removeItem(at: slot).removed
    
    guard n.childCount > 0 else { return item }
    if slot == 0 {
      builder._insertAfterTip(n)
    } else if slot == n.childCount {
      builder._insertBeforeTip(n)
    } else {
      let suffix = n.split(keeping: slot)
      builder._insertBeforeTip(n)
      builder._insertAfterTip(suffix)
    }
    return item
  }
}
