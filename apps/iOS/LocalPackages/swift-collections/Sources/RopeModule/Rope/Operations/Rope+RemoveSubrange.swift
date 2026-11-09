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
  public mutating func removeSubrange(
    _ bounds: Range<Int>,
    in metric: some RopeMetric<Element>
  ) {
    _invalidateIndices()
    precondition(
      bounds.lowerBound >= 0 && bounds.upperBound <= count(in: metric),
      "Position out of bounds")
    guard !bounds.isEmpty else { return }
    // FIXME: Add fast path for tiny removals
    var builder = builder(removing: bounds, in: metric)
    self = builder.finalize()
  }

  @inlinable
  public mutating func replaceSubrange(
    _ bounds: Range<Int>,
    in metric: some RopeMetric<Element>,
    with newElements: __owned some Sequence<Element>
  ) {
    // FIXME: Implement insert(contentsOf:at:in:) and dispatch to it when bounds.isEmpty.
    // FIXME: Add fast path for replacing tiny ranges with tiny data.
    // FIXME: Add special cases if newElements is itself a _Rope etc.
    _invalidateIndices()
    var builder = builder(removing: bounds, in: metric)
    builder.insertBeforeTip(newElements)
    self = builder.finalize()
  }

  @inlinable
  public mutating func builder(
    removing bounds: Range<Int>,
    in metric: some RopeMetric<Element>
  ) -> Builder {
    _invalidateIndices()
    let size = metric.size(of: summary)
    precondition(
      bounds.lowerBound >= 0 && bounds.upperBound <= size,
      "Range out of bounds")

    guard !bounds.isEmpty else {
      return builder(splittingAt: bounds.lowerBound, in: metric)
    }

    var builder = Builder()
    var node = root
    _root = nil
    var lower = bounds.lowerBound
    var upper = bounds.upperBound
    while !node.isLeaf {
      let (l, u) = node.readInner {
        let l = $0.findSlot(at: lower, in: metric, preferEnd: false)
        let u = $0.findSlot(from: l, offsetBy: upper - lower, in: metric, preferEnd: true)
        return (l, u)
      }
      if l.slot < u.slot {
        node._removeSubrange(from: l, to: u, in: metric, into: &builder)
        return builder
      }
      assert(l.slot == u.slot)
      node._innerSplit(at: l.slot, into: &builder)
      lower = l.remaining
      upper = u.remaining
    }

    let (l, u) = node.readLeaf {
      let l = $0.findSlot(at: lower, in: metric, preferEnd: false)
      let u = $0.findSlot(from: l, offsetBy: bounds.count, in: metric, preferEnd: true)
      return (l, u)
    }
    if l.slot < u.slot {
      node._removeSubrange(from: l, to: u, in: metric, into: &builder)
      return builder
    }
    assert(l.slot == u.slot)
    var item = node._leafSplit(at: l.slot, into: &builder)
    let i2 = metric.index(at: u.remaining, in: item.value)
    builder._insertAfterTip(item.split(at: i2))
    let i1 = metric.index(at: l.remaining, in: item.value)
    _ = item.split(at: i1)
    builder._insertBeforeTip(item)
    return builder
  }
}

extension Rope._Node {
  @inlinable
  internal __consuming func _removeSubrange(
    from start: (slot: Int, remaining: Int),
    to end: (slot: Int, remaining: Int),
    in metric: some RopeMetric<Element>,
    into builder: inout Rope.Builder
  ) {
    assert(start.slot >= 0 && start.slot < end.slot && end.slot < childCount)
    assert(start.remaining >= 0)
    assert(end.remaining >= 0)

    builder._insertBeforeTip(slots: 0 ..< start.slot, in: self)
    if end.slot < childCount {
      builder._insertAfterTip(slots: end.slot + 1 ..< childCount, in: self)
    }

    guard isLeaf else {
      // Extract children on boundaries.
      let (lower, upper) = readInner { ($0.children[start.slot], $0.children[end.slot]) }

      // Descend a lever lower.
      lower.removeSuffix(from: start.remaining, in: metric, into: &builder)
      upper.removePrefix(upTo: end.remaining, in: metric, into: &builder)
      return
    }
    // Extract items on boundaries.
    var (lower, upper) = readLeaf { ($0.children[start.slot], $0.children[end.slot]) }

    let i1 = metric.index(at: start.remaining, in: lower.value)
    let i2 = metric.index(at: end.remaining, in: upper.value)
    _ = lower.split(at: i1)
    builder._insertBeforeTip(lower)
    builder._insertAfterTip(upper.split(at: i2))
  }

  @inlinable
  internal __consuming func removeSuffix(
    from position: Int,
    in metric: some RopeMetric<Element>,
    into builder: inout Rope.Builder
  ) {

    var node = self
    var position = position
    while true {
      guard position > 0 else { return }
      guard position < metric.size(of: node.summary) else {
        builder._insertBeforeTip(node)
        return
      }

      guard !node.isLeaf else { break }

      let r = node.readInner { $0.findSlot(at: position, in: metric) }
      position = r.remaining
      node._innerRemoveSuffix(descending: r.slot, into: &builder)
    }
    let r = node.readLeaf { $0.findSlot(at: position, in: metric, preferEnd: false) }
    var item = node._leafRemoveSuffix(returning: r.slot, into: &builder)
    let i = metric.index(at: r.remaining, in: item.value)
    _ = item.split(at: i)
    builder._insertBeforeTip(item)
  }

  @inlinable
  internal __consuming func removePrefix(
    upTo position: Int,
    in metric: some RopeMetric<Element>,
    into builder: inout Rope.Builder
  ) {
    var node = self
    var position = position
    while true {
      guard position > 0 else {
        builder._insertAfterTip(node)
        return
      }
      guard position < metric.size(of: node.summary) else { return }

      guard !node.isLeaf else { break }

      let r = node.readInner { $0.findSlot(at: position, in: metric) }
      position = r.remaining
      node._innerRemovePrefix(descending: r.slot, into: &builder)
    }
    let r = node.readLeaf { $0.findSlot(at: position, in: metric) }
    var item = node._leafRemovePrefix(returning: r.slot, into: &builder)
    let i = metric.index(at: r.remaining, in: item.value)
    builder._insertAfterTip(item.split(at: i))
  }

  @inlinable
  internal mutating func _innerRemoveSuffix(
    descending slot: Int,
    into builder: inout Rope.Builder
  ) {
    assert(!self.isLeaf)
    assert(slot >= 0 && slot <= childCount)

    if slot == 0 {
      self = readInner { $0.children[0] }
      return
    }
    if slot == 1 {
      let (remaining, new) = readInner {
        let c = $0.children
        return (c[0], c[1])
      }
      builder._insertBeforeTip(remaining)
      self = new
      return
    }

    ensureUnique()
    if slot < childCount - 1 {
      let delta = updateInner { $0._removeSuffix($0.childCount - slot - 1) }
      self.summary.subtract(delta)
    }
    var n = _removeNode(at: slot)
    swap(&self, &n)
    assert(n.childCount > 1)
    builder._insertBeforeTip(n)
  }

  @inlinable
  internal __consuming func _leafRemoveSuffix(
    returning slot: Int,
    into builder: inout Rope.Builder
  ) -> _Item {
    assert(self.isLeaf)
    assert(slot >= 0 && slot < childCount)

    if slot == 0 {
      return readLeaf { $0.children[0] }
    }
    if slot == 1 {
      let (remaining, new) = readLeaf {
        let c = $0.children
        return (c[0], c[1])
      }
      builder._insertBeforeTip(remaining)
      return new
    }

    var n = self
    n.ensureUnique()
    if slot < n.childCount - 1 {
      let delta = n.updateLeaf { $0._removeSuffix($0.childCount - slot - 1) }
      n.summary.subtract(delta)
    }
    let item = n._removeItem(at: slot).removed
    builder._insertBeforeTip(n)
    return item
  }

  @inlinable
  internal mutating func _innerRemovePrefix(
    descending slot: Int,
    into builder: inout Rope.Builder
  ) {
    assert(!self.isLeaf)
    assert(slot >= 0 && slot < childCount)

    if slot == childCount - 1 {
      self = readInner { $0.children[$0.childCount - 1] }
      return
    }
    if slot == childCount - 2 {
      let (new, remaining) = readInner {
        let c = $0.children
        return (c[$0.childCount - 2], c[$0.childCount - 1])
      }
      builder._insertAfterTip(remaining)
      self = new
      return
    }

    ensureUnique()
    var (delta, n) = updateInner {
      let n = $0.children[slot]
      let delta = $0._removePrefix(slot + 1)
      return (delta, n)
    }
    self.summary.subtract(delta)
    assert(self.childCount > 1)
    swap(&self, &n)
    builder._insertAfterTip(n)
  }

  @inlinable
  internal __consuming func _leafRemovePrefix(
    returning slot: Int,
    into builder: inout Rope.Builder
  ) -> _Item {
    assert(self.isLeaf)
    assert(slot >= 0 && slot <= childCount)

    if slot == childCount - 1 {
      return readLeaf { $0.children[$0.childCount - 1] }
    }
    if slot == childCount - 2 {
      let (new, remaining) = readLeaf {
        let c = $0.children
        return (c[$0.childCount - 2], c[$0.childCount - 1])
      }
      builder._insertAfterTip(remaining)
      return new
    }

    var n = self
    n.ensureUnique()
    let (delta, item) = n.updateLeaf {
      let n = $0.children[slot]
      let delta = $0._removePrefix(slot + 1)
      return (delta, n)
    }
    n.summary.subtract(delta)
    assert(n.childCount > 1)
    builder._insertAfterTip(n)
    return item
  }
}
