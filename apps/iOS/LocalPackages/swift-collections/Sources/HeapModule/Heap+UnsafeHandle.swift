//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Heap {
  @usableFromInline @frozen
  struct _UnsafeHandle {
    @usableFromInline
    var buffer: UnsafeMutableBufferPointer<Element>

    @inlinable @inline(__always)
    init(_ buffer: UnsafeMutableBufferPointer<Element>) {
      self.buffer = buffer
    }
  }

  @inlinable @inline(__always)
  mutating func _update<R>(_ body: (_UnsafeHandle) -> R) -> R {
    _storage.withUnsafeMutableBufferPointer { buffer in
      body(_UnsafeHandle(buffer))
    }
  }
}

extension Heap._UnsafeHandle {
  @inlinable @inline(__always)
  internal var count: Int {
    buffer.count
  }

  @inlinable
  subscript(node: _HeapNode) -> Element {
    @inline(__always)
    get {
      buffer[node.offset]
    }
    @inline(__always)
    nonmutating _modify {
      yield &buffer[node.offset]
    }
  }

  @inlinable @inline(__always)
  internal func ptr(to node: _HeapNode) -> UnsafeMutablePointer<Element> {
    assert(node.offset < count)
    return buffer.baseAddress! + node.offset
  }

  /// Move the value at the specified node out of the buffer, leaving it
  /// uninitialized.
  @inlinable @inline(__always)
  internal func extract(_ node: _HeapNode) -> Element {
    ptr(to: node).move()
  }

  @inlinable @inline(__always)
  internal func initialize(_ node: _HeapNode, to value: __owned Element) {
    ptr(to: node).initialize(to: value)
  }

  /// Swaps the elements in the heap at the given indices.
  @inlinable @inline(__always)
  internal func swapAt(_ i: _HeapNode, _ j: _HeapNode) {
    buffer.swapAt(i.offset, j.offset)
  }

  /// Swaps the element at the given node with the supplied value.
  @inlinable @inline(__always)
  internal func swapAt(_ i: _HeapNode, with value: inout Element) {
    let p = buffer.baseAddress.unsafelyUnwrapped + i.offset
    swap(&p.pointee, &value)
  }


  @inlinable @inline(__always)
  internal func minValue(_ a: _HeapNode, _ b: _HeapNode) -> _HeapNode {
    // The expression used here matches the implementation of the
    // standard `Swift.min(_:_:)` function. This attempts to
    // preserve any pre-existing order in case `T` has identity.
    // `(min(x, y), max(x, y))` should return `(x, y)` in case `x == y`.
    self[b] < self[a] ? b : a
  }

  @inlinable @inline(__always)
  internal func maxValue(_ a: _HeapNode, _ b: _HeapNode) -> _HeapNode {
    //  In case `a` and `b` match, we need to pick `b`. See `minValue(_:_:)`.
    self[b] >= self[a] ? b : a
  }
}

extension Heap._UnsafeHandle {
  @inlinable
  internal func bubbleUp(_ node: _HeapNode) {
    guard !node.isRoot else { return }

    let parent = node.parent()

    var node = node
    if (node.isMinLevel && self[node] > self[parent])
        || (!node.isMinLevel && self[node] < self[parent]){
      swapAt(node, parent)
      node = parent
    }

    if node.isMinLevel {
      while let grandparent = node.grandParent(),
            self[node] < self[grandparent] {
        swapAt(node, grandparent)
        node = grandparent
      }
    } else {
      while let grandparent = node.grandParent(),
            self[node] > self[grandparent] {
        swapAt(node, grandparent)
        node = grandparent
      }
    }
  }
}

extension Heap._UnsafeHandle {
  /// Sink the item at `node` to its correct position in the heap.
  /// The given node must be minimum-ordered.
  @inlinable
  internal func trickleDownMin(_ node: _HeapNode) {
    assert(node.isMinLevel)
    var node = node
    var value = extract(node)
    _trickleDownMin(node: &node, value: &value)
    initialize(node, to: value)
  }

  @inlinable @inline(__always)
  internal func _trickleDownMin(node: inout _HeapNode, value: inout Element) {
    // Note: `_HeapNode` is quite the useless abstraction here, as we don't need
    // to look at its `level` property, and we need to move sideways amongst
    // siblings/cousins in the tree, for which we don't have direct operations.
    // Luckily, all the `_HeapNode` business gets optimized away, so this only
    // affects the readability of the code, not its performance.
    // The alternative would be to reintroduce offset-based parent/child
    // navigation methods, which seems less palatable.

    var gc0 = node.firstGrandchild()
    while gc0.offset &+ 3 < count {
      // Invariant: buffer slot at `node` is uninitialized

      // We have four grandchildren, so we don't need to compare children.
      let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
      let minA = minValue(gc0, gc1)

      let gc2 = _HeapNode(offset: gc0.offset &+ 2, level: gc0.level)
      let gc3 = _HeapNode(offset: gc0.offset &+ 3, level: gc0.level)
      let minB = minValue(gc2, gc3)

      let min = minValue(minA, minB)
      guard self[min] < value else {
        return // We're done -- `node` is a good place for `value`.
      }

      initialize(node, to: extract(min))
      node = min
      gc0 = node.firstGrandchild()

      let parent = min.parent()
      if self[parent] < value {
        swapAt(parent, with: &value)
      }
    }

    // At this point, we don't have a full complement of grandchildren, but
    // we haven't finished sinking the item.

    let c0 = node.leftChild()
    if c0.offset >= count {
      return // No more descendants to consider.
    }
    let min = _minDescendant(c0: c0, gc0: gc0)
    guard self[min] < value else {
      return // We're done.
    }

    initialize(node, to: extract(min))
    node = min

    if min < gc0 { return }

    // If `min` was a grandchild, check the parent.
    let parent = min.parent()
    if self[parent] < value {
      initialize(node, to: extract(parent))
      node = parent
    }
  }

  /// Returns the node holding the minimal item amongst the children &
  /// grandchildren of a node in the tree. The parent node is not specified;
  /// instead, this function takes the nodes corresponding to its first child
  /// (`c0`) and first grandchild (`gc0`).
  ///
  /// There must be at least one child, but there must not be a full complement
  /// of 4 grandchildren. (Other cases are handled directly above.)
  ///
  /// This method is an implementation detail of `trickleDownMin`. Do not call
  /// it directly.
  @inlinable
  internal func _minDescendant(c0: _HeapNode, gc0: _HeapNode) -> _HeapNode {
    assert(c0.offset < count)
    assert(gc0.offset + 3 >= count)

    if gc0.offset < count {
      if gc0.offset &+ 2 < count {
        // We have three grandchildren. We don't need to compare direct children.
        let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
        let gc2 = _HeapNode(offset: gc0.offset &+ 2, level: gc0.level)
        return minValue(minValue(gc0, gc1), gc2)
      }

      let c1 = _HeapNode(offset: c0.offset &+ 1, level: c0.level)
      let m = minValue(c1, gc0)
      if gc0.offset &+ 1 < count {
        // Two grandchildren.
        let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
        return minValue(m, gc1)
      }

      // One grandchild.
      return m
    }

    let c1 = _HeapNode(offset: c0.offset &+ 1, level: c0.level)
    if c1.offset < count {
      return minValue(c0, c1)
    }

    return c0
  }

  /// Sink the item at `node` to its correct position in the heap.
  /// The given node must be maximum-ordered.
  @inlinable
  internal func trickleDownMax(_ node: _HeapNode) {
    assert(!node.isMinLevel)
    var node = node
    var value = extract(node)

    _trickleDownMax(node: &node, value: &value)
    initialize(node, to: value)
  }

  @inlinable @inline(__always)
  internal func _trickleDownMax(node: inout _HeapNode, value: inout Element) {
    // See note on `_HeapNode` in `_trickleDownMin` above.

    var gc0 = node.firstGrandchild()
    while gc0.offset &+ 3 < count {
      // Invariant: buffer slot at `node` is uninitialized

      // We have four grandchildren, so we don't need to compare children.
      let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
      let maxA = maxValue(gc0, gc1)

      let gc2 = _HeapNode(offset: gc0.offset &+ 2, level: gc0.level)
      let gc3 = _HeapNode(offset: gc0.offset &+ 3, level: gc0.level)
      let maxB = maxValue(gc2, gc3)

      let max = maxValue(maxA, maxB)
      guard value < self[max] else {
        return // We're done -- `node` is a good place for `value`.
      }

      initialize(node, to: extract(max))
      node = max
      gc0 = node.firstGrandchild()

      let parent = max.parent()
      if value < self[parent] {
        swapAt(parent, with: &value)
      }
    }

    // At this point, we don't have a full complement of grandchildren, but
    // we haven't finished sinking the item.

    let c0 = node.leftChild()
    if c0.offset >= count {
      return // No more descendants to consider.
    }
    let max = _maxDescendant(c0: c0, gc0: gc0)
    guard value < self[max] else {
      return // We're done.
    }

    initialize(node, to: extract(max))
    node = max

    if max < gc0 { return }

    // If `max` was a grandchild, check the parent.
    let parent = max.parent()
    if value < self[parent] {
      initialize(node, to: extract(parent))
      node = parent
    }
  }

  /// Returns the node holding the maximal item amongst the children &
  /// grandchildren of a node in the tree. The parent node is not specified;
  /// instead, this function takes the nodes corresponding to its first child
  /// (`c0`) and first grandchild (`gc0`).
  ///
  /// There must be at least one child, but there must not be a full complement
  /// of 4 grandchildren. (Other cases are handled directly above.)
  ///
  /// This method is an implementation detail of `trickleDownMax`. Do not call
  /// it directly.
  @inlinable
  internal func _maxDescendant(c0: _HeapNode, gc0: _HeapNode) -> _HeapNode {
    assert(c0.offset < count)
    assert(gc0.offset + 3 >= count)

    if gc0.offset < count {
      if gc0.offset &+ 2 < count {
        // We have three grandchildren. We don't need to compare direct children.
        let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
        let gc2 = _HeapNode(offset: gc0.offset &+ 2, level: gc0.level)
        return maxValue(maxValue(gc0, gc1), gc2)
      }

      let c1 = _HeapNode(offset: c0.offset &+ 1, level: c0.level)
      let m = maxValue(c1, gc0)
      if gc0.offset &+ 1 < count {
        // Two grandchildren.
        let gc1 = _HeapNode(offset: gc0.offset &+ 1, level: gc0.level)
        return maxValue(m, gc1)
      }

      // One grandchild.
      return m
    }

    let c1 = _HeapNode(offset: c0.offset &+ 1, level: c0.level)
    if c1.offset < count {
      return maxValue(c0, c1)
    }

    return c0
  }
}

extension Heap._UnsafeHandle {
  @inlinable
  internal func heapify() {
    // This is Floyd's linear-time heap construction algorithm.
    // (https://en.wikipedia.org/wiki/Heapsort#Floyd's_heap_construction).
    //
    // FIXME: See if a more cache friendly algorithm would be faster.

    let limit = count / 2 // The first offset without a left child
    var level = _HeapNode.level(forOffset: limit &- 1)
    while level >= 0 {
      let nodes = _HeapNode.allNodes(onLevel: level, limit: limit)
      _heapify(level, nodes)
      level &-= 1
    }
  }

  @inlinable
  internal func _heapify(_ level: Int, _ nodes: ClosedRange<_HeapNode>?) {
    guard let nodes = nodes else { return }
    if _HeapNode.isMinLevel(level) {
      nodes._forEach { node in
        trickleDownMin(node)
      }
    } else {
      nodes._forEach { node in
        trickleDownMax(node)
      }
    }
  }
}
