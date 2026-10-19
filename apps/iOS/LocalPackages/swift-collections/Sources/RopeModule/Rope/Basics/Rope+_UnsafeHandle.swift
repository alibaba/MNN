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
  @usableFromInline
  @frozen // Not really! This module isn't ABI stable.
  internal struct _UnsafeHandle<Child: _RopeItem<Summary>> {
    @usableFromInline internal typealias Summary = Rope.Summary
    
    @usableFromInline
    internal let _header: UnsafeMutablePointer<_RopeStorageHeader>

    @usableFromInline
    internal let _start: UnsafeMutablePointer<Child>
#if DEBUG
    @usableFromInline
    internal let _isMutable: Bool
#endif

    @inlinable
    internal init(
      isMutable: Bool,
      header: UnsafeMutablePointer<_RopeStorageHeader>,
      start: UnsafeMutablePointer<Child>
    ) {
      self._header = header
      self._start = start
#if DEBUG
      self._isMutable = isMutable
#endif
    }
    
    @inlinable @inline(__always)
    internal func assertMutable() {
#if DEBUG
      assert(_isMutable)
#endif
    }
  }
}

extension Rope._UnsafeHandle {
  @inlinable @inline(__always)
  internal var capacity: Int { Summary.maxNodeSize }
  
  @inlinable @inline(__always)
  internal var height: UInt8 { _header.pointee.height }

  @inlinable
  internal var childCount: Int {
    get { _header.pointee.childCount }
    nonmutating set {
      assertMutable()
      _header.pointee.childCount = newValue
    }
  }

  @inlinable
  internal var children: UnsafeBufferPointer<Child> {
    UnsafeBufferPointer(start: _start, count: childCount)
  }

  @inlinable
  internal func child(at slot: Int) -> Child? {
    assert(slot >= 0)
    guard slot < childCount else { return nil }
    return (_start + slot).pointee
  }

  @inlinable
  internal var mutableChildren: UnsafeMutableBufferPointer<Child> {
    assertMutable()
    return UnsafeMutableBufferPointer(start: _start, count: childCount)
  }

  @inlinable
  internal func mutableChildPtr(at slot: Int) -> UnsafeMutablePointer<Child> {
    assertMutable()
    assert(slot >= 0 && slot < childCount)
    return _start + slot
  }

  @inlinable
  internal var mutableBuffer: UnsafeMutableBufferPointer<Child> {
    assertMutable()
    return UnsafeMutableBufferPointer(start: _start, count: capacity)
  }

  @inlinable
  internal func copy() -> Rope._Storage<Child> {
    let new = Rope._Storage<Child>.create(height: self.height)
    let c = self.childCount
    new.header.childCount = c
    new.withUnsafeMutablePointerToElements { target in
      target.initialize(from: self._start, count: c)
    }
    return new
  }

  @inlinable
  internal func copy(
    slots: Range<Int>
  ) -> (object: Rope._Storage<Child>, summary: Summary) {
    assert(slots.lowerBound >= 0 && slots.upperBound <= childCount)
    let object = Rope._Storage<Child>.create(height: self.height)
    let c = slots.count
    let summary = object.withUnsafeMutablePointers { h, p in
      h.pointee.childCount = c
      p.initialize(from: self._start + slots.lowerBound, count: slots.count)
      return UnsafeBufferPointer(start: p, count: c)._sum()
    }
    return (object, summary)
  }

  @inlinable
  internal func _insertChild(_ child: __owned Child, at slot: Int) {
    assertMutable()
    assert(childCount < capacity)
    assert(slot >= 0 && slot <= childCount)
    (_start + slot + 1).moveInitialize(from: _start + slot, count: childCount - slot)
    (_start + slot).initialize(to: child)
    childCount += 1
  }

  @inlinable
  internal func _appendChild(_ child: __owned Child) {
    assertMutable()
    assert(childCount < capacity)
    (_start + childCount).initialize(to: child)
    childCount += 1
  }

  @inlinable
  internal func _removeChild(at slot: Int) -> Child {
    assertMutable()
    assert(slot >= 0 && slot < childCount)
    let result = (_start + slot).move()
    (_start + slot).moveInitialize(from: _start + slot + 1, count: childCount - slot - 1)
    childCount -= 1
    return result
  }

  @inlinable
  internal func _removePrefix(_ n: Int) -> Summary {
    assertMutable()
    assert(n <= childCount)
    var delta = Summary.zero
    let c = mutableChildren
    for i in 0 ..< n {
      let child = c.moveElement(from: i)
      delta.add(child.summary)
    }
    childCount -= n
    _start.moveInitialize(from: _start + n, count: childCount)
    return delta
  }

  @inlinable
  internal func _removeSuffix(_ n: Int) -> Summary {
    assertMutable()
    assert(n <= childCount)
    var delta = Summary.zero
    let c = mutableChildren
    for i in childCount - n ..< childCount {
      let child = c.moveElement(from: i)
      delta.add(child.summary)
    }
    childCount -= n
    return delta
  }

  @inlinable
  internal func _appendChildren(
    movingFromPrefixOf src: Self, count: Int
  ) -> Summary {
    assertMutable()
    src.assertMutable()
    assert(self.height == src.height)
    guard count > 0 else { return .zero }
    assert(count >= 0 && count <= src.childCount)
    assert(count <= capacity - self.childCount)
    
    (_start + childCount).moveInitialize(from: src._start, count: count)
    src._start.moveInitialize(from: src._start + count, count: src.childCount - count)
    childCount += count
    src.childCount -= count
    return children.suffix(count)._sum()
  }

  @inlinable
  internal func _prependChildren(
    movingFromSuffixOf src: Self, count: Int
  ) -> Summary {
    assertMutable()
    src.assertMutable()
    assert(self.height == src.height)
    guard count > 0 else { return .zero }
    assert(count >= 0 && count <= src.childCount)
    assert(count <= capacity - childCount)
    
    (_start + count).moveInitialize(from: _start, count: childCount)
    _start.moveInitialize(from: src._start + src.childCount - count, count: count)
    childCount += count
    src.childCount -= count
    return children.prefix(count)._sum()
  }

  @inlinable
  internal func distance(
    from start: Int, to end: Int, in metric: some RopeMetric<Element>
  ) -> Int {
    if start <= end {
      return children[start ..< end].reduce(into: 0) {
        $0 += metric._nonnegativeSize(of: $1.summary)
      }
    }
    return -children[end ..< start].reduce(into: 0) {
      $0 += metric._nonnegativeSize(of: $1.summary)
    }
  }
}
