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

@usableFromInline
@frozen
internal struct _HashTreeIterator {
  @usableFromInline
  internal struct _Opaque {
    internal var ancestorSlots: _AncestorHashSlots
    internal var ancestorNodes: _HashStack<_UnmanagedHashNode>
    internal var level: _HashLevel
    internal var isAtEnd: Bool

    @usableFromInline
    @_effects(releasenone)
    internal init(_ root: _UnmanagedHashNode) {
      self.ancestorSlots = .empty
      self.ancestorNodes = _HashStack(filledWith: root)
      self.level = .top
      self.isAtEnd = false
    }
  }

  @usableFromInline
  internal let root: _RawHashStorage

  @usableFromInline
  internal var node: _UnmanagedHashNode

  @usableFromInline
  internal var slot: _HashSlot

  @usableFromInline
  internal var endSlot: _HashSlot

  @usableFromInline
  internal var _o: _Opaque

  @usableFromInline
  @_effects(releasenone)
  internal init(root: __shared _RawHashNode) {
    self.root = root.storage
    self.node = root.unmanaged
    self.slot = .zero
    self.endSlot = node.itemsEndSlot
    self._o = _Opaque(self.node)

    if node.hasItems { return }
    if node.hasChildren {
      _descendToLeftmostItem(ofChildAtSlot: .zero)
    } else {
      self._o.isAtEnd = true
    }
  }
}

extension _HashTreeIterator: IteratorProtocol {
  @inlinable
  internal mutating func next(
  ) -> (node: _UnmanagedHashNode, slot: _HashSlot)? {
    guard slot < endSlot else {
      return _next()
    }
    defer { slot = slot.next() }
    return (node, slot)
  }

  @usableFromInline
  @_effects(releasenone)
  internal mutating func _next(
  ) -> (node: _UnmanagedHashNode, slot: _HashSlot)? {
    if _o.isAtEnd { return nil }
    if node.hasChildren {
      _descendToLeftmostItem(ofChildAtSlot: .zero)
      slot = slot.next()
      return (node, .zero)
    }
    while !_o.level.isAtRoot {
      let nextChild = _ascend().next()
      if nextChild < node.childrenEndSlot {
        _descendToLeftmostItem(ofChildAtSlot: nextChild)
        slot = slot.next()
        return (node, .zero)
      }
    }
    // At end
    endSlot = node.itemsEndSlot
    slot = endSlot
    _o.isAtEnd = true
    return nil
  }
}

extension _HashTreeIterator {
  internal mutating func _descend(toChildSlot childSlot: _HashSlot) {
    assert(childSlot < node.childrenEndSlot)
    _o.ancestorSlots[_o.level] = childSlot
    _o.ancestorNodes.push(node)
    _o.level = _o.level.descend()
    node = node.unmanagedChild(at: childSlot)
    slot = .zero
    endSlot = node.itemsEndSlot
  }

  internal mutating func _ascend() -> _HashSlot {
    assert(!_o.level.isAtRoot)
    node = _o.ancestorNodes.pop()
    _o.level = _o.level.ascend()
    let childSlot = _o.ancestorSlots[_o.level]
    _o.ancestorSlots.clear(_o.level)
    return childSlot
  }

  internal mutating func _descendToLeftmostItem(
    ofChildAtSlot childSlot: _HashSlot
  ) {
    _descend(toChildSlot: childSlot)
    while endSlot == .zero {
      assert(node.hasChildren)
      _descend(toChildSlot: .zero)
    }
  }
}
