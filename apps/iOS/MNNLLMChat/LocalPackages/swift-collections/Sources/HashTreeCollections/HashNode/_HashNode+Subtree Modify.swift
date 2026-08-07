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

// MARK: Subtree-level in-place mutation operations

extension _HashNode {
  @inlinable
  internal mutating func ensureUnique(
    level: _HashLevel, at path: _UnsafePath
  ) -> (leaf: _UnmanagedHashNode, slot: _HashSlot) {
    ensureUnique(isUnique: isUnique())
    guard level < path.level else { return (unmanaged, path.currentItemSlot) }
    return update {
      $0[child: path.childSlot(at: level)]
        .ensureUnique(level: level.descend(), at: path)
    }
  }
}

extension _HashNode {
  @usableFromInline
  @frozen
  internal struct ValueUpdateState {
    @usableFromInline
    internal var key: Key

    @usableFromInline
    internal var value: Value?

    @usableFromInline
    internal let hash: _Hash

    @usableFromInline
    internal var path: _UnsafePath

    @usableFromInline
    internal var found: Bool

    @inlinable
    internal init(
      _ key: Key,
      _ hash: _Hash,
      _ path: _UnsafePath
    ) {
      self.key = key
      self.value = nil
      self.hash = hash
      self.path = path
      self.found = false
    }
  }

  @inlinable
  internal mutating func prepareValueUpdate(
    _ key: Key,
    _ hash: _Hash
  ) -> ValueUpdateState {
    var state = ValueUpdateState(key, hash, _UnsafePath(root: raw))
    _prepareValueUpdate(&state)
    return state
  }

  @inlinable
  internal mutating func _prepareValueUpdate(
    _ state: inout ValueUpdateState
  ) {
    // This doesn't make room for a new item if the key doesn't already exist
    // but it does ensure that all parent nodes along its eventual path are
    // uniquely held.
    //
    // If the key already exists, we ensure uniqueness for its node and extract
    // its item but otherwise leave the tree as it was.
    let isUnique = self.isUnique()
    let r = findForInsertion(state.path.level, state.key, state.hash)
    switch r {
    case .found(_, let slot):
      ensureUnique(isUnique: isUnique)
      state.path.node = unmanaged
      state.path.selectItem(at: slot)
      state.found = true
      (state.key, state.value) = update { $0.itemPtr(at: slot).move() }


    case .insert(_, let slot):
      state.path.selectItem(at: slot)

    case .appendCollision:
      state.path.selectItem(at: _HashSlot(self.count))

    case .spawnChild(_, let slot):
      state.path.selectItem(at: slot)

    case .expansion:
      state.path.selectEnd()

    case .descend(_, let slot):
      ensureUnique(isUnique: isUnique)
      update {
        let p = $0.childPtr(at: slot)
        state.path.descendToChild(p.pointee.unmanaged, at: slot)
        p.pointee._prepareValueUpdate(&state)
      }
    }
  }

  @inlinable
  internal mutating func finalizeValueUpdate(
    _ state: __owned ValueUpdateState
  ) {
    switch (state.found, state.value != nil) {
    case (true, true):
      // Fast path: updating an existing value.
      UnsafeHandle.update(state.path.node) {
        $0.itemPtr(at: state.path.currentItemSlot)
          .initialize(to: (state.key, state.value.unsafelyUnwrapped))
      }
    case (true, false):
      // Removal
      let remainder = _finalizeRemoval(.top, state.hash, at: state.path)
      assert(remainder == nil)
    case (false, true):
      // Insertion
      let r = updateValue(.top, forKey: state.key, state.hash) {
        $0.initialize(to: (state.key, state.value.unsafelyUnwrapped))
      }
      assert(r.inserted)
    case (false, false):
      // Noop
      break
    }
  }

  @inlinable
  internal mutating func _finalizeRemoval(
    _ level: _HashLevel, _ hash: _Hash, at path: _UnsafePath
  ) -> Element? {
    assert(isUnique())
    if level == path.level {
      return _removeItemFromUniqueLeafNode(
        level, at: hash[level], path.currentItemSlot, by: { _ in }
      ).remainder
    }
    let slot = path.childSlot(at: level)
    let remainder = update {
      $0[child: slot]._finalizeRemoval(level.descend(), hash, at: path)
    }
    return _fixupUniqueAncestorAfterItemRemoval(
      level, at: { _ in hash[level] }, slot, remainder: remainder)
  }
}

extension _HashNode {
  @usableFromInline
  @frozen
  internal struct DefaultedValueUpdateState {
    @usableFromInline
    internal var item: Element

    @usableFromInline
    internal var node: _UnmanagedHashNode

    @usableFromInline
    internal var slot: _HashSlot

    @usableFromInline
    internal var inserted: Bool

    @inlinable
    internal init(
      _ item: Element,
      in node: _UnmanagedHashNode,
      at slot: _HashSlot,
      inserted: Bool
    ) {
      self.item = item
      self.node = node
      self.slot = slot
      self.inserted = inserted
    }
  }

  @inlinable
  internal mutating func prepareDefaultedValueUpdate(
    _ level: _HashLevel,
    _ key: Key,
    _ defaultValue: () -> Value,
    _ hash: _Hash
  ) -> DefaultedValueUpdateState {
    let isUnique = self.isUnique()
    let r = findForInsertion(level, key, hash)
    switch r {
    case .found(_, let slot):
      ensureUnique(isUnique: isUnique)
      return DefaultedValueUpdateState(
        update { $0.itemPtr(at: slot).move() },
        in: unmanaged,
        at: slot,
        inserted: false)

    case .insert(let bucket, let slot):
      ensureUniqueAndInsertItem(
        isUnique: isUnique, at: bucket, itemSlot: slot
      ) { _ in }
      return DefaultedValueUpdateState(
        (key, defaultValue()),
        in: unmanaged,
        at: slot,
        inserted: true)

    case .appendCollision:
      let slot = ensureUniqueAndAppendCollision(isUnique: isUnique) { _ in }
      return DefaultedValueUpdateState(
        (key, defaultValue()),
        in: unmanaged,
        at: slot,
        inserted: true)

    case .spawnChild(let bucket, let slot):
      let r = ensureUniqueAndSpawnChild(
        isUnique: isUnique,
        level: level,
        replacing: bucket,
        itemSlot: slot,
        newHash: hash) { _ in }
      return DefaultedValueUpdateState(
        (key, defaultValue()),
        in: r.leaf,
        at: r.slot,
        inserted: true)

    case .expansion:
      let r = _HashNode.build(
        level: level,
        item1: { _ in }, hash,
        child2: self, self.collisionHash
      )
      self = r.top
      return DefaultedValueUpdateState(
        (key, defaultValue()),
        in: r.leaf,
        at: r.slot1,
        inserted: true)

    case .descend(_, let slot):
      ensureUnique(isUnique: isUnique)
      let res = update {
        $0[child: slot].prepareDefaultedValueUpdate(
          level.descend(), key, defaultValue, hash)
      }
      if res.inserted { count &+= 1 }
      return res
    }
  }

  @inlinable
  internal mutating func finalizeDefaultedValueUpdate(
    _ state: __owned DefaultedValueUpdateState
  ) {
    UnsafeHandle.update(state.node) {
      $0.itemPtr(at: state.slot).initialize(to: state.item)
    }
  }
}

