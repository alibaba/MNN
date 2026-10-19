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
  @usableFromInline
  @frozen
  internal struct Builder {
    @usableFromInline internal typealias Element = _HashNode.Element

    @usableFromInline
    @frozen
    internal enum Kind {
      case empty
      case item(Element, at: _Bucket)
      case node(_HashNode)
      case collisionNode(_HashNode)
    }

    @usableFromInline
    internal var level: _HashLevel

    @usableFromInline
    internal var kind: Kind

    @inlinable
    internal init(_ level: _HashLevel, _ kind: Kind) {
      self.level = level
      self.kind = kind
    }
  }
}

extension _HashNode.Builder {
  @usableFromInline
  internal func dump() {
    let head = "Builder(level: \(level.depth), kind: "
    switch self.kind {
    case .empty:
      print(head + "empty)")
    case .item(let item, at: let bucket):
      print(head + "item(\(_HashNode._itemString(for: item)), at: \(bucket))")
    case .node(let node):
      print(head + "node)")
      node.dump()
    case .collisionNode(let node):
      print(head + "collisionNode)")
      node.dump()
    }
  }
}

extension _HashNode.Builder {
  @inlinable @inline(__always)
  internal static func empty(_ level: _HashLevel) -> Self {
    Self(level, .empty)
  }

  @inlinable @inline(__always)
  internal static func item(
    _ level: _HashLevel, _ item: __owned Element, at bucket: _Bucket
  ) -> Self {
    Self(level, .item(item, at: bucket))
  }

  @inlinable @inline(__always)
  internal static func node(
    _ level: _HashLevel, _ node: __owned _HashNode
  ) -> Self {
    assert(!node.isCollisionNode)
    return Self(level, .node(node))
  }

  @inlinable @inline(__always)
  internal static func collisionNode(
    _ level: _HashLevel, _ node: __owned _HashNode
  ) -> Self {
    assert(node.isCollisionNode)
    return Self(level, .collisionNode(node))
  }
}

extension _HashNode.Builder {
  @inlinable
  internal var count: Int {
    switch kind {
    case .empty:
      return 0
    case .item:
      return 1
    case .node(let node):
      return node.count
    case .collisionNode(let node):
      return node.count
    }
  }

  @inlinable
  internal var isEmpty: Bool {
    guard case .empty = kind else { return false }
    return true
  }
}

extension _HashNode.Builder {
  @inlinable
  internal init(_ level: _HashLevel, _ node: _HashNode) {
    self.level = level
    if node.count == 0 {
      kind = .empty
    } else if node.isCollisionNode {
      assert(!node.hasSingletonItem)
      kind = .collisionNode(node)
    } else if node.hasSingletonItem {
      kind = node.read { .item($0[item: .zero], at: $0.itemMap.first!) }
    } else {
      kind = .node(node)
    }
  }

  @inlinable
  internal __consuming func finalize(_ level: _HashLevel) -> _HashNode {
    assert(level.isAtRoot && self.level.isAtRoot)
    switch kind {
    case .empty:
      return ._emptyNode()
    case .item(let item, let bucket):
      return ._regularNode(item, bucket)
    case .node(let node):
      return node
    case .collisionNode(let node):
      return node
    }
  }
}

extension _HashNode {
  @inlinable
  internal mutating func applyReplacement(
    _ level: _HashLevel,
    _ replacement: Builder
  ) -> Element? {
    assert(level == replacement.level)
    switch replacement.kind {
    case .empty:
      self = ._emptyNode()
    case .node(let n), .collisionNode(let n):
      self = n
    case .item(let item, let bucket):
      guard level.isAtRoot else {
        self = ._emptyNode()
        return item
      }
      self = ._regularNode(item, bucket)
    }
    return nil
  }
}

extension _HashNode.Builder {
  @inlinable
  internal mutating func addNewCollision(
    _ level: _HashLevel, _ newItem: __owned Element, _ hash: _Hash
  ) {
    assert(level == self.level)
    switch kind {
    case .empty:
      kind = .item(newItem, at: hash[level])
    case .item(let oldItem, at: let bucket):
      assert(hash[level] == bucket)
      let node = _HashNode._collisionNode(hash, oldItem, newItem)
      kind = .collisionNode(node)
    case .collisionNode(var node):
      kind = .empty
      assert(node.isCollisionNode)
      assert(hash == node.collisionHash)
      _ = node.ensureUniqueAndAppendCollision(isUnique: true, newItem)
      kind = .collisionNode(node)
    case .node:
      fatalError()
    }
  }

  @inlinable
  internal mutating func addNewItem(
    _ level: _HashLevel, _ newItem: __owned Element, at newBucket: _Bucket
  ) {
    assert(level == self.level)
    switch kind {
    case .empty:
      kind = .item(newItem, at: newBucket)
    case .item(let oldItem, let oldBucket):
      assert(oldBucket != newBucket)
      let node = _HashNode._regularNode(oldItem, oldBucket, newItem, newBucket)
      kind = .node(node)
    case .node(var node):
      kind = .empty
      let isUnique = node.isUnique()
      node.ensureUniqueAndInsertItem(isUnique: isUnique, newItem, at: newBucket)
      kind = .node(node)
    case .collisionNode(var node):
      // Expansion
      assert(!level.isAtBottom)
      self.kind = .empty
      node = _HashNode._regularNode(
        newItem, newBucket, node, node.collisionHash[level])
      kind = .node(node)
    }
  }

  @inlinable
  internal mutating func addNewChildNode(
    _ level: _HashLevel, _ newChild: __owned _HashNode, at newBucket: _Bucket
  ) {
    assert(level == self.level)
    switch self.kind {
    case .empty:
      if newChild.isCollisionNode {
        // Compression
        assert(!level.isAtBottom)
        self.kind = .collisionNode(newChild)
      } else {
        self.kind = .node(._regularNode(newChild, newBucket))
      }
    case let .item(oldItem, oldBucket):
      let node = _HashNode._regularNode(oldItem, oldBucket, newChild, newBucket)
      self.kind = .node(node)
    case .node(var node):
      self.kind = .empty
      let isUnique = node.isUnique()
      node.ensureUnique(
        isUnique: isUnique, withFreeSpace: _HashNode.spaceForNewChild)
      node.insertChild(newChild, newBucket)
      self.kind = .node(node)
    case .collisionNode(var node):
      // Expansion
      self.kind = .empty
      assert(!level.isAtBottom)
      node = _HashNode._regularNode(
        node, node.collisionHash[level], newChild, newBucket)
      self.kind = .node(node)
    }
  }

  @inlinable
  internal mutating func addNewChildBranch(
    _ level: _HashLevel, _ newChild: __owned Self, at newBucket: _Bucket
  ) {
    assert(level == self.level)
    assert(newChild.level == self.level.descend())
    switch newChild.kind {
    case .empty:
      break
    case .item(let newItem, _):
      self.addNewItem(level, newItem, at: newBucket)
    case .node(let newNode), .collisionNode(let newNode):
      self.addNewChildNode(level, newNode, at: newBucket)
    }
  }

  @inlinable
  internal static func childBranch(
    _ level: _HashLevel, _ child: Self, at bucket: _Bucket
  ) -> Self {
    assert(child.level == level.descend())
    switch child.kind {
    case .empty:
      return self.empty(level)
    case .item(let item, _):
      return self.item(level, item, at: bucket)
    case .node(let n):
      return self.node(level, ._regularNode(n, bucket))
    case .collisionNode(let node):
      // Compression
      assert(!level.isAtBottom)
      return self.collisionNode(level, node)
    }
  }
}

extension _HashNode.Builder {
  @inlinable
  internal mutating func copyCollisions(
    from source: _HashNode.UnsafeHandle,
    upTo end: _HashSlot
  ) {
    assert(isEmpty)
    assert(source.isCollisionNode)
    assert(end < source.itemsEndSlot)
    let h = source.collisionHash
    for slot: _HashSlot in stride(from: .zero, to: end, by: 1) {
      self.addNewCollision(self.level, source[item: slot], h)
    }
  }

  @inlinable
  internal mutating func copyItems(
    _ level: _HashLevel,
    from source: _HashNode.UnsafeHandle,
    upTo end: _Bucket
  ) {
    assert(level == self.level)
    assert(isEmpty)
    assert(!source.isCollisionNode)
    for (b, s) in source.itemMap.intersection(_Bitmap(upTo: end)) {
      self.addNewItem(level, source[item: s], at: b)
    }
  }

  @inlinable
  internal mutating func copyItemsAndChildren(
    _ level: _HashLevel,
    from source: _HashNode.UnsafeHandle,
    upTo end: _Bucket
  ) {
    assert(level == self.level)
    assert(isEmpty)
    assert(!source.isCollisionNode)
    for (b, s) in source.itemMap {
      self.addNewItem(level, source[item: s], at: b)
    }
    for (b, s) in source.childMap.intersection(_Bitmap(upTo: end)) {
      self.addNewChildNode(level, source[child: s], at: b)
    }
  }
}

extension _HashNode.Builder {
  @inlinable
  internal func mapValues<Value2>(
    _ transform: (Element) -> Value2
  ) -> _HashNode<Key, Value2>.Builder {
    switch kind {
    case .empty:
      return .empty(level)
    case let .item(item, at: bucket):
      let value = transform(item)
      return .item(level, (item.key, value), at: bucket)
    case let .node(node):
      return .node(level, node.mapValues(transform))
    case let .collisionNode(node):
      return .collisionNode(level, node.mapValues(transform))
    }
  }

  @inlinable
  internal func mapValuesToVoid() -> _HashNode<Key, Void>.Builder {
    if Value.self == Void.self {
      return unsafeBitCast(self, to: _HashNode<Key, Void>.Builder.self)
    }
    return mapValues { _ in () }
  }
}
