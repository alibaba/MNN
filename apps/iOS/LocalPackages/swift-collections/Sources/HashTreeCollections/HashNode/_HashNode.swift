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

/// A node in the hash tree, logically representing a hash table with
/// 32 buckets, corresponding to a 5-bit slice of a full hash value.
///
/// Each bucket may store either a single key-value pair or a reference
/// to a child node that contains additional items.
///
/// To save space, children and items are stored compressed into the two
/// ends of a single raw storage buffer, with the free space in between
/// available for use by either side.
///
/// The storage is arranged as shown below.
///
///     ┌───┬───┬───┬───┬───┬──────────────────┬───┬───┬───┬───┐
///     │ 0 │ 1 │ 2 │ 3 │ 4 │→   free space   ←│ 3 │ 2 │ 1 │ 0 │
///     └───┴───┴───┴───┴───┴──────────────────┴───┴───┴───┴───┘
///      children                                         items
///
/// Note that the region for items grows downwards from the end, so the item
/// at slot 0 is at the very end of the buffer.
///
/// Two 32-bit bitmaps are used to associate each child and item with their
/// position in the hash table. The bucket occupied by the *n*th child in the
/// buffer above is identified by position of the *n*th true bit in the child
/// map, and the *n*th item's bucket corresponds to the *n*th true bit in the
/// items map.
@usableFromInline
@frozen
internal struct _HashNode<Key: Hashable, Value> {
  // Warning: This struct must be kept layout-compatible with _RawHashNode.
  // Do not add any new stored properties to this type.
  //
  // Swift guarantees that frozen structs with a single stored property will
  // be layout-compatible with the type they are wrapping.
  //
  // (_RawHashNode is used as the Element type of the ManagedBuffer underlying
  // node storage, and the memory is then rebound to `_HashNode` later.
  // This will not work correctly unless `_HashNode` has the exact same alignment
  // and stride as `RawNode`.)

  @usableFromInline
  internal typealias Element = (key: Key, value: Value)

  @usableFromInline
  internal var raw: _RawHashNode

  @inlinable
  internal init(storage: _RawHashStorage, count: Int) {
    self.raw = _RawHashNode(storage: storage, count: count)
  }
}

extension _HashNode {
  @inlinable @inline(__always)
  internal var count: Int {
    get { raw.count }
    set { raw.count = newValue }
  }
}

extension _HashNode {
  @inlinable @inline(__always)
  internal var unmanaged: _UnmanagedHashNode {
    _UnmanagedHashNode(raw.storage)
  }

  @inlinable @inline(__always)
  internal func isIdentical(to other: _UnmanagedHashNode) -> Bool {
    raw.isIdentical(to: other)
  }
}

extension _HashNode {
  @inlinable @inline(__always)
  internal func read<R>(
    _ body: (UnsafeHandle) throws -> R
  ) rethrows -> R {
    try UnsafeHandle.read(raw.storage, body)
  }

  @inlinable @inline(__always)
  internal mutating func update<R>(
    _ body: (UnsafeHandle) throws -> R
  ) rethrows -> R {
    try UnsafeHandle.update(raw.storage, body)
  }
}

// MARK: Shortcuts to reading header data

extension _HashNode {
  @inlinable
  internal var isCollisionNode: Bool {
    read { $0.isCollisionNode }
  }

  @inlinable
  internal var collisionHash: _Hash {
    read { $0.collisionHash }
  }

  @inlinable
  internal var hasSingletonItem: Bool {
    read { $0.hasSingletonItem }
  }

  @inlinable
  internal var hasSingletonChild: Bool {
    read { $0.hasSingletonChild }
  }

  @inlinable
  internal var isAtrophied: Bool {
    read { $0.isAtrophiedNode }
  }
}

extension _HashNode {
  @inlinable
  internal var initialVersionNumber: UInt {
    // Ideally we would simply just generate a true random number, but the
    // memory address of the root node is a reasonable substitute.
    // Alternatively, we could use a per-thread counter along with a thread
    // id, or some sort of global banks of atomic integer counters.
    let address = Unmanaged.passUnretained(raw.storage).toOpaque()
    return UInt(bitPattern: address)
  }
}
