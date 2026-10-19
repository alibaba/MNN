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

extension _RawHashNode {
  /// An unsafe, non-generic view of the data stored inside a node in the
  /// hash tree, hiding the mechanics of accessing storage from the code that
  /// uses it.
  ///
  /// This is the non-generic equivalent of `_HashNode.UnsafeHandle`, sharing some
  /// of its functionality, but it only provides read-only access to the tree
  /// structure (incl. subtree counts) -- it doesn't provide any ways to mutate
  /// the underlying data or to access user payload.
  ///
  /// Handles do not own the storage they access -- it is the client's
  /// responsibility to ensure that handles (and any pointer values generated
  /// by them) do not escape the closure call that received them.
  @usableFromInline
  @frozen
  internal struct UnsafeHandle {
    @usableFromInline
    internal let _header: UnsafePointer<_HashNodeHeader>

    @usableFromInline
    internal let _memory: UnsafeRawPointer

    @inlinable
    internal init(
      _ header: UnsafePointer<_HashNodeHeader>,
      _ memory: UnsafeRawPointer
    ) {
      self._header = header
      self._memory = memory
    }
  }
}

extension _RawHashNode.UnsafeHandle {
  @inlinable @inline(__always)
  static func read<R>(
    _ node: _UnmanagedHashNode,
    _ body: (Self) throws -> R
  ) rethrows -> R {
    try node.ref._withUnsafeGuaranteedRef { storage in
      try storage.withUnsafeMutablePointers { header, elements in
        try body(Self(header, UnsafeRawPointer(elements)))
      }
    }
  }
}

extension _RawHashNode.UnsafeHandle {
  @inline(__always)
  internal var isCollisionNode: Bool {
    _header.pointee.isCollisionNode
  }

  @inline(__always)
  internal var collisionHash: _Hash {
    assert(isCollisionNode)
    return _memory.load(as: _Hash.self)
  }

  @inline(__always)
  internal var hasChildren: Bool {
    _header.pointee.hasChildren
  }

  @inline(__always)
  internal var childCount: Int {
    _header.pointee.childCount
  }

  @inline(__always)
  internal var childrenEndSlot: _HashSlot {
    _header.pointee.childrenEndSlot
  }

  @inline(__always)
  internal var hasItems: Bool {
    _header.pointee.hasItems
  }

  @inline(__always)
  internal var itemCount: Int {
    _header.pointee.itemCount
  }

  @inline(__always)
  internal var itemsEndSlot: _HashSlot {
    _header.pointee.itemsEndSlot
  }

  @inline(__always)
  internal var _childrenStart: UnsafePointer<_RawHashNode> {
    _memory.assumingMemoryBound(to: _RawHashNode.self)
  }

  internal subscript(child slot: _HashSlot) -> _RawHashNode {
    unsafeAddress {
      assert(slot < childrenEndSlot)
      return _childrenStart + slot.value
    }
  }

  internal var children: UnsafeBufferPointer<_RawHashNode> {
    UnsafeBufferPointer(start: _childrenStart, count: childCount)
  }
}
