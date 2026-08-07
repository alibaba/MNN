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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

/// An unsafe, unowned, type-erased reference to a hash tree node; essentially
/// just a lightweight wrapper around `Unmanaged<_RawHashStorage>`.
///
/// Because such a reference may outlive the underlying object, use sites must
/// be extraordinarily careful to never dereference an invalid
/// `_UnmanagedHashNode`. Doing so results in undefined behavior.
@usableFromInline
@frozen
internal struct _UnmanagedHashNode {
  @usableFromInline
  internal var ref: Unmanaged<_RawHashStorage>

  @inlinable @inline(__always)
  internal init(_ storage: _RawHashStorage) {
    self.ref = .passUnretained(storage)
  }
}

extension _UnmanagedHashNode: Equatable {
  /// Indicates whether two unmanaged node references are equal.
  ///
  /// This function is safe to call even if one or both of its arguments are
  /// invalid -- however, it may incorrectly return true in this case.
  /// (This can happen when a destroyed node's memory region is later reused for
  /// a newly created node.)
  @inlinable
  internal static func ==(left: Self, right: Self) -> Bool {
    left.ref.toOpaque() == right.ref.toOpaque()
  }
}

extension _UnmanagedHashNode: CustomStringConvertible {
  @usableFromInline
  internal var description: String {
    _addressString(for: ref.toOpaque())
  }
}

extension _UnmanagedHashNode {
  @inlinable @inline(__always)
  internal func withRaw<R>(_ body: (_RawHashStorage) -> R) -> R {
    ref._withUnsafeGuaranteedRef(body)
  }

  @inline(__always)
  internal func read<R>(_ body: (_RawHashNode.UnsafeHandle) -> R) -> R {
    ref._withUnsafeGuaranteedRef { storage in
      storage.withUnsafeMutablePointers { header, elements in
        body(_RawHashNode.UnsafeHandle(header, UnsafeRawPointer(elements)))
      }
    }
  }

  @inlinable
  internal var hasItems: Bool {
    withRaw { $0.header.hasItems }
  }

  @inlinable
  internal var hasChildren: Bool {
    withRaw { $0.header.hasChildren }
  }

  @inlinable
  internal var itemCount: Int {
    withRaw { $0.header.itemCount }
  }

  @inlinable
  internal var childCount: Int {
    withRaw { $0.header.childCount }
  }

  @inlinable
  internal var itemsEndSlot: _HashSlot {
    withRaw { _HashSlot($0.header.itemCount) }
  }

  @inlinable
  internal var childrenEndSlot: _HashSlot {
    withRaw { _HashSlot($0.header.childCount) }
  }

  @inlinable
  internal func unmanagedChild(at slot: _HashSlot) -> Self {
    withRaw { raw in
      assert(slot.value < raw.header.childCount)
      return raw.withUnsafeMutablePointerToElements { p in
        Self(p[slot.value].storage)
      }
    }
  }
}

