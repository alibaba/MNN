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

@usableFromInline
@frozen // Not really! This module isn't ABI stable.
internal struct _RopeStorageHeader {
  @usableFromInline var _childCount: UInt16
  @usableFromInline let height: UInt8

  @inlinable
  internal init(height: UInt8) {
    self._childCount = 0
    self.height = height
  }

  @inlinable
  internal var childCount: Int {
    get {
      numericCast(_childCount)
    }
    set {
      _childCount = numericCast(newValue)
    }
  }
}

extension Rope {
  @usableFromInline
  @_fixed_layout // Not really! This module isn't ABI stable.
  internal final class _Storage<Child: _RopeItem<Summary>>:
    ManagedBuffer<_RopeStorageHeader, Child>
  {
    @usableFromInline internal typealias Summary = Element.Summary
    @usableFromInline internal typealias _UnsafeHandle = Rope._UnsafeHandle

    @inlinable
    internal static func create(height: UInt8) -> _Storage {
      let object = create(minimumCapacity: Summary.maxNodeSize) { _ in .init(height: height) }
      return unsafeDowncast(object, to: _Storage.self)
    }

    @inlinable
    deinit {
      withUnsafeMutablePointers { h, p in
        p.deinitialize(count: h.pointee.childCount)
        h.pointee._childCount = .max
      }
    }
  }
}
