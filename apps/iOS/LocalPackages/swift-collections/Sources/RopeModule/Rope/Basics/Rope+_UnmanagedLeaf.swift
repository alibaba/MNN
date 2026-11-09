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
  @frozen // Not really! This module isn't ABI stable.
  @usableFromInline
  internal struct _UnmanagedLeaf {
    @usableFromInline internal typealias _Item = Rope._Item
    @usableFromInline internal typealias _Leaf = _Storage<_Item>
    @usableFromInline internal typealias _UnsafeHandle = Rope._UnsafeHandle

    @usableFromInline var _ref: Unmanaged<_Leaf>

    @inlinable
    internal init(_ leaf: __shared _Leaf) {
      _ref = .passUnretained(leaf)
    }
  }
}

extension Rope._UnmanagedLeaf: Equatable {
  @inlinable
  internal static func ==(left: Self, right: Self) -> Bool {
    left._ref.toOpaque() == right._ref.toOpaque()
  }
}

extension Rope._UnmanagedLeaf {
  @inlinable
  internal func read<R>(
    body: (_UnsafeHandle<_Item>) -> R
  ) -> R {
    _ref._withUnsafeGuaranteedRef { leaf in
      leaf.withUnsafeMutablePointers { h, p in
        let handle = _UnsafeHandle(isMutable: false, header: h, start: p)
        return body(handle)
      }
    }
  }
}
