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

extension OrderedSet {
  /// Exposes some private implementation details and low-level unsafe
  /// operations, primarily to allow clear box testing.
  ///
  /// This struct is a private implementation detail, therefore it and its
  /// members are not covered by any source compatibility promises -- they
  /// may disappear in any new release.
  @frozen
  public struct _UnstableInternals {
    @usableFromInline
    internal typealias _Bucket = _HashTable.Bucket

    @usableFromInline
    internal var base: OrderedSet

    @inlinable
    init(_ base: OrderedSet) {
      self.base = base
    }
  }

  @inlinable
  public var __unstable: _UnstableInternals {
    @inline(__always)
    get {
      _UnstableInternals(self)
    }

    @inline(__always) // https://github.com/apple/swift-collections/issues/164
    _modify {
      var view = _UnstableInternals(self)
      self = OrderedSet()
      defer { self = view.base }
      yield &view
    }
  }
}

extension OrderedSet._UnstableInternals: Sendable
where Element: Sendable {}
