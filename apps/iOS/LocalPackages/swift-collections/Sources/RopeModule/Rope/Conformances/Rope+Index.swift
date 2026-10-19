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
  public struct Index: @unchecked Sendable {
    @usableFromInline internal typealias Summary = Rope.Summary
    @usableFromInline internal typealias _Path = Rope._Path

    @usableFromInline
    internal var _version: _RopeVersion

    @usableFromInline
    internal var _path: _Path

    /// A direct reference to the leaf node addressed by this index.
    /// This must only be dereferenced while we own a tree with a matching
    /// version.
    @usableFromInline
    internal var _leaf: _UnmanagedLeaf?

    @inlinable
    internal init(
      version: _RopeVersion, path: _Path, leaf: __shared _UnmanagedLeaf?
    ) {
      self._version = version
      self._path = path
      self._leaf = leaf
    }
  }
}

extension Rope.Index {
  @inlinable
  internal static var _invalid: Self {
    Self(version: _RopeVersion(0), path: _RopePath(_value: .max), leaf: nil)
  }

  @inlinable
  internal var _isValid: Bool {
    _path._value != .max
  }
}

extension Rope.Index: Equatable {
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    left._path == right._path
  }
}
extension Rope.Index: Hashable {
  @inlinable
  public func hash(into hasher: inout Hasher) {
    hasher.combine(_path)
  }
}

extension Rope.Index: Comparable {
  @inlinable
  public static func <(left: Self, right: Self) -> Bool {
    left._path < right._path
  }
}

extension Rope.Index: CustomStringConvertible {
  public var description: String {
    "\(_path)"
  }
}

extension Rope.Index {
  @inlinable
  internal var _height: UInt8 {
    _path.height
  }

  @inlinable
  internal func _isEmpty(below height: UInt8) -> Bool {
    _path.isEmpty(below: height)
  }

  @inlinable
  internal mutating func _clear(below height: UInt8) {
    _path.clear(below: height)
  }
}
