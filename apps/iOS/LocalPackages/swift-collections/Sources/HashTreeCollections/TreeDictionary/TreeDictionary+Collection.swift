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

extension TreeDictionary {
  /// The position of an element in a persistent dictionary.
  ///
  /// An index in a persistent dictionary is a compact encoding of a path in
  /// the underlying prefix tree. Such indices are valid until the tree
  /// structure is changed; hence, indices are usually invalidated every time
  /// the dictionary gets mutated.
  @frozen
  public struct Index {
    @usableFromInline
    internal let _root: _UnmanagedHashNode

    @usableFromInline
    internal var _version: UInt

    @usableFromInline
    internal var _path: _UnsafePath

    @inlinable @inline(__always)
    internal init(
      _root: _UnmanagedHashNode, version: UInt, path: _UnsafePath
    ) {
      self._root = _root
      self._version = version
      self._path = path
    }
  }
}

extension TreeDictionary.Index: @unchecked Sendable
where Key: Sendable, Value: Sendable {}

extension TreeDictionary.Index: Equatable {
  /// Returns a Boolean value indicating whether two index values are equal.
  ///
  /// Note that comparing two indices that do not belong to the same tree
  /// leads to a runtime error.
  ///
  /// - Complexity: O(1)
  @inlinable
  public static func ==(left: Self, right: Self) -> Bool {
    precondition(
      left._root == right._root && left._version == right._version,
      "Indices from different dictionary values aren't comparable")
    return left._path == right._path
  }
}

extension TreeDictionary.Index: Comparable {
  /// Returns a Boolean value indicating whether the value of the first argument
  /// is less than the second argument.
  ///
  /// Note that comparing two indices that do not belong to the same tree
  /// leads to a runtime error.
  ///
  /// - Complexity: O(1)
  public static func <(left: Self, right: Self) -> Bool {
    precondition(
      left._root == right._root && left._version == right._version,
      "Indices from different dictionary values aren't comparable")
    return left._path < right._path
  }
}

extension TreeDictionary.Index: Hashable {
  /// Hashes the essential components of this value by feeding them into the
  /// given hasher.
  ///
  /// - Complexity: O(1)
  @inlinable
  public func hash(into hasher: inout Hasher) {
    hasher.combine(_path)
  }
}

extension TreeDictionary.Index: CustomStringConvertible {
  // A textual representation of this instance.
  public var description: String {
    _path.description
  }
}

extension TreeDictionary.Index: CustomDebugStringConvertible {
  /// A textual representation of this instance, suitable for debugging.
  public var debugDescription: String {
    description
  }
}

extension TreeDictionary: Collection {
  /// A Boolean value indicating whether the collection is empty.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var isEmpty: Bool {
    _root.count == 0
  }

  /// The number of elements in the collection.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var count: Int {
    _root.count
  }

  /// The position of the first element in a nonempty collection, or `endIndex`
  /// if the collection is empty.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var startIndex: Index {
    var path = _UnsafePath(root: _root.raw)
    path.descendToLeftMostItem()
    return Index(_root: _root.unmanaged, version: _version, path: path)
  }

  /// The collection’s “past the end” position—that is, the position one greater
  /// than the last valid subscript argument.
  ///
  /// - Complexity: O(1)
  @inlinable
  public var endIndex: Index {
    var path = _UnsafePath(root: _root.raw)
    path.selectEnd()
    return Index(_root: _root.unmanaged, version: _version, path: path)
  }

  @inlinable @inline(__always)
  internal func _isValid(_ i: Index) -> Bool {
    _root.isIdentical(to: i._root) && i._version == self._version
  }

  @inlinable @inline(__always)
  internal mutating func _invalidateIndices() {
    _version &+= 1
  }

  /// Accesses the key-value pair at the specified position.
  ///
  /// - Parameter position: The position of the element to access. `position`
  ///    must be a valid index of the collection that is not equal to
  ///    `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  public subscript(i: Index) -> Element {
    precondition(_isValid(i), "Invalid index")
    precondition(i._path.isOnItem, "Can't get element at endIndex")
    return _UnsafeHandle.read(i._path.node) {
      $0[item: i._path.currentItemSlot]
    }
  }

  /// Replaces the given index with its successor.
  ///
  /// - Parameter i: A valid index of the collection.
  ///     `i` must be less than `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  public func formIndex(after i: inout Index) {
    precondition(_isValid(i), "Invalid index")
    guard i._path.findSuccessorItem(under: _root.raw) else {
      preconditionFailure("The end index has no successor")
    }
  }

  /// Returns the position immediately after the given index.
  ///
  /// - Parameter i: A valid index of the collection.
  ///    `i` must be less than `endIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func index(after i: Index) -> Index {
    var i = i
    formIndex(after: &i)
    return i
  }

  /// Returns the distance between two arbitrary valid indices in this
  /// collection.
  ///
  /// - Parameter start: A valid index of the collection.
  /// - Parameter end: Another valid index of the collection.
  /// - Returns: The distance between `start` and `end`.
  ///    (The result can be negative, even though `TreeDictionary` is not
  ///    a bidirectional collection.)
  /// - Complexity: O(log(`count`))
  @inlinable
  public func distance(from start: Index, to end: Index) -> Int {
    precondition(_isValid(start) && _isValid(end), "Invalid index")
    return _root.raw.distance(.top, from: start._path, to: end._path)
  }

  /// Returns an index that is the specified distance from the given index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection.
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`. As a special exception,
  ///     `distance` is allowed to be negative even though
  ///     `TreeDictionary` isn't a bidirectional collection.
  /// - Returns: An index offset by `distance` from the index `i`. If
  ///   `distance` is positive, this is the same value as the result of
  ///   `distance` calls to `index(after:)`. If distance is negative, then
  ///   `distance` calls to `index(after:)` on the returned value will be the
  ///   same as `start`.
  ///
  /// - Complexity: O(log(`distance`))
  @inlinable
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    precondition(_isValid(i), "Invalid index")
    var i = i
    let r = _root.raw.seek(.top, &i._path, offsetBy: distance)
    precondition(r, "Index offset out of bounds")
    return i
  }

  /// Returns an index that is the specified distance from the given index,
  /// unless that distance is beyond a given limiting index.
  ///
  /// The value passed as `distance` must not offset `i` beyond the bounds of
  /// the collection, unless the index passed as `limit` prevents offsetting
  /// beyond those bounds.
  ///
  /// - Parameters:
  ///   - i: A valid index of the collection.
  ///   - distance: The distance to offset `i`. As a special exception,
  ///     `distance` is allowed to be negative even though
  ///     `TreeDictionary` isn't a bidirectional collection.
  ///   - limit: A valid index of the collection to use as a limit. If
  ///     `distance > 0`, a limit that is less than `i` has no effect.
  ///     Likewise, if `distance < 0`, a limit that is greater than `i` has no
  ///     effect.
  /// - Returns: An index offset by `distance` from the index `i`, unless that
  ///   index would be beyond `limit` in the direction of movement. In that
  ///   case, the method returns `nil`.
  ///
  /// - Complexity: O(log(`distance`))
  @inlinable
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    precondition(_isValid(i), "Invalid index")
    precondition(_isValid(limit), "Invalid limit index")
    var i = i
    let (found, limited) = _root.raw.seek(
      .top, &i._path, offsetBy: distance, limitedBy: limit._path
    )
    if found { return i }
    precondition(limited, "Index offset out of bounds")
    return nil
  }
}

#if false
// Note: Let's not do this. `BidirectionalCollection` would imply that
// the ordering of elements would be meaningful, which isn't true for
// `TreeDictionary`.
extension TreeDictionary: BidirectionalCollection {
  /// Replaces the given index with its predecessor.
  ///
  /// - Parameter i: A valid index of the collection.
  ///     `i` must be greater than `startIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable
  public func formIndex(before i: inout Index) {
    precondition(_isValid(i), "Invalid index")
    guard i._path.findPredecessorItem(under: _root.raw) else {
      preconditionFailure("The start index has no predecessor")
    }
  }

  /// Returns the position immediately before the given index.
  ///
  /// - Parameter i: A valid index of the collection.
  ///    `i` must be greater than `startIndex`.
  ///
  /// - Complexity: O(1)
  @inlinable @inline(__always)
  public func index(before i: Index) -> Index {
    var i = i
    formIndex(before: &i)
    return i
  }
}
#endif
