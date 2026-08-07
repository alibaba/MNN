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

extension TreeSet: _UniqueCollection {}

extension TreeSet {
  /// Removes the element at the given valid index.
  ///
  /// Calling this method invalidates all existing indices of the collection.
  ///
  /// - Parameter position: The index of the member to remove. `position` must
  ///    be a valid index of the set, and it must not be equal to the set’s
  ///    end index.
  /// - Returns: The element that was removed from the set.
  /// - Complexity: O(log(`count`)) if set storage might be shared; O(1)
  ///    otherwise.
  @discardableResult
  public mutating func remove(at position: Index) -> Element {
    precondition(_isValid(position))
    _invalidateIndices()
    let r = _root.remove(.top, at: position._path)
    precondition(r.remainder == nil)
    return r.removed.key
  }

  /// Replace the member at the given index with a new value that compares equal
  /// to it.
  ///
  /// This is useful when equal elements can be distinguished by identity
  /// comparison or some other means. Updating a member through this method
  /// does not require any hashing operations.
  ///
  /// Calling this method invalidates all existing indices of the collection.
  ///
  /// - Parameter item: The new value that should replace the original element.
  ///     `item` must compare equal to the original value.
  ///
  /// - Parameter index: The index of the element to be replaced.
  ///
  /// - Returns: The original element that was replaced.
  ///
  /// - Complexity: O(log(`count`)) if set storage might be shared; O(1)
  ///    otherwise.
  public mutating func update(_ member: Element, at index: Index) -> Element {
    defer { _fixLifetime(self) }
    precondition(_isValid(index), "Invalid index")
    precondition(index._path.isOnItem, "Can't get element at endIndex")
    _invalidateIndices()
    return _UnsafeHandle.update(index._path.node) {
      let p = $0.itemPtr(at: index._path.currentItemSlot)
      var old = member
      precondition(
        member == p.pointee.key,
        "The replacement item must compare equal to the original")
      swap(&p.pointee.key, &old)
      return old
    }
  }
}
