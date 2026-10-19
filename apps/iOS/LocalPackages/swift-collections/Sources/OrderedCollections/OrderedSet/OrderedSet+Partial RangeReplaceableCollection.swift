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

// The parts of RangeReplaceableCollection that OrderedSet is able to implement.

extension OrderedSet {
  /// Removes all members from the set.
  ///
  /// - Parameter keepingCapacity: If `true`, the set's storage capacity is
  ///   preserved; if `false`, the underlying storage is released. The default
  ///   is `false`.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    _elements.removeAll(keepingCapacity: keepCapacity)
    guard keepCapacity else {
      _table = nil
      return
    }
    guard _table != nil else { return }
    _ensureUnique()
    _table!.update { hashTable in
      hashTable.clear()
    }
  }

  /// Removes and returns the element at the specified position.
  ///
  /// All the elements following the specified position are moved to close the
  /// resulting gap.
  ///
  /// - Parameter index: The position of the element to remove. `index` must be
  ///    a valid index of the collection that is not equal to the collection's
  ///    end index.
  ///
  /// - Returns: The removed element.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  @discardableResult
  public mutating func remove(at index: Int) -> Self.Element {
    _elements._failEarlyRangeCheck(index, bounds: startIndex ..< endIndex)
    let bucket = _bucket(for: index)
    return _removeExistingMember(at: index, in: bucket)
  }

  /// Removes the specified subrange of elements from the collection.
  ///
  /// All the elements following the specified subrange are moved to close the
  /// resulting gap.
  ///
  /// - Parameter bounds: The subrange of the collection to remove. The bounds
  ///   of the range must be valid indices of the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeSubrange(_ bounds: Range<Int>) {
    _elements._failEarlyRangeCheck(
      bounds,
      bounds: _elements.startIndex ..< _elements.endIndex)
    guard _table != nil else {
      _elements.removeSubrange(bounds)
      _checkInvariants()
      return
    }
    let c = bounds.count
    guard c > 0 else { return }
    let remainingCount = _elements.count - c
    if remainingCount <= count / 2 || remainingCount < _minimumCapacity {
      // Just generate a new table from scratch.
      _elements.removeSubrange(bounds)
      _regenerateHashTable()
      _checkInvariants()
      return
    }

    _ensureUnique()
    _table!.update { hashTable in
      // Delete the hash table entries for all members we're removing.
      for item in _elements[bounds] {
        let (offset, bucket) = hashTable._find(item, in: _elements)
        precondition(offset != nil, "Corrupt hash table")
        hashTable.delete(
          bucket: bucket,
          hashValueGenerator: { offset, seed in
            return _elements[offset]._rawHashValue(seed: seed)
          })
      }
      hashTable.adjustContents(preparingForRemovalOf: bounds, in: _elements)
    }
    _elements.removeSubrange(bounds)
    _checkInvariants()
  }

  /// Removes the specified subrange of elements from the collection.
  ///
  /// All the elements following the specified subrange are moved to close the
  /// resulting gap.
  ///
  /// - Parameter bounds: The subrange of the collection to remove. The bounds
  ///   of the range must be valid indices of the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeSubrange(_ bounds: some RangeExpression<Int>) {
    removeSubrange(bounds.relative(to: self))
  }

  /// Removes the last element of a non-empty set.
  ///
  /// - Complexity: Expected to be O(`1`) on average, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  @discardableResult
  public mutating func removeLast() -> Element {
    precondition(!isEmpty, "Cannot remove last element of an empty collection")
    guard _table != nil else {
      return _elements.removeLast()
    }
    guard _elements.count - 1 >= _minimumCapacity else {
      let old = _elements.removeLast()
      _regenerateHashTable()
      return old
    }
    defer { _checkInvariants() }
    let old = _elements.removeLast()
    _ensureUnique()
    _table!.update { hashTable in
      var it = hashTable.bucketIterator(for: old)
      it.advance(until: _elements.count)
      // Delete the entry for the removed member.
      hashTable.delete(
        bucket: it.currentBucket,
        hashValueGenerator: { offset, seed in
          _elements[offset]._rawHashValue(seed: seed)
        })
    }
    return old
  }

  /// Removes the last `n` element of the set.
  ///
  /// - Parameter n: The number of elements to remove from the collection.
  ///   `n` must be greater than or equal to zero and must not exceed the
  ///   number of elements in the collection.
  ///
  /// - Complexity: Expected to be O(`n`) on average, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  public mutating func removeLast(_ n: Int) {
    precondition(n >= 0, "Can't remove a negative number of elements")
    precondition(n <= count, "Can't remove more elements than there are in the collection")
    removeSubrange(count - n ..< count)
  }

  /// Removes the first element of a non-empty set.
  ///
  /// The members following the removed item need to be moved to close the
  /// resulting gap in the storage array.
  ///
  /// - Complexity: O(`count`).
  @inlinable
  @discardableResult
  public mutating func removeFirst() -> Element {
    precondition(!isEmpty, "Cannot remove first element of an empty collection")
    return remove(at: startIndex)
  }

  /// Removes the first `n` elements of the set.
  ///
  /// The members following the removed items need to be moved to close the
  /// resulting gap in the storage array.
  ///
  /// - Parameter n: The number of elements to remove from the collection.
  ///   `n` must be greater than or equal to zero and must not exceed the
  ///   number of elements in the set.
  ///
  /// - Complexity: O(`count`).
  @inlinable
  public mutating func removeFirst(_ n: Int) {
    precondition(n >= 0, "Can't remove a negative number of elements")
    precondition(n <= count, "Can't remove more elements than there are in the collection")
    removeSubrange(0 ..< n)
  }

  /// Removes all the elements that satisfy the given predicate.
  ///
  /// Use this method to remove every element in a collection that meets
  /// particular criteria. The order of the remaining elements is preserved.
  ///
  /// This example removes all the odd values from an
  /// array of numbers:
  ///
  ///     var numbers: OrderedSet = [5, 6, 7, 8, 9, 10, 11]
  ///     numbers.removeAll(where: { !$0.isMultiple(of: 2) })
  ///     // numbers == [6, 8, 10]
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   set as its argument and returns a Boolean value indicating
  ///   whether the element should be removed from the set.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(
    where shouldBeRemoved: (Element) throws -> Bool
  ) rethrows {
    defer {
      _regenerateHashTable()
      _checkInvariants()
    }
    try _elements.removeAll(where: shouldBeRemoved)
  }
}
