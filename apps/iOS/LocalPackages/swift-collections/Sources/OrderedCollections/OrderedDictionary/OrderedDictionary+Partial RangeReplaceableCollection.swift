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

// The parts of RangeReplaceableCollection that OrderedDictionary is able to implement.

extension OrderedDictionary {
  /// Reserves enough space to store the specified number of elements.
  ///
  /// This method ensures that the dictionary has unique, mutable, contiguous
  /// storage, with space allocated for at least the requested number of
  /// elements.
  ///
  /// If you are adding a known number of elements to a dictionary, call this
  /// method once before the first insertion to avoid multiple reallocations.
  ///
  /// Do not call this method in a loop -- it does not use an exponential
  /// allocation strategy, so doing that can result in quadratic instead of
  /// linear performance.
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the
  ///   dictionary should be able to store without reallocating its storage.
  ///
  /// - Complexity: O(`max(count, minimumCapacity)`)
  @inlinable
  public mutating func reserveCapacity(_ minimumCapacity: Int) {
    self._keys.reserveCapacity(minimumCapacity)
    self._values.reserveCapacity(minimumCapacity)
  }

  /// Removes all members from the dictionary.
  ///
  /// - Parameter keepingCapacity: If `true`, the dictionary's storage capacity
  ///   is preserved; if `false`, the underlying storage is released. The
  ///   default is `false`.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    _keys.removeAll(keepingCapacity: keepCapacity)
    _values.removeAll(keepingCapacity: keepCapacity)
  }

  /// Removes and returns the key-value pair at the specified index.
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
  public mutating func remove(at index: Int) -> Element {
    let key = _keys.remove(at: index)
    let value = _values.remove(at: index)
    return (key, value)
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
    _keys.removeSubrange(bounds)
    _values.removeSubrange(bounds)
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
  public mutating func removeSubrange(
    _ bounds: some RangeExpression<Int>
  ) {
    removeSubrange(bounds.relative(to: elements))
  }


  /// Removes the last element of a non-empty dictionary.
  ///
  /// - Complexity: Expected to be O(`1`) on average, if `Element` implements
  ///    high-quality hashing.
  @inlinable
  @discardableResult
  public mutating func removeLast() -> Element {
    precondition(!isEmpty, "Cannot remove last element of an empty collection")
    return remove(at: count - 1)
  }

  /// Removes the last `n` element of the dictionary.
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
    _keys.removeLast(n)
    _values.removeLast(n)
  }

  /// Removes the first element of a non-empty dictionary.
  ///
  /// The members following the removed key-value pair need to be moved to close
  /// the resulting gaps in the storage arrays.
  ///
  /// - Complexity: O(`count`).
  @inlinable
  @discardableResult
  public mutating func removeFirst() -> Element {
    precondition(!isEmpty, "Cannot remove first element of an empty collection")
    return remove(at: 0)
  }

  /// Removes the first `n` elements of the dictionary.
  ///
  /// The members following the removed items need to be moved to close the
  /// resulting gaps in the storage arrays.
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
    _keys.removeFirst(n)
    _values.removeFirst(n)
  }

  /// Removes all the elements that satisfy the given predicate.
  ///
  /// Use this method to remove every element in a collection that meets
  /// particular criteria. The order of the remaining elements is preserved.
  ///
  /// - Parameter shouldBeRemoved: A closure that takes an element of the
  ///   dictionary as its argument and returns a Boolean value indicating
  ///   whether the element should be removed from the collection.
  ///
  /// - Complexity: O(`count`)
  @inlinable
  public mutating func removeAll(
    where shouldBeRemoved: (Self.Element) throws -> Bool
  ) rethrows {
    let pivot = try _values.withUnsafeMutableBufferPointer { values in
      try _keys._halfStablePartition(
        values: values,
        by: shouldBeRemoved)
    }
    removeSubrange(pivot...)
    _checkInvariants()
  }
}

