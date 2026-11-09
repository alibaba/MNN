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
  /// Append a new item to the end of the set, assuming it's not already
  /// a member.
  ///
  /// In optimized builds, this does not check for duplicates.
  ///
  /// - Parameter item: The item to add to the set. The item must no be already
  ///    a member.
  /// - Complexity: Expected to be O(1) on average if `Element`
  ///    implements high-quality hashing.
  @inlinable
  internal mutating func _appendNew(_ item: Element) {
    assert(!contains(item), "Duplicate item")
    _elements.append(item)
    guard _elements.count <= _capacity else {
      _regenerateHashTable()
      return
    }
    guard _table != nil else { return }
    _ensureUnique()
    _table!.update { hashTable in
      var it = hashTable.bucketIterator(for: item)
      it.advanceToNextUnoccupiedBucket()
      it.currentValue = _elements.count - 1
    }
  }

  /// Append a new member to the end of the set, registering it in the
  /// specified hash table bucket, without verifying that the set
  /// doesn't already contain it.
  ///
  /// This operation performs no hashing operations unless it needs to
  /// reallocate the hash table.
  ///
  /// - Complexity: Amortized O(1)
  @inlinable
  internal mutating func _appendNew(_ item: Element, in bucket: _Bucket) {
    _elements.append(item)

    guard _elements.count <= _capacity else {
      _regenerateHashTable()
      return
    }
    guard _table != nil else { return }
    _ensureUnique()
    _table!.update { hashTable in
      assert(!hashTable.isOccupied(bucket))
      hashTable[bucket] = _elements.count - 1
    }
  }

  @inlinable
  @discardableResult
  internal mutating func _append(
    _ item: Element
  ) -> (inserted: Bool, index: Int) {
    let (index, bucket) = _find(item)
    if let index = index { return (false, index) }
    _appendNew(item, in: bucket)
    return (true, _elements.index(before: _elements.endIndex))
  }

  /// Append a new member to the end of the set, if the set doesn't
  /// already contain it.
  ///
  /// - Parameter item: The element to add to the set.
  ///
  /// - Returns: A pair `(inserted, index)`, where `inserted` is a Boolean value
  ///    indicating whether the operation added a new element, and `index` is
  ///    the index of `item` in the resulting set.
  ///
  /// - Complexity: The operation is expected to perform O(1) copy, hash, and
  ///    compare operations on the `Element` type, if it implements high-quality
  ///    hashing.
  @inlinable
  @inline(__always)
  @discardableResult
  public mutating func append(_ item: Element) -> (inserted: Bool, index: Int) {
    let result = _append(item)
    _checkInvariants()
    return result
  }

  /// Append the contents of a sequence to the end of the set, excluding
  /// elements that are already members.
  ///
  /// This is functionally equivalent to `self.formUnion(elements)`, but it's
  /// more explicit about how the new members are ordered in the new set.
  ///
  /// - Parameter elements: A finite sequence of elements to append.
  ///
  /// - Complexity: The operation is expected to perform amortized O(1) copy,
  ///    hash, and compare operations on the `Element` type, if it implements
  ///    high-quality hashing.
  @inlinable
  public mutating func append(
    contentsOf elements: some Sequence<Element>
  ) {
    for item in elements {
      _append(item)
    }
    _checkInvariants()
  }
}

extension OrderedSet {
  @inlinable
  internal mutating func _insertNew(
    _ item: Element,
    at index: Int,
    in bucket: _Bucket
  ) {
    guard _elements.count < _capacity else {
      _elements.insert(item, at: index)
      _regenerateHashTable()
      return
    }
    guard _table != nil else {
      _elements.insert(item, at: index)
      return
    }

    _ensureUnique()
    _table!.update { hashTable in
      assert(!hashTable.isOccupied(bucket))
      hashTable.adjustContents(preparingForInsertionOfElementAtOffset: index, in: _elements)
      hashTable[bucket] = index
    }
    _elements.insert(item, at: index)
    _checkInvariants()
  }

  /// Insert a new member to this set at the specified index, if the set doesn't
  /// already contain it.
  ///
  /// - Parameter item: The element to insert.
  ///
  /// - Returns: A pair `(inserted, index)`, where `inserted` is a Boolean value
  ///    indicating whether the operation added a new element, and `index` is
  ///    the index of `item` in the resulting set. If `inserted` is false, then
  ///    the returned `index` may be different from the index requested.
  ///
  /// - Complexity: The operation is expected to perform amortized
  ///    O(`self.count`) copy, hash, and compare operations on the `Element`
  ///    type, if it implements high-quality hashing. (Insertions need to make
  ///    room in the storage array to add the inserted element.)
  @inlinable
  @discardableResult
  public mutating func insert(
    _ item: Element,
    at index: Int
  ) -> (inserted: Bool, index: Int) {
    let (existing, bucket) = _find(item)
    if let existing = existing { return (false, existing) }
    _insertNew(item, at: index, in: bucket)
    return (true, index)
  }
}

extension OrderedSet {
  /// Replace the member at the given index with a new value that compares equal
  /// to it.
  ///
  /// This is useful when equal elements can be distinguished by identity
  /// comparison or some other means.
  ///
  /// - Parameter item: The new value that should replace the original element.
  ///     `item` must compare equal to the original value.
  ///
  /// - Parameter index: The index of the element to be replaced.
  ///
  /// - Returns: The original element that was replaced.
  ///
  /// - Complexity: Amortized O(1).
  @inlinable
  @discardableResult
  public mutating func update(_ item: Element, at index: Int) -> Element {
    let old = _elements[index]
    precondition(
      item == old,
      "The replacement item must compare equal to the original")
    _elements[index] = item
    return old
  }
}

extension OrderedSet {
  /// Adds the given element to the set unconditionally, either appending it to
  /// the set, or replacing an existing value if it's already present.
  ///
  /// This is useful when equal elements can be distinguished by identity
  /// comparison or some other means.
  ///
  /// - Parameter item: The value to append or replace.
  ///
  /// - Returns: The original element that was replaced by this operation, or
  ///    `nil` if the value was appended to the end of the collection.
  ///
  /// - Complexity: The operation is expected to perform amortized O(1) copy,
  ///    hash, and compare operations on the `Element` type, if it implements
  ///    high-quality hashing.
  @inlinable
  @discardableResult
  public mutating func updateOrAppend(_ item: Element) -> Element? {
    let (inserted, index) = _append(item)
    if inserted { return nil }
    let old = _elements[index]
    _elements[index] = item
    _checkInvariants()
    return old
  }

  /// Adds the given element into the set unconditionally, either inserting it
  /// at the specified index, or replacing an existing value if it's already
  /// present.
  ///
  /// This is useful when equal elements can be distinguished by identity
  /// comparison or some other means.
  ///
  /// - Parameter item: The value to append or replace.
  ///
  /// - Parameter index: The index at which to insert the new member if `item`
  ///    isn't already in the set.
  ///
  /// - Returns: The original element that was replaced by this operation, or
  ///    `nil` if the value was newly inserted into the collection.
  ///
  /// - Complexity: The operation is expected to perform amortized O(1) copy,
  ///    hash, and compare operations on the `Element` type, if it implements
  ///    high-quality hashing.
  @inlinable
  @discardableResult
  public mutating func updateOrInsert(
    _ item: Element,
    at index: Int
  ) -> (originalMember: Element?, index: Int) {
    let (existing, bucket) = _find(item)
    if let existing = existing {
      let old = _elements[existing]
      _elements[existing] = item
      return (old, existing)
    }
    _insertNew(item, at: index, in: bucket)
    return (nil, index)
  }
}
