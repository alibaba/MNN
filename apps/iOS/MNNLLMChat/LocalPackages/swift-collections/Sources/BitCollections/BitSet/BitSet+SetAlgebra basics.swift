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

extension BitSet {
  /// Returns a Boolean value that indicates whether the given element exists
  /// in the set.
  ///
  /// - Parameter element: An element to look for in the set.
  ///
  /// - Returns: `true` if `member` exists in the set; otherwise, `false`.
  ///
  /// - Complexity: O(1)
  @usableFromInline
  internal func _contains(_ member: UInt) -> Bool {
    _read { $0.contains(member) }
  }

  /// Returns a Boolean value that indicates whether the given element exists
  /// in the set.
  ///
  /// - Parameter element: An element to look for in the set.
  ///
  /// - Returns: `true` if `member` exists in the set; otherwise, `false`.
  ///
  /// - Complexity: O(1)
  public func contains(_ member: Int) -> Bool {
    guard let member = UInt(exactly: member) else { return false }
    return _contains(member)
  }
}

extension BitSet {
  /// Insert the given element in the set if it is not already present.
  ///
  /// If an element equal to `newMember` is already contained in the set, this
  /// method has no effect.
  ///
  /// If `newMember` was not already a member, it gets inserted into the set.
  ///
  /// - Parameter newMember: An element to insert into the set.
  ///
  /// - Returns: True if `newMember` was not contained in the
  ///    set, false otherwise.
  ///
  /// - Complexity: O(1) if the set is a unique value (with no other copies),
  ///     and the inserted value is not greater than the largest value currently
  ///     contained in the set (named *max*). Otherwise the complexity is
  ///     O(max(`newMember`, *max*)).
  @discardableResult
  @usableFromInline
  internal mutating func _insert(_ newMember: UInt) -> Bool {
    _ensureCapacity(forValue: newMember)
    return _update { $0.insert(newMember) }
  }

  /// Inserts the given element in the set if it is not already present.
  ///
  /// If an element equal to `newMember` is already contained in the set, this
  /// method has no effect.
  ///
  /// If `newMember` was not already a member, it gets inserted into the set.
  ///
  /// - Parameter newMember: An element to insert into the set.
  ///
  /// - Returns: `(true, newMember)` if `newMember` was not contained in the
  ///    set. If `newMember` was already contained in the set, the method
  ///    returns `(false, newMember)`.
  ///
  /// - Complexity: O(1) if the set is a unique value (with no other copies),
  ///     and the inserted value is not greater than the largest value currently
  ///     contained in the set (named *max*). Otherwise the complexity is
  ///     O(max(`newMember`, *max*)).
  @discardableResult
  public mutating func insert(
    _ newMember: Int
  ) -> (inserted: Bool, memberAfterInsert: Int) {
    guard let i = UInt(exactly: newMember) else {
      preconditionFailure("Value out of range")
    }
    return (_insert(i), newMember)
  }

  /// Inserts the given element into the set unconditionally.
  ///
  /// - Parameter newMember: An element to insert into the set.
  ///
  /// - Returns: `newMember` if the set already contained it; otherwise, `nil`.
  ///
  /// - Complexity: O(1) if the set is a unique value (with no live copies),
  ///     and the inserted value is not greater than the largest value currently
  ///     contained in the set (named *max*). Otherwise the complexity is
  ///     O(max(`newMember`, *max*)).
  @discardableResult
  public mutating func update(with newMember: Int) -> Int? {
    insert(newMember).inserted ? newMember : nil
  }
}

extension BitSet {
  @discardableResult
  @usableFromInline
  internal mutating func _remove(_ member: UInt) -> Bool {
    _updateThenShrink { handle, shrink in
      shrink = handle.remove(member)
      return shrink
    }
  }

  /// Removes the given element from the set.
  ///
  /// - Parameter member: The element of the set to remove.
  ///
  /// - Returns: `member` if it was contained in the set; otherwise, `nil`.
  ///
  /// - Complexity: O(`1`) if the set is a unique value (with no live copies),
  ///    and the removed value is less than the largest value currently in the
  ///    set (named *max*). Otherwise the complexity is at worst O(*max*).
  @discardableResult
  public mutating func remove(_ member: Int) -> Int? {
    guard let m = UInt(exactly: member) else { return nil }
    return _remove(m) ? member : nil
  }
}
