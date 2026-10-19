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

extension TreeSet {
  /// Creates an empty set.
  ///
  /// This initializer is equivalent to initializing with an empty array
  /// literal.
  ///
  /// - Complexity: O(1)
  @inlinable
  public init() {
    self.init(_new: ._emptyNode())
  }

  /// Creates a new set from a finite sequence of items.
  ///
  /// - Parameter items: The elements to use as members of the new set.
  ///    The sequence is allowed to contain duplicate elements, but only
  ///    the first duplicate instance is preserved in the result.
  ///
  /// - Complexity: This operation is expected to perform O(*n*)
  ///    hashing and equality comparisons on average (where *n*
  ///    is the number of elements in the sequence), provided that
  ///    `Element` properly implements hashing.
  @inlinable
  public init(_ items: __owned some Sequence<Element>) {
    if let items = _specialize(items, for: Self.self) {
      self = items
      return
    }
    self.init()
    for item in items {
      self._insert(item)
    }
  }

  /// Creates a new set from a an existing set. This is functionally the same as
  /// copying the value of `items` into a new variable.
  ///
  /// - Parameter items: The elements to use as members of the new set.
  ///
  /// - Complexity: O(1)
  @inlinable
  public init(_ items: __owned Self) {
    self = items
  }

  /// Creates a new persistent set from the keys view of an existing persistent
  /// dictionary.
  ///
  /// - Parameter items: The elements to use as members of the new set.
  ///
  /// - Complexity: O(*items.count*)
  @inlinable
  public init<Value>(
    _ item: __owned TreeDictionary<Element, Value>.Keys
  ) {
    self.init(_new: item._base._root.mapValues { _ in () })
  }
}
