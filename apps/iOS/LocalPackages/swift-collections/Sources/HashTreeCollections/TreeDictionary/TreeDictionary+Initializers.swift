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

extension TreeDictionary {
  /// Creates an empty dictionary.
  ///
  /// This initializer is equivalent to initializing with an empty dictionary
  /// literal.
  ///
  /// - Complexity: O(1)
  @inlinable
  public init() {
    self.init(_new: ._emptyNode())
  }

  /// Makes a copy of an existing persistent dictionary.
  ///
  /// - Complexity: O(1)
  @inlinable
  public init(_ other: TreeDictionary<Key, Value>) {
    self = other
  }

  /// Creates a new persistent dictionary that contains the same key-value
  /// pairs as the given `Dictionary` instance, although not necessarily
  /// in the same order.
  ///
  /// - Complexity: O(`other.count`)
  @inlinable
  public init(_ other: Dictionary<Key, Value>) {
    self.init(_uniqueKeysWithValues: other)
  }

  /// Creates a new persistent dictionary by associating the given persistent
  /// set of keys with the values generated using the specified closure.
  ///
  /// - Complexity: O(`other.count`)
  @inlinable
  public init(
    keys: TreeSet<Key>,
    valueGenerator valueTransform: (Key) throws -> Value
  ) rethrows {
    // FIXME: This is a non-standard addition
    let root = try keys._root.mapValues { try valueTransform($0.key) }
    self.init(_new: root)
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence.
  ///
  /// You use this initializer to create a dictionary when you have a sequence
  /// of key-value tuples with unique keys. Passing a sequence with duplicate
  /// keys to this initializer results in a runtime error. If your
  /// sequence might have duplicate keys, use the
  /// `Dictionary(_:uniquingKeysWith:)` initializer instead.
  ///
  /// - Parameter keysAndValues: A sequence of key-value pairs to use for
  ///   the new dictionary. Every key in `keysAndValues` must be unique.
  ///
  /// - Returns: A new dictionary initialized with the elements of
  ///   `keysAndValues`.
  ///
  /// - Precondition: The sequence must not have duplicate keys.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count if
  ///    key-value pairs, if `Key` properly implements hashing.
  @inlinable
  public init(
    uniqueKeysWithValues keysAndValues: some Sequence<(Key, Value)>
  ) {
    self.init()
    for item in keysAndValues {
      let hash = _Hash(item.0)
      let r = _root.insert(.top, item, hash)
      precondition(r.inserted, "Duplicate key: '\(item.0)'")
    }
    _invariantCheck()
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence.
  ///
  /// You use this initializer to create a dictionary when you have a sequence
  /// of key-value tuples with unique keys. Passing a sequence with duplicate
  /// keys to this initializer results in a runtime error. If your
  /// sequence might have duplicate keys, use the
  /// `Dictionary(_:uniquingKeysWith:)` initializer instead.
  ///
  /// - Parameter keysAndValues: A sequence of key-value pairs to use for
  ///   the new dictionary. Every key in `keysAndValues` must be unique.
  ///
  /// - Returns: A new dictionary initialized with the elements of
  ///   `keysAndValues`.
  ///
  /// - Precondition: The sequence must not have duplicate keys.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count if
  ///    key-value pairs, if `Key` properly implements hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  public init(
    uniqueKeysWithValues keysAndValues: some Sequence<Element>
  ) {
    if let keysAndValues = _specialize(keysAndValues, for: Self.self) {
      self = keysAndValues
      return
    }
    if let keysAndValues = _specialize(
      keysAndValues, for: Dictionary<Key, Value>.self
    ) {
      self.init(keysAndValues)
      return
    }
    self.init(_uniqueKeysWithValues: keysAndValues)
  }

  @inlinable
  internal init(
    _uniqueKeysWithValues keysAndValues: some Sequence<Element>
  ) {
    self.init()
    for item in keysAndValues {
      let hash = _Hash(item.key)
      let r = _root.insert(.top, item, hash)
      precondition(r.inserted, "Duplicate key: '\(item.key)'")
    }
    _invariantCheck()
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence,
  /// using a combining closure to determine the value for any duplicate keys.
  ///
  /// You use this initializer to create a dictionary when you have a sequence
  /// of key-value tuples that might have duplicate keys. As the dictionary is
  /// built, the initializer calls the `combine` closure with the current and
  /// new values for any duplicate keys. Pass a closure as `combine` that
  /// returns the value to use in the resulting dictionary: The closure can
  /// choose between the two values, combine them to produce a new value, or
  /// even throw an error.
  ///
  ///     let pairsWithDuplicateKeys = [("a", 1), ("b", 2), ("a", 3), ("b", 4)]
  ///
  ///     let firstValues = TreeDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (first, _) in first })
  ///     // ["a": 1, "b": 2]
  ///
  ///     let lastValues = TreeDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (_, last) in last })
  ///     // ["a": 3, "b": 4]
  ///
  /// - Parameters:
  ///   - keysAndValues: A sequence of key-value pairs to use for the new
  ///     dictionary.
  ///   - combine: A closure that is called with the values for any duplicate
  ///     keys that are encountered. The closure returns the desired value for
  ///     the final dictionary.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count of
  ///    key-value pairs, if `Key` properly implements hashing.
  public init(
    _ keysAndValues: some Sequence<(Key, Value)>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    self.init()
    try self.merge(keysAndValues, uniquingKeysWith: combine)
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence,
  /// using a combining closure to determine the value for any duplicate keys.
  ///
  /// You use this initializer to create a dictionary when you have a sequence
  /// of key-value tuples that might have duplicate keys. As the dictionary is
  /// built, the initializer calls the `combine` closure with the current and
  /// new values for any duplicate keys. Pass a closure as `combine` that
  /// returns the value to use in the resulting dictionary: The closure can
  /// choose between the two values, combine them to produce a new value, or
  /// even throw an error.
  ///
  ///     let pairsWithDuplicateKeys = [("a", 1), ("b", 2), ("a", 3), ("b", 4)]
  ///
  ///     let firstValues = TreeDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (first, _) in first })
  ///     // ["a": 1, "b": 2]
  ///
  ///     let lastValues = TreeDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (_, last) in last })
  ///     // ["a": 3, "b": 4]
  ///
  /// - Parameters:
  ///   - keysAndValues: A sequence of key-value pairs to use for the new
  ///     dictionary.
  ///   - combine: A closure that is called with the values for any duplicate
  ///     keys that are encountered. The closure returns the desired value for
  ///     the final dictionary.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count of
  ///    key-value pairs, if `Key` properly implements hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  public init(
    _ keysAndValues: some Sequence<Element>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    try self.init(
      keysAndValues.lazy.map { ($0.key, $0.value) },
      uniquingKeysWith: combine)
  }
}

extension TreeDictionary {
  /// Creates a new dictionary whose keys are the groupings returned by the
  /// given closure and whose values are arrays of the elements that returned
  /// each key.
  ///
  /// The arrays in the "values" position of the new dictionary each contain at
  /// least one element, with the elements in the same order as the source
  /// sequence.
  ///
  /// The following example declares an array of names, and then creates a
  /// dictionary from that array by grouping the names by first letter:
  ///
  ///     let students = ["Kofi", "Abena", "Efua", "Kweku", "Akosua"]
  ///     let studentsByLetter = TreeDictionary(grouping: students, by: { $0.first! })
  ///     // ["K": ["Kofi", "Kweku"], "A": ["Abena", "Akosua"], "E": ["Efua"]]
  ///
  /// The new `studentsByLetter` dictionary has three entries, with students'
  /// names grouped by the keys `"E"`, `"K"`, and `"A"`.
  ///
  /// - Parameters:
  ///   - values: A sequence of values to group into a dictionary.
  ///   - keyForValue: A closure that returns a key for each element in
  ///     `values`.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count of
  ///    values, if `Key` properly implements hashing.
  @inlinable @inline(__always)
  public init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows
  where Value: RangeReplaceableCollection, Value.Element == S.Element
  {
    try self.init(_grouping: values, by: keyForValue)
  }

  /// Creates a new dictionary whose keys are the groupings returned by the
  /// given closure and whose values are arrays of the elements that returned
  /// each key.
  ///
  /// The arrays in the "values" position of the new dictionary each contain at
  /// least one element, with the elements in the same order as the source
  /// sequence.
  ///
  /// The following example declares an array of names, and then creates a
  /// dictionary from that array by grouping the names by first letter:
  ///
  ///     let students = ["Kofi", "Abena", "Efua", "Kweku", "Akosua"]
  ///     let studentsByLetter = TreeDictionary(grouping: students, by: { $0.first! })
  ///     // ["K": ["Kofi", "Kweku"], "A": ["Abena", "Akosua"], "E": ["Efua"]]
  ///
  /// The new `studentsByLetter` dictionary has three entries, with students'
  /// names grouped by the keys `"E"`, `"K"`, and `"A"`.
  ///
  /// - Parameters:
  ///   - values: A sequence of values to group into a dictionary.
  ///   - keyForValue: A closure that returns a key for each element in
  ///     `values`.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count of
  ///    values, if `Key` properly implements hashing.
  @inlinable @inline(__always)
  public init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows
  where Value == [S.Element]
  {
    // Note: this extra overload is necessary to make type inference work
    // for the `Value` type -- we want it to default to `[S.Element`].
    // (https://github.com/apple/swift-collections/issues/139)
    try self.init(_grouping: values, by: keyForValue)
  }

  @inlinable
  internal init<S: Sequence>(
    _grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows
  where Value: RangeReplaceableCollection, Value.Element == S.Element
  {
    self.init()
    for value in values {
      let key = try keyForValue(value)
      self.updateValue(forKey: key, default: Value()) { array in
        array.append(value)
      }
    }
    _invariantCheck()
  }
}
