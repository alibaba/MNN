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

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension OrderedDictionary {
  /// Creates an empty dictionary.
  ///
  /// This initializer is equivalent to initializing with an empty dictionary
  /// literal.
  ///
  /// - Complexity: O(1)
  @inlinable
  @inline(__always)
  public init() {
    self._keys = OrderedSet()
    self._values = []
  }

  /// Creates an empty dictionary with preallocated space for at least the
  /// specified number of elements.
  ///
  /// Use this initializer to avoid intermediate reallocations of a dictionary's
  /// storage buffer when you know in advance how many elements you'll insert
  /// into it after creation.
  ///
  /// If you have a good idea of the expected working size of the dictionary,
  /// calling this initializer with `persistent` set to true can sometimes
  /// improve performance by eliminating churn due to repeated rehashings when
  /// the dictionary temporarily shrinks below its regular size. You can cancel
  /// any capacity you've previously reserved by persistently reserving a
  /// capacity of zero. (This also shrinks the hash table to the ideal size for
  /// its current number elements.)
  ///
  /// - Parameter minimumCapacity: The minimum number of elements that the newly
  ///   created dictionary should be able to store without reallocating its
  ///   storage.
  ///
  /// - Parameter persistent: If set to true, prevent removals from shrinking
  ///   storage below the specified capacity. By default, removals are allowed
  ///   to shrink storage below any previously reserved capacity.
  ///
  /// - Complexity: O(`minimumCapacity`)
  @inlinable
  @inline(__always)
  public init(minimumCapacity: Int, persistent: Bool = false) {
    self._keys = OrderedSet(minimumCapacity: minimumCapacity, persistent: persistent)
    self._values = []
    _values.reserveCapacity(minimumCapacity)
  }
}

extension OrderedDictionary {
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  public init(
    uniqueKeysWithValues keysAndValues: some Sequence<(key: Key, value: Value)>
  ) {
    if let keysAndValues = _specialize(
      keysAndValues, for: Dictionary<Key, Value>.self
    ) {
      self.init(_uncheckedUniqueKeysWithValues: keysAndValues)
      return
    }
    self.init()
    reserveCapacity(keysAndValues.underestimatedCount)
    for (key, value) in keysAndValues {
      guard _keys._append(key).inserted else {
        preconditionFailure("Duplicate key: '\(key)'")
      }
      _values.append(value)
    }
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @inlinable
  public init(
    uniqueKeysWithValues keysAndValues: some Sequence<(Key, Value)>
  ) {
    self.init()
    reserveCapacity(keysAndValues.underestimatedCount)
    for (key, value) in keysAndValues {
      guard _keys._append(key).inserted else {
        preconditionFailure("Duplicate key: '\(key)'")
      }
      _values.append(value)
    }
  }
}

extension OrderedDictionary {
  /// Creates a new dictionary from separate sequences of keys and values.
  ///
  /// You use this initializer to create a dictionary when you have two
  /// sequences with unique keys and their associated values, respectively.
  /// Passing a `keys` sequence with duplicate keys to this initializer results
  /// in a runtime error.
  ///
  /// - Parameter keys: A sequence of unique keys.
  ///
  /// - Parameter values: A sequence of values associated with items in `keys`.
  ///
  /// - Returns: A new dictionary initialized with the data in
  ///   `keys` and `values`.
  ///
  /// - Precondition: The sequence must not have duplicate keys, and `keys` and
  ///    `values` must contain an equal number of elements.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count if
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @inlinable
  public init(
    uniqueKeys keys: some Sequence<Key>,
    values: some Sequence<Value>
  ) {
    let keys = ContiguousArray(keys)
    let values = ContiguousArray(values)
    precondition(keys.count == values.count,
                 "Mismatching element counts between keys and values")
    self._keys = .init(keys)
    self._values = values
    precondition(_keys.count == _values.count, "Duplicate keys")
    _checkInvariants()
  }
}

extension OrderedDictionary {
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
  ///     let firstValues = OrderedDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (first, _) in first })
  ///     // ["a": 1, "b": 2]
  ///
  ///     let lastValues = OrderedDictionary(
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  @inline(__always)
  public init(
    _ keysAndValues: some Sequence<(key: Key, value: Value)>,
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
  ///     let firstValues = OrderedDictionary(
  ///       pairsWithDuplicateKeys,
  ///       uniquingKeysWith: { (first, _) in first })
  ///     // ["a": 1, "b": 2]
  ///
  ///     let lastValues = OrderedDictionary(
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public init(
    _ keysAndValues: some Sequence<(Key, Value)>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    self.init()
    try self.merge(keysAndValues, uniquingKeysWith: combine)
  }
}

extension OrderedDictionary {
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
  ///     let studentsByLetter = OrderedDictionary(grouping: students, by: { $0.first! })
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
  ///    values, if `Key` implements high-quality hashing.
  @inlinable
  public init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows where Value: RangeReplaceableCollection, Value.Element == S.Element {
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
  ///     let studentsByLetter = OrderedDictionary(grouping: students, by: { $0.first! })
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
  ///    values, if `Key` implements high-quality hashing.
  @inlinable
  public init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows where Value == [S.Element] {
    // Note: this extra overload is necessary to make type inference work
    // for the `Value` type -- we want it to default to `[S.Element`].
    // (https://github.com/apple/swift-collections/issues/139)
    try self.init(_grouping: values, by: keyForValue)
  }

  @inlinable
  internal init<S: Sequence>(
    _grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows where Value: RangeReplaceableCollection, Value.Element == S.Element {
    self.init()
    for value in values {
      let key = try keyForValue(value)
      self.updateValue(forKey: key, default: Value()) { array in
        array.append(value)
      }
    }
  }
}

extension OrderedDictionary {
  @inlinable
  internal init(
    _uncheckedUniqueKeysWithValues keysAndValues: 
    some Sequence<(key: Key, value: Value)>
  ) {
    self.init()
    reserveCapacity(keysAndValues.underestimatedCount)
    for (key, value) in keysAndValues {
      _keys._appendNew(key)
      _values.append(value)
    }
    _checkInvariants()
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence,
  /// which must not contain duplicate keys.
  ///
  /// In optimized builds, this initializer does not verify that the keys are
  /// actually unique. This makes creating the dictionary somewhat faster if you
  /// know for sure that the elements are unique (e.g., because they come from
  /// another collection with guaranteed-unique members, such as a
  /// `Dictionary`). However, if you accidentally call this initializer with
  /// duplicate members, it can return a corrupt dictionary value that may be
  /// difficult to debug.
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  public init(
    uncheckedUniqueKeysWithValues keysAndValues:
    some Sequence<(key: Key, value: Value)>
  ) {
#if DEBUG
    self.init(uniqueKeysWithValues: keysAndValues)
#else
    self.init(_uncheckedUniqueKeysWithValues: keysAndValues)
#endif
  }

  /// Creates a new dictionary from the key-value pairs in the given sequence,
  /// which must not contain duplicate keys.
  ///
  /// In optimized builds, this initializer does not verify that the keys are
  /// actually unique. This makes creating the dictionary somewhat faster if you
  /// know for sure that the elements are unique (e.g., because they come from
  /// another collection with guaranteed-unique members, such as a
  /// `Dictionary`). However, if you accidentally call this initializer with
  /// duplicate members, it can return a corrupt dictionary value that may be
  /// difficult to debug.
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
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @inlinable
  public init(
    uncheckedUniqueKeysWithValues keysAndValues: some Sequence<(Key, Value)>
  ) {
    // Add tuple labels
    let keysAndValues = keysAndValues.lazy.map { (key: $0.0, value: $0.1) }
    self.init(uncheckedUniqueKeysWithValues: keysAndValues)
  }
}

extension OrderedDictionary {
  /// Creates a new dictionary from separate sequences of unique keys and
  /// associated values.
  ///
  /// In optimized builds, this initializer does not verify that the keys are
  /// actually unique. This makes creating the dictionary somewhat faster if you
  /// know for sure that the elements are unique (e.g., because they come from
  /// another collection with guaranteed-unique members, such as a
  /// `Dictionary`). However, if you accidentally call this initializer with
  /// duplicate members, it can return a corrupt dictionary value that may be
  /// difficult to debug.
  ///
  /// - Parameter keys: A sequence of unique keys.
  ///
  /// - Parameter values: A sequence of values associated with items in `keys`.
  ///
  /// - Returns: A new dictionary initialized with the data in
  ///   `keys` and `values`.
  ///
  /// - Precondition: The sequence must not have duplicate keys, and `keys` and
  ///    `values` must contain an equal number of elements.
  ///
  /// - Complexity: Expected O(*n*) on average, where *n* is the count if
  ///    key-value pairs, if `Key` implements high-quality hashing.
  @inlinable
  @inline(__always)
  public init(
    uncheckedUniqueKeys keys: some Sequence<Key>,
    values: some Sequence<Value>
  ) {
#if DEBUG
    self.init(uniqueKeys: keys, values: values)
#else
    self._keys = .init(uncheckedUniqueElements: keys)
    self._values = .init(values)
    precondition(_keys.count == _values.count)
    _checkInvariants()
#endif
  }
}
