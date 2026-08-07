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
  /// Merges the key-value pairs in the given sequence into the dictionary,
  /// using a combining closure to determine the value for any duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the updated
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     var dictionary: TreeDictionary = ["a": 1, "b": 2]
  ///
  ///     // Keeping existing value for key "a":
  ///     dictionary.merge(zip(["a", "c"], [3, 4])) { (current, _) in current }
  ///     // ["a": 1, "b": 2, "c": 4] (in some order)
  ///
  ///     // Taking the new value for key "a":
  ///     dictionary.merge(zip(["a", "d"], [5, 6])) { (_, new) in new }
  ///     // ["a": 5, "b": 2, "c": 4, "d": 6] (in some order)
  ///
  /// - Parameters:
  ///   - keysAndValues: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  @inlinable
  public mutating func merge(
    _ keysAndValues: Self,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    _invalidateIndices()
    _ = try _root.merge(.top, keysAndValues._root, combine)
    _invariantCheck()
  }

  /// Merges the key-value pairs in the given sequence into the dictionary,
  /// using a combining closure to determine the value for any duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the updated
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     var dictionary: TreeDictionary = ["a": 1, "b": 2]
  ///
  ///     // Keeping existing value for key "a":
  ///     dictionary.merge(zip(["a", "c"], [3, 4])) { (current, _) in current }
  ///     // ["a": 1, "b": 2, "c": 4] (in some order)
  ///
  ///     // Taking the new value for key "a":
  ///     dictionary.merge(zip(["a", "d"], [5, 6])) { (_, new) in new }
  ///     // ["a": 5, "b": 2, "c": 4, "d": 6] (in some order)
  ///
  /// - Parameters:
  ///   - keysAndValues: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  @inlinable
  public mutating func merge(
    _ keysAndValues: __owned some Sequence<(Key, Value)>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    for (key, value) in keysAndValues {
      try self.updateValue(forKey: key) { target in
        if let old = target {
          target = try combine(old, value)
        } else {
          target = value
        }
      }
    }
    _invariantCheck()
  }

  /// Merges the key-value pairs in the given sequence into the dictionary,
  /// using a combining closure to determine the value for any duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the updated
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     var dictionary: TreeDictionary = ["a": 1, "b": 2]
  ///
  ///     // Keeping existing value for key "a":
  ///     dictionary.merge(zip(["a", "c"], [3, 4])) { (current, _) in current }
  ///     // ["a": 1, "b": 2, "c": 4] (in some order)
  ///
  ///     // Taking the new value for key "a":
  ///     dictionary.merge(zip(["a", "d"], [5, 6])) { (_, new) in new }
  ///     // ["a": 5, "b": 2, "c": 4, "d": 6] (in some order)
  ///
  /// - Parameters:
  ///   - keysAndValues: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  public mutating func merge(
    _ keysAndValues: __owned some Sequence<Element>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows {
    try merge(
      keysAndValues.lazy.map { ($0.key, $0.value) },
      uniquingKeysWith: combine)
  }

  /// Creates a dictionary by merging key-value pairs in a sequence into this
  /// dictionary, using a combining closure to determine the value for
  /// duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the returned
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     let dictionary: OrderedDictionary = ["a": 1, "b": 2]
  ///     let newKeyValues = zip(["a", "b"], [3, 4])
  ///
  ///     let keepingCurrent = dictionary.merging(newKeyValues) { (current, _) in current }
  ///     // ["a": 1, "b": 2]
  ///     let replacingCurrent = dictionary.merging(newKeyValues) { (_, new) in new }
  ///     // ["a": 3, "b": 4]
  ///
  /// - Parameters:
  ///   - other: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  ///
  /// - Returns: A new dictionary with the combined keys and values of this
  ///    dictionary and `other`. The order of keys in the result dictionary
  ///    matches that of `self`, with additional key-value pairs (if any)
  ///    appended at the end in the order they appear in `other`.
  ///
  /// - Complexity: Expected to be O(`count` + *n*) on average, where *n* is the
  ///    number of elements in `keysAndValues`, if `Key` implements high-quality
  ///    hashing.
  @inlinable
  public func merging(
    _ other: Self,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self {
    var copy = self
    try copy.merge(other, uniquingKeysWith: combine)
    return copy
  }

  /// Creates a dictionary by merging key-value pairs in a sequence into this
  /// dictionary, using a combining closure to determine the value for
  /// duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the returned
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     let dictionary: OrderedDictionary = ["a": 1, "b": 2]
  ///     let newKeyValues = zip(["a", "b"], [3, 4])
  ///
  ///     let keepingCurrent = dictionary.merging(newKeyValues) { (current, _) in current }
  ///     // ["a": 1, "b": 2]
  ///     let replacingCurrent = dictionary.merging(newKeyValues) { (_, new) in new }
  ///     // ["a": 3, "b": 4]
  ///
  /// - Parameters:
  ///   - other: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  ///
  /// - Returns: A new dictionary with the combined keys and values of this
  ///    dictionary and `other`. The order of keys in the result dictionary
  ///    matches that of `self`, with additional key-value pairs (if any)
  ///    appended at the end in the order they appear in `other`.
  ///
  /// - Complexity: Expected to be O(`count` + *n*) on average, where *n* is the
  ///    number of elements in `keysAndValues`, if `Key` implements high-quality
  ///    hashing.
  @inlinable
  public func merging(
    _ other: __owned some Sequence<(Key, Value)>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self {
    var copy = self
    try copy.merge(other, uniquingKeysWith: combine)
    return copy
  }

  /// Creates a dictionary by merging key-value pairs in a sequence into this
  /// dictionary, using a combining closure to determine the value for
  /// duplicate keys.
  ///
  /// Use the `combine` closure to select a value to use in the returned
  /// dictionary, or to combine existing and new values. As the key-value
  /// pairs are merged with the dictionary, the `combine` closure is called
  /// with the current and new values for any duplicate keys that are
  /// encountered.
  ///
  /// This example shows how to choose the current or new values for any
  /// duplicate keys:
  ///
  ///     let dictionary: OrderedDictionary = ["a": 1, "b": 2]
  ///     let newKeyValues = zip(["a", "b"], [3, 4])
  ///
  ///     let keepingCurrent = dictionary.merging(newKeyValues) { (current, _) in current }
  ///     // ["a": 1, "b": 2]
  ///     let replacingCurrent = dictionary.merging(newKeyValues) { (_, new) in new }
  ///     // ["a": 3, "b": 4]
  ///
  /// - Parameters:
  ///   - other: A sequence of key-value pairs.
  ///   - combine: A closure that takes the current and new values for any
  ///     duplicate keys. The closure returns the desired value for the final
  ///     dictionary.
  ///
  /// - Returns: A new dictionary with the combined keys and values of this
  ///    dictionary and `other`. The order of keys in the result dictionary
  ///    matches that of `self`, with additional key-value pairs (if any)
  ///    appended at the end in the order they appear in `other`.
  ///
  /// - Complexity: Expected to be O(`count` + *n*) on average, where *n* is the
  ///    number of elements in `keysAndValues`, if `Key` implements high-quality
  ///    hashing.
  @_disfavoredOverload // https://github.com/apple/swift-collections/issues/125
  @inlinable
  public func merging(
    _ other: __owned some Sequence<Element>,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self {
    var copy = self
    try copy.merge(other, uniquingKeysWith: combine)
    return copy
  }
}
