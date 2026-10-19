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

/// A test protocol for validating that dictionary-like types implement users'
/// baseline expectations.
///
/// To ensure maximum utility, this protocol doesn't refine `Collection`,
/// although it does share some of the same requirements.
public protocol DictionaryAPIChecker {
  associatedtype Key
  associatedtype Value
  associatedtype Index

  typealias Element = (key: Key, value: Value)

  associatedtype Keys: Collection
  where Keys.Element == Key, Keys.Index == Index

  var keys: Keys { get }

  // `Values` ought to be a `MutableCollection` when possible, with `values`
  // providing a setter. Unfortunately, tree-based dictionaries need to
  // invalidate indices when mutating keys.
  associatedtype Values: Collection
  where Values.Element == Value, Values.Index == Index

  var values: Values { get }

  var isEmpty: Bool { get }
  var count: Int { get }

  func index(forKey key: Key) -> Index?

  subscript(key: Key) -> Value? { get set }
  subscript(
    key: Key,
    default defaultValue: @autoclosure () -> Value
  ) -> Value { get set }

  mutating func updateValue(_ value: Value, forKey key: Key) -> Value?
  mutating func removeValue(forKey key: Key) -> Value?
  mutating func remove(at index: Index) -> Element

  init()

  init<S: Sequence>(
    _ keysAndValues: S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S.Element == (Key, Value)

  mutating func merge<S: Sequence>(
    _ keysAndValues: __owned S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S.Element == (Key, Value)

  __consuming func merging<S: Sequence>(
    _ other: __owned S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self where S.Element == (Key, Value)

  func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Self

#if false
  // We can't express these as protocol requirements:
  func mapValues<T>(
    _ transform: (Value) throws -> T
  ) rethrows -> Self<Key, T>

  func compactMapValues<T>(
    _ transform: (Value) throws -> T?
  ) rethrows -> Self<Key, T>

  init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows where Value: RangeReplaceableCollection, Value.Element == S.Element

  public init<S: Sequence>(
    grouping values: S,
    by keyForValue: (S.Element) throws -> Key
  ) rethrows where Value == [S.Element]
#endif
}

extension Dictionary: DictionaryAPIChecker {}

/// Additional entry points provided by this package that aren't provided
/// by `Dictionary` (yet?).
public protocol DictionaryAPIExtras: DictionaryAPIChecker {
  // Extras (not in the Standard Library)

  init<S: Sequence>(
    uniqueKeysWithValues keysAndValues: S
  ) where S.Element == (Key, Value)

  init<S: Sequence>(
    uniqueKeysWithValues keysAndValues: S
  ) where S.Element == Element

//  init<Keys: Sequence, Values: Sequence>(
//    uniqueKeys keys: Keys,
//    values: Values
//  ) where Keys.Element == Key, Values.Element == Value

  init<S: Sequence>(
    _ keysAndValues: S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S.Element == Element

  mutating func merge<S: Sequence>(
    _ keysAndValues: __owned S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows where S.Element == Element

  __consuming func merging<S: Sequence>(
    _ other: __owned S,
    uniquingKeysWith combine: (Value, Value) throws -> Value
  ) rethrows -> Self where S.Element == Element

  mutating func updateValue<R>(
    forKey key: Key,
    default defaultValue: @autoclosure () -> Value,
    with body: (inout Value) throws -> R
  ) rethrows -> R

#if false
  // Potential additions implemented by TreeDictionary:

  func contains(_ key: Key) -> Bool

  mutating func updateValue<R>(
    forKey key: Key,
    with body: (inout Value?) throws -> R
  ) rethrows -> R

#endif
}
