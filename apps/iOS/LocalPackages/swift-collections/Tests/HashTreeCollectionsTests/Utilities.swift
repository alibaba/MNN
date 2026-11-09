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

#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import _CollectionsTestSupport
import HashTreeCollections
#endif

extension LifetimeTracker {
  func shareableDictionary<Keys: Sequence>(
    keys: Keys
  ) -> (
    dictionary: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>>,
    keys: [LifetimeTracked<Int>],
    values: [LifetimeTracked<Int>]
  )
  where Keys.Element == Int
  {
    let k = Array(keys)
    let keys = self.instances(for: k)
    let values = self.instances(for: k.map { $0 + 100 })
    let dictionary = TreeDictionary(
      uniqueKeysWithValues: zip(keys, values))
    return (dictionary, keys, values)
  }
}

protocol DataGenerator {
  associatedtype Key: Hashable
  associatedtype Value: Equatable

  func key(for i: Int) -> Key
  func value(for i: Int) -> Value
}

struct IntDataGenerator: DataGenerator {
  typealias Key = Int
  typealias Value = Int

  let valueOffset: Int

  init(valueOffset: Int) {
    self.valueOffset = valueOffset
  }

  func key(for i: Int) -> Key {
    i
  }

  func value(for i: Int) -> Value {
    i + valueOffset
  }
}

struct ColliderDataGenerator: DataGenerator {
  typealias Key = Collider
  typealias Value = Int

  let groups: Int
  let valueOffset: Int

  init(groups: Int, valueOffset: Int) {
    self.groups = groups
    self.valueOffset = valueOffset
  }

  func key(for i: Int) -> Key {
    Collider(i, Hash(i % groups))
  }

  func value(for i: Int) -> Value {
    i + valueOffset
  }
}

extension LifetimeTracker {
  func shareableDictionary<Payloads: Sequence, G: DataGenerator>(
    _ payloads: Payloads,
    with generator: G
  ) -> (
    map: TreeDictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>>,
    expected: [LifetimeTracked<G.Key>: LifetimeTracked<G.Value>]
  )
  where Payloads.Element == Int
  {
    typealias Key = LifetimeTracked<G.Key>
    typealias Value = LifetimeTracked<G.Value>
    func gen() -> [(Key, Value)] {
      payloads.map {
        (instance(for: generator.key(for: $0)),
         instance(for: generator.value(for: $0)))
      }
    }
    let map = TreeDictionary(uniqueKeysWithValues: gen())
    let expected = Dictionary(uniqueKeysWithValues: gen())
    return (map, expected)
  }

  func shareableDictionary<Keys: Sequence>(
    keys: Keys
  ) -> (
    map: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>>,
    expected: [LifetimeTracked<Int>: LifetimeTracked<Int>]
  )
  where Keys.Element == Int
  {
    shareableDictionary(keys, with: IntDataGenerator(valueOffset: 100))
  }
}

func expectEqualSets<Element: Hashable>(
  _ set: TreeSet<Element>,
  _ ref: [Element],
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  expectEqualSets(
    set, Set(ref),
    message(),
    trapping: trapping,
    file: file, line: line)
}

func expectEqualSets<C: Collection>(
  _ set: C,
  _ ref: Set<C.Element>,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  var ref = ref
  var seen: Set<C.Element> = []
  var extras: [C.Element] = []
  var dupes: [C.Element: Int] = [:]
  for item in set {
    if !seen.insert(item).inserted {
      dupes[item, default: 1] += 1
    } else if ref.remove(item) == nil {
      extras.append(item)
    }
  }
  let missing = Array(ref)
  var msg = ""
  if !extras.isEmpty {
    msg += "\nUnexpected items: \(extras)"
  }
  if !missing.isEmpty {
    msg += "\nMissing items: \(missing)"
  }
  if !dupes.isEmpty {
    msg += "\nDuplicate items: \(dupes)"
  }
  if !msg.isEmpty {
    _expectFailure(
      "\n\(msg)",
      message, trapping: trapping, file: file, line: line)
  }
}


func expectEqualDictionaries<Key: Hashable, Value: Equatable>(
  _ map: TreeDictionary<Key, Value>,
  _ ref: [(key: Key, value: Value)],
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  expectEqualDictionaries(
    map, Dictionary(uniqueKeysWithValues: ref),
    message(),
    trapping: trapping,
    file: file, line: line)
}

func expectEqualDictionaries<Key: Hashable, Value: Equatable>(
  _ map: TreeDictionary<Key, Value>,
  _ dict: Dictionary<Key, Value>,
  _ message: @autoclosure () -> String = "",
  trapping: Bool = false,
  file: StaticString = #file,
  line: UInt = #line
) {
  expectEqual(map.count, dict.count, "Mismatching count", file: file, line: line)
  var dict = dict
  var seen: Set<Key> = []
  var mismatches: [(key: Key, map: Value?, dict: Value?)] = []
  var dupes: [(key: Key, map: Value)] = []
  for (key, value) in map {
    if !seen.insert(key).inserted {
      dupes.append((key, value))
    } else {
      let expected = dict.removeValue(forKey: key)
      if value != expected {
        mismatches.append((key, value, expected))
      }
    }
  }
  for (key, value) in dict {
    mismatches.append((key, nil, value))
  }
  if !mismatches.isEmpty || !dupes.isEmpty {
    let msg1 = mismatches.lazy.map { k, m, d in
      "\n  \(k): \(m == nil ? "nil" : "\(m!)") vs \(d == nil ? "nil" : "\(d!)")"
    }.joined(separator: "")
    let msg2 = dupes.lazy.map { k, v in
      "\n  \(k): \(v) (duped)"
    }.joined(separator: "")
    _expectFailure(
      "\n\(mismatches.count) mismatches (actual vs expected):\(msg1)\(msg2)",
      message, trapping: trapping, file: file, line: line)
  }
}

func mutate<T, R>(
  _ value: inout T,
  _ body: (inout T) throws -> R
) rethrows -> R {
  try body(&value)
}
