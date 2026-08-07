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

extension TreeDictionary: DictionaryAPIExtras {}

class TreeDictionaryTests: CollectionTestCase {
  func test_empty() {
    let d = TreeDictionary<String, Int>()
    expectEqualElements(d, [])
    expectEqual(d.count, 0)

    var it = d.makeIterator()
    expectNil(it.next())
    expectNil(it.next())
    expectNil(it.next())

    expectEqual(d.startIndex, d.endIndex)

    expectEqual(d.distance(from: d.startIndex, to: d.endIndex), 0)
  }

  func test_remove_update_basics() throws {
    var d = TreeDictionary<String, Int>()

    d.updateValue(1, forKey: "One")
    d.updateValue(2, forKey: "Two")
    d.updateValue(3, forKey: "Three")

    expectEqual(d.count, 3)
    expectEqual(d["One"], 1)
    expectEqual(d["Two"], 2)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    expectEqual(d.removeValue(forKey: "Two"), 2)

    expectEqual(d.count, 2)
    expectEqual(d["One"], 1)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    expectEqual(d.removeValue(forKey: "Two"), nil)
    expectEqual(d.removeValue(forKey: "One"), 1)

    expectEqual(d.count, 1)
    expectEqual(d["One"], nil)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    expectEqual(d.removeValue(forKey: "One"), nil)
    expectEqual(d.removeValue(forKey: "Two"), nil)
    expectEqual(d.removeValue(forKey: "Three"), 3)

    expectEqual(d.count, 0)
    expectEqual(d["One"], nil)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], nil)
    expectEqual(d["Four"], nil)
  }

  func test_subscript_setter_basics() throws {
    var d = TreeDictionary<String, Int>()

    d["One"] = 1
    d["Two"] = 2
    d["Three"] = 3

    expectEqual(d.count, 3)
    expectEqual(d["One"], 1)
    expectEqual(d["Two"], 2)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    d["Two"] = nil

    expectEqual(d.count, 2)
    expectEqual(d["One"], 1)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    d["Two"] = nil
    d["One"] = nil

    expectEqual(d.count, 1)
    expectEqual(d["One"], nil)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], 3)
    expectEqual(d["Four"], nil)

    d["One"] = nil
    d["Two"] = nil
    d["Three"] = nil

    expectEqual(d.count, 0)
    expectEqual(d["One"], nil)
    expectEqual(d["Two"], nil)
    expectEqual(d["Three"], nil)
    expectEqual(d["Four"], nil)
  }

  func test_add_remove() throws {
    var d = TreeDictionary<String, Int>()

    let c = 400
    for i in 0 ..< c {
      expectNil(d.updateValue(i, forKey: "\(i)"))
      expectEqual(d.count, i + 1)
    }

    for i in 0 ..< c {
      expectEqual(d["\(i)"], i)
    }

    for i in 0 ..< c {
      expectEqual(d.updateValue(2 * i, forKey: "\(i)"), i)
      expectEqual(d.count, c)
    }

    for i in 0 ..< c {
      expectEqual(d["\(i)"], 2 * i)
    }

    var remaining = c
    for i in 0 ..< c {
      expectEqual(d.removeValue(forKey: "\(i)"), 2 * i)
      remaining -= 1
      expectEqual(d.count, remaining)
    }
  }

  func test_collisions() throws {
    var d = TreeDictionary<Collider, Int>()

    let count = 100
    let groups = 20

    for i in 0 ..< count {
      let h = i % groups
      let key = Collider(i, Hash(h))
      expectEqual(d[key], nil)
      expectNil(d.updateValue(i, forKey: key))
      expectEqual(d[key], i)
    }

    for i in 0 ..< count {
      let h = i % groups
      let key = Collider(i, Hash(h))
      expectEqual(d[key], i)
      expectEqual(d.updateValue(2 * i, forKey: key), i)
      expectEqual(d[key], 2 * i)
    }

    for i in 0 ..< count {
      let h = i % groups
      let key = Collider(i, Hash(h))
      expectEqual(d[key], 2 * i)
      expectEqual(d.removeValue(forKey: key), 2 * i)
      expectEqual(d[key], nil)
    }
  }

  func test_shared_copies() throws {
    var d = TreeDictionary<Int, Int>()

    let c = 200
    for i in 0 ..< c {
      expectNil(d.updateValue(i, forKey: i))
    }

    let copy = d
    for i in 0 ..< c {
      expectEqual(d.updateValue(2 * i, forKey: i), i)
    }

    for i in 0 ..< c {
      expectEqual(copy[i], i)
    }

    let copy2 = d
    for i in 0 ..< c {
      expectEqual(d.removeValue(forKey: i), 2 * i)
    }

    for i in 0 ..< c {
      expectEqual(copy2[i], 2 * i)
    }
  }

  func test_Sequence_basic() {
    var d: TreeDictionary<Int, Int> = [1: 2]
    var it = d.makeIterator()
    expectEquivalent(it.next(), (1, 2), by: { $0 == $1 })
    expectNil(it.next())
    expectNil(it.next())

    d[1] = nil
    it = d.makeIterator()
    expectNil(it.next())
    expectNil(it.next())
  }

  func test_Sequence_400() {
    var d = TreeDictionary<Int, Int>()
    let c = 400
    for i in 0 ..< c {
      expectNil(d.updateValue(i, forKey: i))
    }

    var seen: Set<Int> = []
    for (key, value) in d {
      expectEqual(key, value)
      expectTrue(seen.insert(key).inserted, "Duplicate key seen: \(key)")
    }
    expectEqual(seen.count, c)
    expectTrue(seen.isSuperset(of: 0 ..< c))
  }

  func test_Sequence_collisions() {
    var d = TreeDictionary<Collider, Int>()

    let count = 100
    let groups = 20

    for i in 0 ..< count {
      let h = i % groups
      let key = Collider(i, Hash(h))
      expectNil(d.updateValue(i, forKey: key))
    }

    var seen: Set<Int> = []
    for (key, value) in d {
      expectEqual(key.identity, value)
      expectTrue(seen.insert(key.identity).inserted, "Duplicate key: \(key)")
    }
    expectEqual(seen.count, count)
    expectTrue(seen.isSuperset(of: 0 ..< count))
  }

  func test_BidirectionalCollection_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (d, ref) = tracker.shareableDictionary(for: fixture)
        checkCollection(d, expectedContents: ref, by: ==)
        _checkBidirectionalCollection_indexOffsetBy(
          d, expectedContents: ref, by: ==)
      }
    }
  }

  func test_BidirectionalCollection_random100() {
    let d = TreeDictionary<Int, Int>(
      uniqueKeysWithValues: (0 ..< 100).map { ($0, $0) })
    let ref = Array(d)
    checkCollection(d, expectedContents: ref, by: ==)
    _checkBidirectionalCollection_indexOffsetBy(
      d, expectedContents: ref, by: ==)
  }

  @available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *)
  struct FancyDictionaryKey: CodingKeyRepresentable, Hashable, Codable {
    var value: Int

    var codingKey: CodingKey { value.codingKey }

    init(_ value: Int) {
      self.value = value
    }

    init?<T: CodingKey>(codingKey: T) {
      guard let value = codingKey.intValue else { return nil}
      self.init(value)
    }
  }

  struct BoringDictionaryKey: Hashable, Codable {
    var value: Int

    init(_ value: Int) {
      self.value = value
    }
  }

  func test_Encodable() throws {
    let d1: TreeDictionary<String, Int> = [
      "one": 1, "two": 2, "three": 3]
    let v1: MinimalEncoder.Value = .dictionary([
      "one": .int(1), "two": .int(2), "three": .int(3)])
    expectEqual(try MinimalEncoder.encode(d1), v1)

    let d2: TreeDictionary<Int, String> = [
      1: "one", 2: "two", 3: "three"]
    let v2: MinimalEncoder.Value = .dictionary([
      "1": .string("one"),
      "2": .string("two"),
      "3": .string("three"),
    ])
    expectEqual(try MinimalEncoder.encode(d2), v2)

    if #available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *) {
      let d3: TreeDictionary<FancyDictionaryKey, Int16> = [
        FancyDictionaryKey(1): 10, FancyDictionaryKey(2): 20
      ]
      let v3: MinimalEncoder.Value = .dictionary([
        "1": .int16(10), "2": .int16(20)
      ])
      expectEqual(try MinimalEncoder.encode(d3), v3)
    }

    let d4: TreeDictionary<BoringDictionaryKey, UInt8> = [
      // Note: we only have a single element to prevent ordering
      // problems.
      BoringDictionaryKey(42): 23
    ]
    let v4: MinimalEncoder.Value = .array([
      .dictionary(["value": .int(42)]),
      .uint8(23),
    ])
    expectEqual(try MinimalEncoder.encode(d4), v4)
  }

  func test_Decodable() throws {
    typealias PD<Key: Hashable, Value> = TreeDictionary<Key, Value>

    let d1: TreeDictionary<String, Int> = [
      "one": 1, "two": 2, "three": 3]
    let v1: MinimalEncoder.Value = .dictionary([
      "one": .int(1), "two": .int(2), "three": .int(3)])
    expectEqual(
      try MinimalDecoder.decode(v1, as: PD<String, Int>.self),
      d1)

    let d2: TreeDictionary<Int, String> = [
      1: "one", 2: "two", 3: "three"]
    let v2: MinimalEncoder.Value = .dictionary([
      "1": .string("one"),
      "2": .string("two"),
      "3": .string("three"),
    ])
    expectEqual(
      try MinimalDecoder.decode(v2, as: PD<Int, String>.self),
      d2)

    if #available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *) {
      let d3: TreeDictionary<FancyDictionaryKey, Int16> = [
        FancyDictionaryKey(1): 10, FancyDictionaryKey(2): 20
      ]
      let v3: MinimalEncoder.Value = .dictionary([
        "1": .int16(10), "2": .int16(20)
      ])
      expectEqual(
        try MinimalDecoder.decode(v3, as: PD<FancyDictionaryKey, Int16>.self),
        d3)
    }

    let d4: TreeDictionary<BoringDictionaryKey, UInt8> = [
      // Note: we only have a single element to prevent ordering
      // problems.
      BoringDictionaryKey(42): 23
    ]
    let v4: MinimalEncoder.Value = .array([
      .dictionary(["value": .int(42)]),
      .uint8(23),
    ])
    expectEqual(
      try MinimalDecoder.decode(v4, as: PD<BoringDictionaryKey, UInt8>.self),
      d4)

    let v5: MinimalEncoder.Value = .dictionary([
      "This is not a number": .string("bus"),
      "42": .string("train")])
    expectThrows(try MinimalDecoder.decode(v5, as: PD<Int, String>.self)) {
      expectTrue($0 is DecodingError)
    }

    if #available(macOS 12.3, iOS 15.4, watchOS 8.5, tvOS 15.4, *) {
      let v6: MinimalEncoder.Value = .dictionary([
        "This is not a number": .string("bus"),
        "42": .string("train")])
      expectThrows(
        try MinimalDecoder.decode(
          v6, as: PD<FancyDictionaryKey, String>.self)
      ) {
        expectTrue($0 is DecodingError)
      }

      let v7: MinimalEncoder.Value = .dictionary([
        "23": .string("bus"),
        "42": .string("train")])
      let d7: PD<FancyDictionaryKey, String> = [
        FancyDictionaryKey(23): "bus",
        FancyDictionaryKey(42): "train",
      ]
      expectEqual(
        try MinimalDecoder.decode(v7, as: PD<FancyDictionaryKey, String>.self),
        d7)
    }

    let v8: MinimalEncoder.Value = .array([
      .int32(1), .string("bike"), .int32(2),
    ])
    expectThrows(try MinimalDecoder.decode(v8, as: PD<Int32, String>.self))
  }

  func test_CustomReflectable() {
    do {
      let d: TreeDictionary<Int, Int> = [1: 2, 3: 4, 5: 6]
      let mirror = Mirror(reflecting: d)
      expectEqual(mirror.displayStyle, .dictionary)
      expectNil(mirror.superclassMirror)
      expectTrue(mirror.children.compactMap { $0.label }.isEmpty) // No label
      expectEqualElements(
        mirror.children.compactMap { $0.value as? (key: Int, value: Int) },
        d.map { $0 })
    }
  }

  func test_Equatable_Hashable() {
    let samples: [[TreeDictionary<Int, Int>]] = [
      [[:], [:]],
      [[1: 100], [1: 100]],
      [[2: 200], [2: 200]],
      [[3: 300], [3: 300]],
      [[100: 1], [100: 1]],
      [[1: 1], [1: 1]],
      [[100: 100], [100: 100]],
      [[1: 100, 2: 200], [2: 200, 1: 100]],
      [[1: 100, 2: 200, 3: 300],
       [1: 100, 3: 300, 2: 200],
       [2: 200, 1: 100, 3: 300],
       [2: 200, 3: 300, 1: 100],
       [3: 300, 1: 100, 2: 200],
       [3: 300, 2: 200, 1: 100]]
    ]
    checkHashable(equivalenceClasses: samples)
  }

  func test_CustomStringConvertible() {
    let a: TreeDictionary<RawCollider, Int> = [:]
    expectEqual(a.description, "[:]")

    let b: TreeDictionary<RawCollider, Int> = [
      RawCollider(0): 1
    ]
    expectEqual(b.description, "[0#0: 1]")

    let c: TreeDictionary<RawCollider, Int> = [
      RawCollider(0): 1,
      RawCollider(2): 3,
      RawCollider(4): 5,
    ]
    expectEqual(c.description, "[0#0: 1, 2#2: 3, 4#4: 5]")
  }

  func test_CustomDebugStringConvertible() {
    let a: TreeDictionary<Int, Int> = [:]
    expectEqual(a.debugDescription, "[:]")

    let b: TreeDictionary<Int, Int> = [0: 1]
    expectEqual(b.debugDescription, "[0: 1]")

    let c: TreeDictionary<Int, Int> = [0: 1, 2: 3, 4: 5]
    let cd = c.map { "\($0.key): \($0.value)"}.joined(separator: ", ")
    expectEqual(c.debugDescription, "[\(cd)]")
  }


  func test_index_descriptions() {
    let a: TreeDictionary = [
      RawCollider(1, "1"): 1,
      RawCollider(2, "21"): 2,
      RawCollider(3, "22"): 3,
    ]

    let i = a.startIndex
    expectEqual(i.description, "@[0]")
    expectEqual(i.debugDescription, "@[0]")

    let j = a.index(i, offsetBy: 1)
    expectEqual(j.description, "@.0[0]")
    expectEqual(j.debugDescription, "@.0[0]")

    let k = a.index(j, offsetBy: 1)
    expectEqual(k.description, "@.0[1]")
    expectEqual(k.debugDescription, "@.0[1]")

    let end = a.endIndex
    expectEqual(end.description, "@.end(1)")
    expectEqual(end.debugDescription, "@.end(1)")
  }


  func test_updateValueForKey_fixtures() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var d: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
          var ref: Dictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
          withEvery("i", in: 0 ..< fixture.count) { i in
            withHiddenCopies(if: isShared, of: &d) { d in
              let item = fixture.itemsInInsertionOrder[i]
              let key1 = tracker.instance(for: item)
              let key2 = tracker.instance(for: item)
              let value = tracker.instance(for: 1000 + item.identity)
              d[key1] = value
              ref[key2] = value
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_updateValueForKey_fixtures_tiny() {
    struct Empty: Hashable {}

    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var d: TreeDictionary<LifetimeTracked<RawCollider>, Empty> = [:]
          var ref: Dictionary<LifetimeTracked<RawCollider>, Empty> = [:]
          withEvery("i", in: 0 ..< fixture.count) { i in
            withHiddenCopies(if: isShared, of: &d) { d in
              let item = fixture.itemsInInsertionOrder[i]
              let key = tracker.instance(for: item)
              d[key] = Empty()
              ref[key] = Empty()
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_removeValueForKey_fixtures_justOne() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            withHiddenCopies(if: isShared, of: &d) { d in
              let old = d.removeValue(forKey: ref[offset].key)
              d._invariantCheck()
              expectEqual(old, ref[offset].value)
              ref.remove(at: offset)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_removeValueForKey_fixtures_all() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          var rng = RepeatableRandomNumberGenerator(seed: 0)
          withEvery("i", in: 0 ..< fixture.count) { _ in
            withHiddenCopies(if: isShared, of: &d) { d in
              let offset = ref.indices.randomElement(using: &rng)!
              let old = d.removeValue(forKey: ref[offset].key)
              d._invariantCheck()
              expectEqual(old, ref[offset].value)
              ref.remove(at: offset)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_getter_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withLifetimeTracking { tracker in
          let (d, ref) = tracker.shareableDictionary(0 ..< count, with: generator)
          withEvery("key", in: 0 ..< count) { key in
            let key = tracker.instance(for: generator.key(for: key))
            expectEqual(d[key], ref[key])
          }
          expectNil(d[tracker.instance(for: generator.key(for: -1))])
          expectNil(d[tracker.instance(for: generator.key(for: count))])
        }
      }
    }

    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_getter_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (d, ref) = tracker.shareableDictionary(for: fixture)
        for (k, v) in ref {
          expectEqual(d[k], v, "\(k)")
        }
      }
    }
  }

  func test_subscript_setter_update_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("key", in: 0 ..< count) { key in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, ref) = tracker.shareableDictionary(
                0 ..< count, with: generator)
              let key = tracker.instance(for: generator.key(for: key))
              let value = tracker.instance(for: generator.value(for: -1))
              withHiddenCopies(if: isShared, of: &d) { d in
                d[key] = value
                ref[key] = value
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 40
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_setter_update_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            let replacement = tracker.instance(for: -1000)
            withHiddenCopies(if: isShared, of: &d) { d in
              let key = ref[offset].key
              d[key] = replacement
              ref[offset].value = replacement
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_remove_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("key", in: 0 ..< count) { key in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, reference) = tracker.shareableDictionary(keys: 0 ..< count)
              let key = tracker.instance(for: key)
              withHiddenCopies(if: isShared, of: &d) { d in
                d[key] = nil
                reference.removeValue(forKey: key)
                expectEqualDictionaries(d, reference)
              }
            }
          }
        }
      }
    }
    let c = 40
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_setter_remove_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            withHiddenCopies(if: isShared, of: &d) { d in
              d[ref[offset].key] = nil
              ref.remove(at: offset)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_remove_fixtures_removeAll() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withEvery("i", in: 0 ..< ref.count) { _ in
            withHiddenCopies(if: isShared, of: &d) { d in
              d[ref[0].key] = nil
              ref.remove(at: 0)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_insert_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            let keys = tracker.instances(
              for: (0 ..< count).map { generator.key(for: $0) })
            let values = tracker.instances(
              for: (0 ..< count).map { generator.value(for: $0) })
            var d: TreeDictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            var ref: Dictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            withEvery("offset", in: 0 ..< count) { offset in
              withHiddenCopies(if: isShared, of: &d) { d in
                d[keys[offset]] = values[offset]
                ref[keys[offset]] = values[offset]
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_setter_insert_fixtures() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var d: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
          var ref: Dictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
          withEvery("i", in: 0 ..< fixture.count) { i in
            withHiddenCopies(if: isShared, of: &d) { d in
              let item = fixture.itemsInInsertionOrder[i]
              let key = tracker.instance(for: item)
              let value = tracker.instance(for: 1000 + item.identity)
              d[key] = value
              ref[key] = value
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_noop() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(0 ..< count, with: generator)
            let key = tracker.instance(for: generator.key(for: -1))
            withHiddenCopies(if: isShared, of: &d) { d in
              d[key] = nil
            }
            expectEqualDictionaries(d, ref)
          }
        }
      }
    }
    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_modify_basics() {
    func check<G: DataGenerator>(count: Int, generator: G) where G.Value == Int {
      context.withTrace("count: \(count), generator: \(generator)") {
        var d: TreeDictionary<G.Key, G.Value> = [:]
        var ref: Dictionary<G.Key, G.Value> = [:]

        // Insertions
        withEvery("i", in: 0 ..< count) { i in
          let key = generator.key(for: i)
          let value = generator.value(for: i)
          mutate(&d[key]) { v in
            expectNil(v)
            v = value
          }
          ref[key] = value
          expectEqualDictionaries(d, ref)
        }

        // Updates
        withEvery("i", in: 0 ..< count) { i in
          let key = generator.key(for: i)
          let value = generator.value(for: i)

          mutate(&d[key]) { v in
            expectEqual(v, value)
            v! *= 2
          }
          ref[key]! *= 2
          expectEqualDictionaries(d, ref)
        }

        // Removals
        withEvery("i", in: 0 ..< count) { i in
          let key = generator.key(for: i)
          let value = generator.value(for: i)

          mutate(&d[key]) { v in
            expectEqual(v, 2 * value)
            v = nil
          }
          ref[key] = nil
          expectEqualDictionaries(d, ref)
        }
      }
    }

    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 3, valueOffset: c))
  }

  func test_subscript_modify_update_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("key", in: 0 ..< count) { key in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, ref) = tracker.shareableDictionary(
                0 ..< count, with: generator)
              let key = tracker.instance(for: generator.key(for: key))
              let replacement = tracker.instance(for: generator.value(for: -1))
              withHiddenCopies(if: isShared, of: &d) { d in
                mutate(&d[key]) { value in
                  expectNotNil(value)
                  value = replacement
                }
                ref[key] = replacement
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 50
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_modify_update_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            let replacement = tracker.instance(for: -1000)
            withHiddenCopies(if: isShared, of: &d) { d in
              let key = ref[offset].key
              mutate(&d[key]) { value in
                expectNotNil(value)
                value = replacement
              }
              ref[offset].value = replacement
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_in_place() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withLifetimeTracking { tracker in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          let key = ref[offset].key
          mutate(&d[key]) { value in
            expectNotNil(value)
            expectTrue(isKnownUniquelyReferenced(&value))
          }
        }
      }
    }
  }



  func test_subscript_modify_remove_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            withHiddenCopies(if: isShared, of: &d) { d in
              mutate(&d[ref[offset].key]) { value in
                expectNotNil(value)
                value = nil
              }
              ref.remove(at: offset)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_remove_fixtures_removeAll() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withEvery("i", in: 0 ..< ref.count) { i in
            withHiddenCopies(if: isShared, of: &d) { d in
              mutate(&d[ref[0].key]) { value in
                expectNotNil(value)
                value = nil
              }
              ref.remove(at: 0)
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_insert_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            let keys = tracker.instances(
              for: (0 ..< count).map { generator.key(for: $0) })
            let values = tracker.instances(
              for: (0 ..< count).map { generator.value(for: $0) })
            var d: TreeDictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            var ref: Dictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            withEvery("offset", in: 0 ..< count) { offset in
              withHiddenCopies(if: isShared, of: &d) { d in
                mutate(&d[keys[offset]]) { value in
                  expectNil(value)
                  value = values[offset]
                }
                ref[keys[offset]] = values[offset]
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_subscript_modify_insert_fixtures() {
    withEachFixture { fixture in
      withEvery("seed", in: 0..<3) { seed in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var d: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            var ref: Dictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            withEvery("i", in: 0 ..< fixture.count) { i in
              withHiddenCopies(if: isShared, of: &d) { d in
                let item = fixture.itemsInInsertionOrder[i]
                let key = tracker.instance(for: item)
                let value = tracker.instance(for: 1000 + item.identity)
                mutate(&d[key]) { v in
                  expectNil(v)
                  v = value
                }
                ref[key] = value
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_noop_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(0 ..< count, with: generator)
            let key = tracker.instance(for: generator.key(for: -1))
            withHiddenCopies(if: isShared, of: &d) { d in
              mutate(&d[key]) { value in
                expectNil(value)
                value = nil
              }
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
    let c = 50
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_defaulted_subscript_basics() {
    var d: TreeDictionary<Int, Int> = [:]

    expectEqual(d[1, default: 0], 0)

    d[1, default: 0] = 2
    expectEqual(d[1, default: 0], 2)
    expectEqual(d[2, default: 0], 0)

    mutate(&d[2, default: 0]) { value in
      expectEqual(value, 0)
      value = 4
    }
    expectEqual(d[2, default: 0], 4)

    mutate(&d[2, default: 0]) { value in
      expectEqual(value, 4)
      value = 6
    }
    expectEqual(d[2, default: 0], 6)
  }

  func test_defaulted_subscript_getter_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (d, ref) = tracker.shareableDictionary(for: fixture)
        let def = tracker.instance(for: -1)
        for (k, v) in ref {
          expectEqual(d[k, default: def], v, "\(k)")
        }
      }
    }
  }

  func test_defaulted_subscript_setter_update_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("key", in: 0 ..< count) { key in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, ref) = tracker.shareableDictionary(
                0 ..< count, with: generator)
              let key = tracker.instance(for: generator.key(for: key))
              let value = tracker.instance(for: generator.value(for: -1))
              let def = tracker.instance(for: generator.value(for: -2))
              withHiddenCopies(if: isShared, of: &d) { d in
                d[key, default: def] = value
                ref[key, default: def] = value
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 40
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_defaulted_subscript_setter_update_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            let replacement = tracker.instance(for: -1000)
            let def = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d) { d in
              let key = ref[offset].key
              d[key, default: def] = replacement
              ref[offset].value = replacement
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_defaulted_subscript_setter_insert_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            let keys = tracker.instances(
              for: (0 ..< count).map { generator.key(for: $0) })
            let values = tracker.instances(
              for: (0 ..< count).map { generator.value(for: $0) })
            var d: TreeDictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            var ref: Dictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            let def = tracker.instance(for: generator.value(for: -1000))
            withEvery("offset", in: 0 ..< count) { offset in
              withHiddenCopies(if: isShared, of: &d) { d in
                d[keys[offset], default: def] = values[offset]
                ref[keys[offset]] = values[offset]
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_defaulted_subscript_setter_insert_fixtures() {
    withEachFixture { fixture in
      withEvery("seed", in: 0..<3) { seed in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var d: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            var ref: Dictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            let def = tracker.instance(for: -1000)
            withEvery("i", in: 0 ..< fixture.count) { i in
              withHiddenCopies(if: isShared, of: &d) { d in
                let item = fixture.itemsInInsertionOrder[i]
                let key = tracker.instance(for: item)
                let value = tracker.instance(for: 1000 + item.identity)
                d[key, default: def] = value
                ref[key] = value
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
  }

  func test_defaulted_subscript_modify_update_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("key", in: 0 ..< count) { key in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, ref) = tracker.shareableDictionary(
                0 ..< count, with: generator)
              let key = tracker.instance(for: generator.key(for: key))
              let replacement = tracker.instance(for: generator.value(for: -1))
              let def = tracker.instance(for: generator.value(for: -2))
              withHiddenCopies(if: isShared, of: &d) { d in
                mutate(&d[key, default: def]) { value in
                  expectNotEqual(value, def)
                  value = replacement
                }
                ref[key] = replacement
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 50
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_defaulted_subscript_modify_update_fixtures() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            let replacement = tracker.instance(for: -1000)
            let def = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d) { d in
              let key = ref[offset].key
              mutate(&d[key, default: def]) { value in
                expectNotNil(value)
                value = replacement
              }
              ref[offset].value = replacement
              expectEqualDictionaries(d, ref)
            }
          }
        }
      }
    }
  }

  func test_defaulted_subscript_modify_in_place() {
    withEachFixture { fixture in
      withEvery("offset", in: 0 ..< fixture.count) { offset in
        withLifetimeTracking { tracker in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          let key = ref[offset].key
          mutate(&d[key, default: tracker.instance(for: -1)]) { value in
            expectNotNil(value)
            expectTrue(isKnownUniquelyReferenced(&value))
          }
        }
      }
    }
  }

  func test_defaulted_subscript_modify_insert_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            let keys = tracker.instances(
              for: (0 ..< count).map { generator.key(for: $0) })
            let values = tracker.instances(
              for: (0 ..< count).map { generator.value(for: $0) })
            let def = tracker.instance(for: generator.value(for: -1000))
            var d: TreeDictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            var ref: Dictionary<LifetimeTracked<G.Key>, LifetimeTracked<G.Value>> = [:]
            withEvery("offset", in: 0 ..< count) { offset in
              withHiddenCopies(if: isShared, of: &d) { d in
                mutate(&d[keys[offset], default: def]) { value in
                  expectEqual(value, def)
                  value = values[offset]
                }
                d._invariantCheck()
                ref[keys[offset]] = values[offset]
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
    let c = 100
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_defaulted_subscript_modify_insert_fixtures() {
    withEachFixture { fixture in
      withEvery("seed", in: 0..<3) { seed in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var d: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            var ref: Dictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>> = [:]
            let def = tracker.instance(for: -1)
            withEvery("i", in: 0 ..< fixture.count) { i in
              withHiddenCopies(if: isShared, of: &d) { d in
                let item = fixture.itemsInInsertionOrder[i]
                let key = tracker.instance(for: item)
                let value = tracker.instance(for: 1000 + item.identity)
                mutate(&d[key, default: def]) { v in
                  expectEqual(v, def)
                  v = value
                }
                ref[key] = value
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
  }


  func test_indexForKey_data() {
    func check<G: DataGenerator>(count: Int, generator: G) {
      context.withTrace("count: \(count), generator: \(generator)") {
        withLifetimeTracking { tracker in
          let (d, ref) = tracker.shareableDictionary(0 ..< count, with: generator)
          withEvery("key", in: ref.keys) { key in
            let index = d.index(forKey: key)
            expectNotNil(index) { index in
              expectEqual(d[index].key, key)
              expectEqual(d[index].value, ref[key])
            }
          }
        }
      }
    }
    let c = 50
    check(count: c, generator: IntDataGenerator(valueOffset: c))
    check(count: c, generator: ColliderDataGenerator(groups: 5, valueOffset: c))
  }

  func test_indexForKey_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (d, ref) = tracker.shareableDictionary(for: fixture)
        withEvery("offset", in: ref.indices) { offset in
          let key = ref[offset].key
          let value = ref[offset].value
          let index = d.index(forKey: key)
          expectNotNil(index) { index in
            expectEqual(d[index].key, key)
            expectEqual(d[index].value, value)
          }
        }
      }
    }
  }

  func test_removeAt_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        withEvery("isShared", in: [false, true]) { isShared in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withEvery("i", in: ref.indices) { _ in
            let (key, value) = ref.removeFirst()
            let index = d.index(forKey: key)
            expectNotNil(index) { index in
              withHiddenCopies(if: isShared, of: &d) { d in
                let (k, v) = d.remove(at: index)
                expectEqual(k, key)
                expectEqual(v, value)
                expectEqualDictionaries(d, ref)
              }
            }
          }
        }
      }
    }
  }

  func test_mapValues_basics() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = TreeDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.mapValues { value -> String in
      c += 1
      expectTrue(value.isMultiple(of: 100))
      return "\(value)"
    }
    expectEqual(c, 100)
    expectEqualDictionaries(d, items)

    expectEqualDictionaries(d2, (0 ..< 100).compactMap { key in
      (key: key, value: "\(100 * key)")
    })
  }

  func test_mapValues_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        withEvery("isShared", in: [false, true]) { isShared in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withHiddenCopies(if: isShared, of: &d) { d in
            let d2 = d.mapValues { tracker.instance(for: "\($0.payload)") }
            let ref2 = Dictionary(uniqueKeysWithValues: ref.lazy.map {
              ($0.key, tracker.instance(for: "\($0.value.payload)"))
            })
            expectEqualDictionaries(d2, ref2)
          }
        }
      }
    }
  }

  func test_compactMapValues_basics() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = TreeDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.compactMapValues { value -> String? in
      c += 1
      guard value.isMultiple(of: 200) else { return nil }
      expectTrue(value.isMultiple(of: 100))
      return "\(value)"
    }
    expectEqual(c, 100)
    expectEqualDictionaries(d, items)

    expectEqualDictionaries(d2, (0 ..< 50).map { key in
      (key: 2 * key, value: "\(200 * key)")
    })
  }

  func test_compactMapValues_fixtures() {
    typealias Key = LifetimeTracked<RawCollider>
    typealias Value = LifetimeTracked<Int>
    typealias Value2 = LifetimeTracked<String>

    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        func transform(_ value: Value) -> Value2? {
          guard value.payload.isMultiple(of: 2) else { return nil }
          return tracker.instance(for: "\(value.payload)")
        }

        withEvery("isShared", in: [false, true]) { isShared in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withHiddenCopies(if: isShared, of: &d) { d in
            let d2 = d.compactMapValues(transform)
            let r: [(Key, Value2)] = ref.compactMap {
              guard let v = transform($0.value) else { return nil }
              return ($0.key, v)
            }
            let ref2 = Dictionary(uniqueKeysWithValues: r)
            expectEqualDictionaries(d2, ref2)
          }
        }
      }
    }
  }

  func test_filter_basics() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = TreeDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.filter { item in
      c += 1
      expectEqual(item.value, 100 * item.key)
      return item.key.isMultiple(of: 2)
    }
    expectEqual(c, 100)
    expectEqualDictionaries(d, items)

    expectEqualDictionaries(d2, (0 ..< 50).compactMap { key in
      return (key: 2 * key, value: 200 * key)
    })
  }

  func test_filter_fixtures_evens() {
    typealias Key = LifetimeTracked<RawCollider>
    typealias Value = LifetimeTracked<Int>

    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        withEvery("isShared", in: [false, true]) { isShared in
          var (d, ref) = tracker.shareableDictionary(for: fixture)
          withHiddenCopies(if: isShared, of: &d) { d in
            func predicate(_ item: (key: Key, value: Value)) -> Bool {
              expectEqual(item.value.payload, 1000 + item.key.payload.identity)
              return item.value.payload.isMultiple(of: 2)
            }
            let d2 = d.filter(predicate)
            let ref2 = Dictionary(
              uniqueKeysWithValues: ref.filter(predicate))
            expectEqualDictionaries(d2, ref2)
          }
        }
      }
    }
  }

  func test_filter_fixtures_filter_one() {
    typealias Key = LifetimeTracked<RawCollider>
    typealias Value = LifetimeTracked<Int>

    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        withEvery("isShared", in: [false, true]) { isShared in
          withEvery("i", in: 0 ..< fixture.count) { i in
            var (d, ref) = tracker.shareableDictionary(for: fixture)
            withHiddenCopies(if: isShared, of: &d) { d in
              func predicate(_ item: (key: Key, value: Value)) -> Bool {
                expectEqual(item.value.payload, 1000 + item.key.payload.identity)
                return item.key.payload.identity == i
              }
              let d2 = d.filter(predicate)
              let ref2 = Dictionary(
                uniqueKeysWithValues: ref.filter(predicate))
              expectEqualDictionaries(d2, ref2)
            }
          }
        }
      }
    }
  }

  func test_filter_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeDictionary(
        uniqueKeysWithValues: a.lazy.map {
          ($0, $0.identity + 100)
        })
      withEverySubset("b", of: a) { b in
        let y = Dictionary(
          uniqueKeysWithValues: b.lazy.map {
            ($0, $0.identity + 100)
          })
        expectEqualDictionaries(x.filter { y[$0.key] == $0.value }, y)
      }
    }
  }

  func test_removeAll_where_exhaustive() {
    withEvery("isShared", in: [false, true]) { isShared in
      withEverySubset("a", of: testItems) { a in
        withEverySubset("b", of: a) { b in
          var x = TreeDictionary(
            uniqueKeysWithValues: a.lazy.map {
              ($0, $0.identity + 100)
            })
          let y = Dictionary(
            uniqueKeysWithValues: b.lazy.map {
              ($0, $0.identity + 100)
            })
          withHiddenCopies(if: isShared, of: &x) { x in
            x.removeAll { y[$0.key] == nil }
            expectEqualDictionaries(x, y)
          }
        }
      }
    }
  }

  func test_merge_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      withEverySubset("b", of: testItems) { b in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            func combine(
              _ a: LifetimeTracked<Int>, _ b: LifetimeTracked<Int>
            ) -> LifetimeTracked<Int> {
              expectEqual(b.payload, a.payload + 100)
              return tracker.instance(for: 100 - a.payload)
            }
            var expected1 = tracker.dictionary(
              for: a, by: { $0.identity + 100 })
            let expected2 = tracker.dictionary(
              for: b, by: { $0.identity + 200 })

            var actual1 = tracker.shareableDictionary(
              for: a, by: { $0.identity + 100 })
            let actual2 = tracker.shareableDictionary(
              for: b, by: { $0.identity + 200 })

            expected1.merge(expected2, uniquingKeysWith: combine)

            withHiddenCopies(if: isShared, of: &expected1) { expected1 in
              actual1.merge(actual2, uniquingKeysWith: combine)
              expectEqualDictionaries(actual1, expected1)
            }
          }
        }
      }
    }
  }


  // MARK: -

  //  func test_uniqueKeysWithValues_Dictionary() {
  //    let items: Dictionary<String, Int> = [
  //      "zero": 0,
  //      "one": 1,
  //      "two": 2,
  //      "three": 3,
  //    ]
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //    expectEqualElements(d.sorted(by: <), items.sorted(by: <))
  //  }

  //  func test_uniqueKeysWithValues_labeled_tuples() {
  //    let items: KeyValuePairs<String, Int> = [
  //      "zero": 0,
  //      "one": 1,
  //      "two": 2,
  //      "three": 3,
  //    ]
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //    expectEqualElements(d.sorted(by: <), items.sorted(by: <))
  //  }

  func test_uniqueKeysWithValues_unlabeled_tuples() {
    let items: [(String, Int)] = [
      ("zero", 0),
      ("one", 1),
      ("two", 2),
      ("three", 3),
    ]
    let d = TreeDictionary(uniqueKeysWithValues: items)
    expectEqualElements(d.sorted(by: <), items.sorted(by: <))
  }

  //  func test_uniquing_initializer_labeled_tuples() {
  //    let items: KeyValuePairs<String, Int> = [
  //      "a": 1,
  //      "b": 1,
  //      "c": 1,
  //      "a": 2,
  //      "a": 2,
  //      "b": 1,
  //      "d": 3,
  //    ]
  //    let d = TreeDictionary(items, uniquingKeysWith: +)
  //    expectEqualElements(d, [
  //      (key: "a", value: 5),
  //      (key: "b", value: 2),
  //      (key: "c", value: 1),
  //      (key: "d", value: 3)
  //    ])
  //  }

  //  func test_uniquing_initializer_unlabeled_tuples() {
  //    let items: [(String, Int)] = [
  //      ("a", 1),
  //      ("b", 1),
  //      ("c", 1),
  //      ("a", 2),
  //      ("a", 2),
  //      ("b", 1),
  //      ("d", 3),
  //    ]
  //    let d = TreeDictionary(items, uniquingKeysWith: +)
  //    expectEqualElements(d, [
  //      (key: "a", value: 5),
  //      (key: "b", value: 2),
  //      (key: "c", value: 1),
  //      (key: "d", value: 3)
  //    ])
  //  }

  //  func test_grouping_initializer() {
  //    let items: [String] = [
  //      "one", "two", "three", "four", "five",
  //      "six", "seven", "eight", "nine", "ten"
  //    ]
  //    let d = TreeDictionary<Int, [String]>(
  //      grouping: items,
  //      by: { $0.count })
  //    expectEqualElements(d, [
  //      (key: 3, value: ["one", "two", "six", "ten"]),
  //      (key: 5, value: ["three", "seven", "eight"]),
  //      (key: 4, value: ["four", "five", "nine"]),
  //    ])
  //  }

  //  func test_uniqueKeysWithValues_labeled_tuples() {
  //    let items: KeyValuePairs<String, Int> = [
  //      "zero": 0,
  //      "one": 1,
  //      "two": 2,
  //      "three": 3,
  //    ]
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //    expectEqualElements(d, items)
  //  }

  //  func test_uniqueKeysWithValues_unlabeled_tuples() {
  //    let items: [(String, Int)] = [
  //      ("zero", 0),
  //      ("one", 1),
  //      ("two", 2),
  //      ("three", 3),
  //    ]
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //    expectEqualElements(d, items)
  //  }

  func test_ExpressibleByDictionaryLiteral() {
    let d0: TreeDictionary<String, Int> = [:]
    expectTrue(d0.isEmpty)

    let d1: TreeDictionary<String, Int> = [
      "1~one": 1,
      "2~two": 2,
      "3~three": 3,
      "4~four": 4,
    ]
    expectEqualElements(
      d1.map { $0.key }.sorted(),
      ["1~one", "2~two", "3~three", "4~four"])
    expectEqualElements(
      d1.map { $0.value }.sorted(),
      [1, 2, 3, 4])
  }

  //  func test_keys() {
  //    let d: TreeDictionary = [
  //      "one": 1,
  //      "two": 2,
  //      "three": 3,
  //      "four": 4,
  //    ]
  //    expectEqual(d.keys, ["one", "two", "three", "four"] as OrderedSet)
  //  }

  func test_counts() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, _, _) = tracker.shareableDictionary(keys: 0 ..< count)
        expectEqual(d.isEmpty, count == 0)
        expectEqual(d.count, count)
        expectEqual(d.underestimatedCount, count)
      }
    }
  }

  #if false
  // TODO: determine how to best calculate the expected order of the hash tree
  // for testing purposes, without relying on the actual implementation
  func test_index_forKey() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, _, _) = tracker.shareableDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          expectEqual(
            // NOTE: uses the actual order `d.keys`
            d.index(forKey: d.keys[offset])?._value,
            offset)
        }
        expectNil(d.index(forKey: tracker.instance(for: -1)))
        expectNil(d.index(forKey: tracker.instance(for: count)))
      }
    }
  }
  #endif

  #if false
  // TODO: determine how to best calculate the expected order of the hash tree
  // for testing purposes, without relying on the actual implementation
  func test_subscript_offset() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, _, _) = tracker.shareableDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          let item = d[TreeDictionaryIndex(value: offset)]
          // NOTE: uses the actual order `d.keys`
          expectEqual(item.key, d.keys[offset])
          // NOTE: uses the actual order `d.values`
          expectEqual(item.value, d.values[offset])
        }
      }
    }
  }
  #endif

  func test_subscript_getter() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          expectEqual(d[keys[offset]], values[offset])
        }
        expectNil(d[tracker.instance(for: -1)])
        expectNil(d[tracker.instance(for: count)])
      }
    }
  }

//    func test_subscript_setter_update() {
//      withEvery("count", in: 0 ..< 30) { count in
//        withEvery("offset", in: 0 ..< count) { offset in
//          withEvery("isShared", in: [false, true]) { isShared in
//            withLifetimeTracking { tracker in
//              var (d, keys, values) = tracker.shareableDictionary(
//                keys: 0 ..< count)
//              let replacement = tracker.instance(for: -1)
//              withHiddenCopies(if: isShared, of: &d) { d in
//                d[keys[offset]] = replacement
//                values[offset] = replacement
//                withEvery("i", in: 0 ..< count) { i in
//                  let (k, v) = d[offset: i]
//                  expectEqual(k, keys[i])
//                  expectEqual(v, values[i])
//                }
//              }
//            }
//          }
//        }
//      }
//    }

//    func test_subscript_setter_remove() {
//      withEvery("count", in: 0 ..< 30) { count in
//        withEvery("offset", in: 0 ..< count) { offset in
//          withEvery("isShared", in: [false, true]) { isShared in
//            withLifetimeTracking { tracker in
//              var (d, keys, values) = tracker.shareableDictionary(
//                keys: 0 ..< count)
//              withHiddenCopies(if: isShared, of: &d) { d in
//                d[keys[offset]] = nil
//                keys.remove(at: offset)
//                values.remove(at: offset)
//                withEvery("i", in: 0 ..< count - 1) { i in
//                  let (k, v) = d[offset: i]
//                  expectEqual(k, keys[i])
//                  expectEqual(v, values[i])
//                }
//              }
//            }
//          }
//        }
//      }
//    }

//    func test_subscript_setter_insert() {
//      withEvery("count", in: 0 ..< 30) { count in
//        withEvery("isShared", in: [false, true]) { isShared in
//          withLifetimeTracking { tracker in
//            let keys = tracker.instances(for: 0 ..< count)
//            let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
//            var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
//            withEvery("offset", in: 0 ..< count) { offset in
//              withHiddenCopies(if: isShared, of: &d) { d in
//                d[keys[offset]] = values[offset]
//                withEvery("i", in: 0 ... offset) { i in
//                  let (k, v) = d[offset: i]
//                  expectEqual(k, keys[i])
//                  expectEqual(v, values[i])
//                }
//              }
//            }
//          }
//        }
//      }
//    }

//    func test_subscript_setter_noop() {
//      withEvery("count", in: 0 ..< 30) { count in
//        withEvery("isShared", in: [false, true]) { isShared in
//          withLifetimeTracking { tracker in
//            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
//            let key = tracker.instance(for: -1)
//            withHiddenCopies(if: isShared, of: &d) { d in
//              d[key] = nil
//            }
//            withEvery("i", in: 0 ..< count) { i in
//              let (k, v) = d[offset: i]
//              expectEqual(k, keys[i])
//              expectEqual(v, values[i])
//            }
//          }
//        }
//      }
//    }

  //  func test_subscript_modify_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              mutate(&d[keys[offset]]) { $0 = replacement }
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_subscript_modify_remove() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              mutate(&d[key]) { v in
  //                expectEqual(v, values[offset])
  //                v = nil
  //              }
  //              keys.remove(at: offset)
  //              values.remove(at: offset)
  //              withEvery("i", in: 0 ..< count - 1) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_subscript_modify_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              mutate(&d[keys[offset]]) { v in
  //                expectNil(v)
  //                v = values[offset]
  //              }
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_subscript_modify_noop() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //          let key = tracker.instance(for: -1)
  //          withHiddenCopies(if: isShared, of: &d) { d in
  //            mutate(&d[key]) { v in
  //              expectNil(v)
  //              v = nil
  //            }
  //          }
  //          withEvery("i", in: 0 ..< count) { i in
  //            let (k, v) = d[offset: i]
  //            expectEqual(k, keys[i])
  //            expectEqual(v, values[i])
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_defaulted_subscript_getter() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //          let fallback = tracker.instance(for: -1)
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            let key = keys[offset]
  //            expectEqual(d[key, default: fallback], values[offset])
  //          }
  //          expectEqual(
  //            d[tracker.instance(for: -1), default: fallback],
  //            fallback)
  //          expectEqual(
  //            d[tracker.instance(for: count), default: fallback],
  //            fallback)
  //        }
  //      }
  //    }
  //  }

  //  func test_defaulted_subscript_modify_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            let fallback = tracker.instance(for: -1)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              mutate(&d[key, default: fallback]) { v in
  //                expectEqual(v, values[offset])
  //                v = replacement
  //              }
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_defaulted_subscript_modify_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          let fallback = tracker.instance(for: -1)
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              mutate(&d[key, default: fallback]) { v in
  //                expectEqual(v, fallback)
  //                v = values[offset]
  //              }
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_updateValue_forKey_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              let old = d.updateValue(replacement, forKey: key)
  //              expectEqual(old, values[offset])
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_updateValue_forKey_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              let old = d.updateValue(values[offset], forKey: key)
  //              expectNil(old)
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_updateValue_forKey_insertingAt_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              let (old, index) =
  //                d.updateValue(replacement, forKey: key, insertingAt: 0)
  //              expectEqual(old, values[offset])
  //              expectEqual(index, offset)
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_updateValue_forKey_insertingAt_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[count - 1 - offset]
  //              let value = values[count - 1 - offset]
  //              let (old, index) =
  //                d.updateValue(value, forKey: key, insertingAt: 0)
  //              expectNil(old)
  //              expectEqual(index, 0)
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[count - 1 - offset + i])
  //                expectEqual(v, values[count - 1 - offset + i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_modifyValue_forKey_default_closure_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            let fallback = tracker.instance(for: -2)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              d.modifyValue(forKey: key, default: fallback) { value in
  //                expectEqual(value, values[offset])
  //                value = replacement
  //              }
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_modifyValue_forKey_default_closure_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          let fallback = tracker.instance(for: -2)
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              d.modifyValue(forKey: key, default: fallback) { value in
  //                expectEqual(value, fallback)
  //                value = values[offset]
  //              }
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_modifyValue_forKey_insertingDefault_at_closure_update() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            let replacement = tracker.instance(for: -1)
  //            let fallback = tracker.instance(for: -2)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[offset]
  //              let value = values[offset]
  //              d.modifyValue(forKey: key, insertingDefault: fallback, at: 0) { v in
  //                expectEqual(v, value)
  //                v = replacement
  //              }
  //              values[offset] = replacement
  //              withEvery("i", in: 0 ..< count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_modifyValue_forKey_insertingDefault_at_closure_insert() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          let keys = tracker.instances(for: 0 ..< count)
  //          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
  //          var d: TreeDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
  //          let fallback = tracker.instance(for: -2)
  //          withEvery("offset", in: 0 ..< count) { offset in
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys[count - 1 - offset]
  //              let value = values[count - 1 - offset]
  //              d.modifyValue(forKey: key, insertingDefault: fallback, at: 0) { v in
  //                expectEqual(v, fallback)
  //                v = value
  //              }
  //              expectEqual(d.count, offset + 1)
  //              withEvery("i", in: 0 ... offset) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[count - 1 - offset + i])
  //                expectEqual(v, values[count - 1 - offset + i])
  //              }
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeValue_forKey() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let key = keys.remove(at: offset)
  //              let expected = values.remove(at: offset)
  //              let actual = d.removeValue(forKey: key)
  //              expectEqual(actual, expected)
  //
  //              expectEqual(d.count, values.count)
  //              withEvery("i", in: 0 ..< values.count) { i in
  //                let (k, v) = d[offset: i]
  //                expectEqual(k, keys[i])
  //                expectEqual(v, values[i])
  //              }
  //              expectNil(d.removeValue(forKey: key))
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_merge_labeled_tuple() {
  //    var d: TreeDictionary = [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ]
  //
  //    let items: KeyValuePairs = [
  //      "one": 1,
  //      "one": 1,
  //      "three": 1,
  //      "four": 1,
  //      "one": 1,
  //    ]
  //
  //    d.merge(items, uniquingKeysWith: +)
  //
  //    expectEqualElements(d, [
  //      "one": 4,
  //      "two": 1,
  //      "three": 2,
  //      "four": 1,
  //    ] as KeyValuePairs)
  //  }

  //  func test_merge_unlabeled_tuple() {
  //    var d: TreeDictionary = [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ]
  //
  //    let items: [(String, Int)] = [
  //      ("one", 1),
  //      ("one", 1),
  //      ("three", 1),
  //      ("four", 1),
  //      ("one", 1),
  //    ]
  //
  //    d.merge(items, uniquingKeysWith: +)
  //
  //    expectEqualElements(d, [
  //      "one": 4,
  //      "two": 1,
  //      "three": 2,
  //      "four": 1,
  //    ] as KeyValuePairs)
  //  }

  //  func test_merging_labeled_tuple() {
  //    let d: TreeDictionary = [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ]
  //
  //    let items: KeyValuePairs = [
  //      "one": 1,
  //      "one": 1,
  //      "three": 1,
  //      "four": 1,
  //      "one": 1,
  //    ]
  //
  //    let d2 = d.merging(items, uniquingKeysWith: +)
  //
  //    expectEqualElements(d, [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ] as KeyValuePairs)
  //
  //    expectEqualElements(d2, [
  //      "one": 4,
  //      "two": 1,
  //      "three": 2,
  //      "four": 1,
  //    ] as KeyValuePairs)
  //  }

  //  func test_merging_unlabeled_tuple() {
  //    let d: TreeDictionary = [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ]
  //
  //    let items: [(String, Int)] = [
  //      ("one", 1),
  //      ("one", 1),
  //      ("three", 1),
  //      ("four", 1),
  //      ("one", 1),
  //    ]
  //
  //    let d2 = d.merging(items, uniquingKeysWith: +)
  //
  //    expectEqualElements(d, [
  //      "one": 1,
  //      "two": 1,
  //      "three": 1,
  //    ] as KeyValuePairs)
  //
  //    expectEqualElements(d2, [
  //      "one": 4,
  //      "two": 1,
  //      "three": 2,
  //      "four": 1,
  //    ] as KeyValuePairs)
  //  }

  //  func test_filter() {
  //    let items = (0 ..< 100).map { ($0, 100 * $0) }
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //
  //    var c = 0
  //    let d2 = d.filter { item in
  //      c += 1
  //      expectEqual(item.value, 100 * item.key)
  //      return item.key.isMultiple(of: 2)
  //    }
  //    expectEqual(c, 100)
  //    expectEqualElements(d, items)
  //
  //    expectEqualElements(d2, (0 ..< 50).compactMap { key in
  //      return (key: 2 * key, value: 200 * key)
  //    })
  //  }

  //  func test_mapValues() {
  //    let items = (0 ..< 100).map { ($0, 100 * $0) }
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //
  //    var c = 0
  //    let d2 = d.mapValues { value -> String in
  //      c += 1
  //      expectTrue(value.isMultiple(of: 100))
  //      return "\(value)"
  //    }
  //    expectEqual(c, 100)
  //    expectEqualElements(d, items)
  //
  //    expectEqualElements(d2, (0 ..< 100).compactMap { key in
  //      (key: key, value: "\(100 * key)")
  //    })
  //  }

  //  func test_compactMapValue() {
  //    let items = (0 ..< 100).map { ($0, 100 * $0) }
  //    let d = TreeDictionary(uniqueKeysWithValues: items)
  //
  //    var c = 0
  //    let d2 = d.compactMapValues { value -> String? in
  //      c += 1
  //      guard value.isMultiple(of: 200) else { return nil }
  //      expectTrue(value.isMultiple(of: 100))
  //      return "\(value)"
  //    }
  //    expectEqual(c, 100)
  //    expectEqualElements(d, items)
  //
  //    expectEqualElements(d2, (0 ..< 50).map { key in
  //      (key: 2 * key, value: "\(200 * key)")
  //    })
  //  }

  //  func test_removeAll() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("isShared", in: [false, true]) { isShared in
  //        withLifetimeTracking { tracker in
  //          var (d, _, _) = tracker.shareableDictionary(keys: 0 ..< count)
  //          withHiddenCopies(if: isShared, of: &d) { d in
  //            d.removeAll()
  //            expectEqual(d.keys.__unstable.scale, 0)
  //            expectEqualElements(d, [])
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_remove_at() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("offset", in: 0 ..< count) { offset in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              let actual = d.remove(at: offset)
  //              let expectedKey = keys.remove(at: offset)
  //              let expectedValue = values.remove(at: offset)
  //              expectEqual(actual.key, expectedKey)
  //              expectEqual(actual.value, expectedValue)
  //              expectEqualElements(
  //                d,
  //                zip(keys, values).map { (key: $0.0, value: $0.1) })
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeSubrange() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEveryRange("range", in: 0 ..< count) { range in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              d.removeSubrange(range)
  //              keys.removeSubrange(range)
  //              values.removeSubrange(range)
  //              expectEqualElements(
  //                d,
  //                zip(keys, values).map { (key: $0.0, value: $0.1) })
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeSubrange_rangeExpression() {
  //    let d = TreeDictionary(uniqueKeys: 0 ..< 30, values: 100 ..< 130)
  //    let item = (0 ..< 30).map { (key: $0, value: 100 + $0) }
  //
  //    var d1 = d
  //    d1.removeSubrange(...10)
  //    expectEqualElements(d1, item[11...])
  //
  //    var d2 = d
  //    d2.removeSubrange(..<10)
  //    expectEqualElements(d2, item[10...])
  //
  //    var d3 = d
  //    d3.removeSubrange(10...)
  //    expectEqualElements(d3, item[0 ..< 10])
  //  }
  //
  //  func test_removeLast() {
  //    withEvery("isShared", in: [false, true]) { isShared in
  //      withLifetimeTracking { tracker in
  //        var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< 30)
  //        withEvery("i", in: 0 ..< d.count) { i in
  //          withHiddenCopies(if: isShared, of: &d) { d in
  //            let actual = d.removeLast()
  //            let expectedKey = keys.removeLast()
  //            let expectedValue = values.removeLast()
  //            expectEqual(actual.key, expectedKey)
  //            expectEqual(actual.value, expectedValue)
  //            expectEqualElements(
  //              d,
  //              zip(keys, values).map { (key: $0.0, value: $0.1) })
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeFirst() {
  //    withEvery("isShared", in: [false, true]) { isShared in
  //      withLifetimeTracking { tracker in
  //        var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< 30)
  //        withEvery("i", in: 0 ..< d.count) { i in
  //          withHiddenCopies(if: isShared, of: &d) { d in
  //            let actual = d.removeFirst()
  //            let expectedKey = keys.removeFirst()
  //            let expectedValue = values.removeFirst()
  //            expectEqual(actual.key, expectedKey)
  //            expectEqual(actual.value, expectedValue)
  //            expectEqualElements(
  //              d,
  //              zip(keys, values).map { (key: $0.0, value: $0.1) })
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeLast_n() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("suffix", in: 0 ..< count) { suffix in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              d.removeLast(suffix)
  //              keys.removeLast(suffix)
  //              values.removeLast(suffix)
  //              expectEqualElements(
  //                d,
  //                zip(keys, values).map { (key: $0.0, value: $0.1) })
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeFirst_n() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("prefix", in: 0 ..< count) { prefix in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              d.removeFirst(prefix)
  //              keys.removeFirst(prefix)
  //              values.removeFirst(prefix)
  //              expectEqualElements(
  //                d,
  //                zip(keys, values).map { (key: $0.0, value: $0.1) })
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  func test_removeAll_where() {
  //    withEvery("count", in: 0 ..< 30) { count in
  //      withEvery("n", in: [2, 3, 4]) { n in
  //        withEvery("isShared", in: [false, true]) { isShared in
  //          withLifetimeTracking { tracker in
  //            var (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
  //            var items = zip(keys, values).map { (key: $0.0, value: $0.1) }
  //            withHiddenCopies(if: isShared, of: &d) { d in
  //              d.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
  //              items.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
  //              expectEqualElements(d, items)
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  func test_Sequence() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, keys, values) = tracker.shareableDictionary(keys: 0 ..< count)
        let items = zip(keys, values).map { (key: $0.0, value: $0.1) }
        checkSequence(
          { d.sorted(by: <) },
          expectedContents: items,
          by: { $0.key == $1.0 && $0.value == $1.1 })
      }
    }
  }

}
