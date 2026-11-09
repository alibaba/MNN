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

#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import _CollectionsTestSupport
import HashTreeCollections
#endif

/// A set of items whose subsets will produce a bunch of interesting test
/// cases.
///
/// Note: Try to keep this short. Every new item added here will quadruple
/// testing costs.
let testItems: [RawCollider] = {
  var testItems = [
    RawCollider(1, "A"),
    RawCollider(2, "B"),
    RawCollider(3, "ACA"),
    RawCollider(4, "ACB"),
    RawCollider(5, "ACAD"),
    RawCollider(6, "ACAD"),
  ]
  if MemoryLayout<Int>.size == 8 {
    // Cut testing workload down a bit on 32-bit systems. In practice a 32-bit Int
    // usually means we're running on a watchOS device (arm64_32), and those are relatively slow
    // to run these.
    testItems += [
      RawCollider(7, "ACAEB"),
      RawCollider(8, "ACAEB"),
      RawCollider(9, "ACAEB"),
    ]
  }
  #if false // Enable for even deeper testing
  testItems += [
    RawCollider(10, "ACAEB"),
    RawCollider(11, "ADAD"),
    RawCollider(12, "ACC"),
  ]
  #endif
  return testItems
}()

extension LifetimeTracker {
  func shareableDictionary<Key, Value, C: Collection>(
    for items: C,
    keyTransform: (C.Element) -> Key,
    valueTransform: (C.Element) -> Value
  ) -> TreeDictionary<LifetimeTracked<Key>, LifetimeTracked<Value>> {
    let keys = instances(for: items, by: keyTransform)
    let values = instances(for: items, by: valueTransform)
    return TreeDictionary(uniqueKeysWithValues: zip(keys, values))
  }

  func shareableDictionary<Value, C: Collection>(
    for keys: C,
    by valueTransform: (C.Element) -> Value
  ) -> TreeDictionary<LifetimeTracked<C.Element>, LifetimeTracked<Value>>
  where C.Element: Hashable {
    let k = instances(for: keys)
    let v = instances(for: keys, by: valueTransform)
    return TreeDictionary(uniqueKeysWithValues: zip(k, v))
  }

  func dictionary<Key, Value, C: Collection>(
    for items: C,
    keyTransform: (C.Element) -> Key,
    valueTransform: (C.Element) -> Value
  ) -> Dictionary<LifetimeTracked<Key>, LifetimeTracked<Value>> {
    Dictionary(
      uniqueKeysWithValues: zip(
        instances(for: items, by: keyTransform),
        instances(for: items, by: valueTransform)))
  }

  func dictionary<Value, C: Collection>(
    for keys: C,
    by valueTransform: (C.Element) -> Value
  ) -> Dictionary<LifetimeTracked<C.Element>, LifetimeTracked<Value>>
  where C.Element: Hashable {
    Dictionary(
      uniqueKeysWithValues: zip(
        instances(for: keys),
        instances(for: keys, by: valueTransform)))
  }

}

/// A list of example trees to use while testing persistent hash maps.
///
/// Each example has a name and a list of path specifications or collisions.
///
/// A path spec is an ASCII `String` representing the hash of a key/value pair,
/// a.k.a a path in the prefix tree. Each character in the string identifies
/// a bucket index of a tree node, starting from the root.
/// (Encoded in radix 32, with digits 0-9 followed by letters. In order to
/// prepare for a potential reduction in the maximum node size, it is best to
/// keep the digits in the range 0-F.) The prefix tree's depth is limited by the
/// size of hash values.
///
/// For example, the string "5A" corresponds to a key/value pair
/// that is in bucket 10 of a second-level node that is found at bucket 5
/// of the root node.
///
/// Hash collisions are modeled by strings of the form `<path>*<count>` where
/// `<path>` is a path specification, and `<count>` is the number of times that
/// path needs to be repeated. (To implement the collisions, the path is
/// extended with an infinite number of zeroes.)
///
/// To generate input data from these fixtures, the items are sorted into
/// the same order as we expect a preorder walk would visit them in the
/// resulting tree. The resulting ordering is then used to insert key/value
/// pairs into the map, with sequentially increasing keys.
let fixtures: [Fixture] = {
  var fixtures: Array<Fixture> = []

  enum FixtureFlavor {
    case any
    case small // 32-bit platforms
    case large // 64-bit platforms

    func isAllowed() -> Bool {
      let reject: FixtureFlavor
#if _pointerBitWidth(_32)
      reject = .large
#elseif _pointerBitWidth(_64)
      precondition(MemoryLayout<Int>.size == 8, "Unknown platform")
      reject = .small
#else
#error("Unexpected pointer bit width")
#endif
      return self != reject
    }
  }

  func add(_ title: String, flavor: FixtureFlavor = .any, _ contents: [String]) {
    // Ignore unsupported flavors
    guard flavor.isAllowed() else { return }
    fixtures.append(Fixture(title: title, contents: contents))
  }
  
  add("empty", [])
  add("single-item", ["A"])
  add("single-node", [
    "0",
    "1",
    "2",
    "3",
    "4",
    "A",
    "B",
    "C",
    "D",
  ])
  add("few-collisions", [
    "42*5"
  ])
  add("many-collisions", [
    "42*40"
  ])
  
  add("few-different-collisions", [
    "1*3",
    "21*3",
    "22*3",
    "3*3",
  ])
  
  add("everything-on-the-2nd-level", [
    "00", "01", "02", "03", "04",
    "10", "11", "12", "13", "14",
    "20", "21", "22", "23", "24",
    "30", "31", "32", "33", "34",
  ])
  add("two-levels-mixed", [
    "00", "01",
    "2",
    "30", "33",
    "4",
    "5",
    "60", "61", "66",
    "71", "75", "77",
    "8",
    "94", "98", "9A",
    "A3", "A4",
  ])
  add("vee", [
    "11110",
    "11115",
    "11119",
    "1111B",
    "66664",
    "66667",
  ])
  
  add("fork", [
    "31110",
    "31115",
    "31119",
    "3111B",
    "36664",
    "36667",
  ])
  add("chain-left", [
    "0",
    "10",
    "110",
    "1110",
    "11110",
    "11111",
  ])
  add("chain-right", [
    "1",
    "01",
    "001",
    "0001",
    "00001",
    "000001",
  ])
  add("expansion0", [
    "000001*3",
    "0001",
  ])
  add("expansion1", [
    "000001*3",
    "01",
    "0001",
  ])
  add("expansion2", [
    "111111*3",
    "10",
    "1110",
  ])
  add("expansion3", [
    "01",
    "0001",
    "000001*3",
  ])
  add("expansion4", [
    "10",
    "1110",
    "111111*3",
  ])
  add("nested", flavor: .large, [
    "50",
    "51",
    "520",
    "521",
    "5220",
    "5221",
    "52220",
    "52221",
    "522220",
    "522221",
    "5222220",
    "5222221",
    "52222220",
    "52222221",
    "522222220",
    "522222221",
    "5222222220",
    "5222222221",
    "5222222222",
    "5222222223",
    "522222223",
    "522222224",
    "52222223",
    "52222224",
    "5222223",
    "5222224",
    "522223",
    "522224",
    "52223",
    "52224",
    "5223",
    "5224",
    "53",
    "54",
  ])
  add("deep", [
    "0",
    
    // Deeply nested children with only the leaf containing items
    "123450",
    "123451",
    "123452",
    "123453",
    
    "22",
    "25",
  ])
  return fixtures
}()

struct Fixture {
  let title: String
  let itemsInIterationOrder: [RawCollider]
  let itemsInInsertionOrder: [RawCollider]

  init(title: String, contents: [String]) {
    self.title = title

    let maxDepth = TreeDictionary<Int, Int>._maxDepth

    func normalized(_ path: String) -> String {
      precondition(path.unicodeScalars.count < maxDepth)
      let c = Swift.max(0, maxDepth - path.unicodeScalars.count)
      return path.uppercased() + String(repeating: "0", count: c)
    }

    var items: [(path: String, item: RawCollider)] = []
    var seen: Set<String> = []
    for path in contents {
      if let i = path.unicodeScalars.firstIndex(of: "*") {
        // We need to extend the path of collisions with zeroes to
        // make sure they sort correctly.
        let p = String(path.unicodeScalars.prefix(upTo: i))
        guard let count = Int(path.suffix(from: i).dropFirst(), radix: 10)
        else { fatalError("Invalid item: '\(path)'") }
        let path = normalized(p)
        let hash = Hash(path)!
        for _ in 0 ..< count {
          items.append((path, RawCollider(items.count, hash)))
        }
      } else {
        let path = normalized(path)
        let hash = Hash(path)!
        items.append((path, RawCollider(items.count, hash)))
      }

      if !seen.insert(path).inserted {
        fatalError("Unexpected duplicate path: '\(path)'")
      }
    }

    var seenPrefixes: Set<Substring> = []
    var collidingPrefixes: Set<Substring> = []
    for p in items {
      assert(p.path.count == maxDepth)
      for i in p.path.indices {
        let prefix = p.path[..<i]
        if !seenPrefixes.insert(prefix).inserted {
          collidingPrefixes.insert(prefix)
        }
      }
      if !seenPrefixes.insert(p.path[...]).inserted {
        collidingPrefixes.insert(p.path[...])
      }
    }

    self.itemsInInsertionOrder = items.map { $0.item }

    // Sort paths into the order that we expect items will appear in the
    // dictionary.
    items.sort { a, b in
      var i = a.path.startIndex
      var j = b.path.startIndex
      while i < a.path.endIndex && j < b.path.endIndex {
        let ac = collidingPrefixes.contains(a.path[...i])
        let bc = collidingPrefixes.contains(b.path[...j])
        switch (ac, bc) {
        case (true, false): return false
        case (false, true): return true
        default: break
        }
        if a.path[i] < b.path[j] { return true }
        if a.path[j] > b.path[j] { return false }
        a.path.formIndex(after: &i)
        b.path.formIndex(after: &j)
      }
      precondition(i == a.path.endIndex && j == b.path.endIndex)
      return false
    }

    self.itemsInIterationOrder = items.map { $0.item }
  }

  var count: Int { itemsInInsertionOrder.count }
}

func withEachFixture(
  _ label: String = "fixture",
  body: (Fixture) -> Void
) {
  for fixture in fixtures {
    let entry = TestContext.current.push("\(label): \(fixture.title)")
    defer { TestContext.current.pop(entry) }

    body(fixture)
  }
}

extension LifetimeTracker {
  func shareableSet<Element: Hashable>(
    for fixture: Fixture,
    with transform: (RawCollider) -> Element
  ) -> (
    map: TreeSet<LifetimeTracked<Element>>,
    ref: [LifetimeTracked<Element>]
  ) {
    let ref = fixture.itemsInIterationOrder.map { key in
      self.instance(for: transform(key))
    }
    let ref2 = fixture.itemsInInsertionOrder.map { key in
      self.instance(for: transform(key))
    }
    return (TreeSet(ref2), ref)
  }

  func shareableSet(
    for fixture: Fixture
  ) -> (
    map: TreeSet<LifetimeTracked<RawCollider>>,
    ref: [LifetimeTracked<RawCollider>]
  ) {
    shareableSet(for: fixture) { key in key }
  }

  func shareableDictionary<Key, Value>(
    for fixture: Fixture,
    keyTransform: (RawCollider) -> Key,
    valueTransform: (RawCollider) -> Value
  ) -> (
    map: TreeDictionary<LifetimeTracked<Key>, LifetimeTracked<Value>>,
    ref: [(key: LifetimeTracked<Key>, value: LifetimeTracked<Value>)]
  ) {
    typealias K = LifetimeTracked<Key>
    typealias V = LifetimeTracked<Value>

    let ref: [(key: K, value: V)] = fixture.itemsInIterationOrder.map { item in
      let key = keyTransform(item)
      let value = valueTransform(item)
      return (key: self.instance(for: key), value: self.instance(for: value))
    }
    let ref2: [(key: K, value: V)] = fixture.itemsInInsertionOrder.map { item in
      let key = keyTransform(item)
      let value = valueTransform(item)
      return (key: self.instance(for: key), value: self.instance(for: value))
    }
    return (TreeDictionary(uniqueKeysWithValues: ref2), ref)
  }

  func shareableDictionary<Value>(
    for fixture: Fixture,
    valueTransform: (Int) -> Value
  ) -> (
    map: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Value>>,
    ref: [(key: LifetimeTracked<RawCollider>, value: LifetimeTracked<Value>)]
  ) {
    shareableDictionary(
      for: fixture,
      keyTransform: { $0 },
      valueTransform: { valueTransform($0.identity) })
  }

  func shareableDictionary(
    for fixture: Fixture
  ) -> (
    map: TreeDictionary<LifetimeTracked<RawCollider>, LifetimeTracked<Int>>,
    ref: [(key: LifetimeTracked<RawCollider>, value: LifetimeTracked<Int>)]
  ) {
    shareableDictionary(for: fixture) { $0 + 1000 }
  }
}
