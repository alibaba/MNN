//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2019 - 2024 Apple Inc. and the Swift project authors
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

extension TreeDictionary {
  fileprivate func contains(_ key: Key) -> Bool {
    self[key] != nil
  }
}

final class TreeDictionarySmokeTests: CollectionTestCase {
  func testDummy() {
    let map = TreeDictionary(
      uniqueKeysWithValues: (0 ..< 100).map { ($0, 2 * $0) })
    var it = map.makeIterator()
    var seen: Set<Int> = []
    while let item = it.next() {
      if !seen.insert(item.key).inserted {
        print(item)
      }
    }
    expectEqual(seen.count, 100)
    print("---")
    for item in map {
      if seen.remove(item.key) == nil {
        print(item)
      }
    }
    expectEqual(seen.count, 0)
  }

  func testSubscriptAdd() {
    var map: TreeDictionary<Int, String> = [1: "a", 2: "b"]

    map[3] = "x"
    map[4] = "y"

    expectEqual(map.count, 4)
    expectEqual(map[1], "a")
    expectEqual(map[2], "b")
    expectEqual(map[3], "x")
    expectEqual(map[4], "y")
  }

  func testSubscriptOverwrite() {
    var map: TreeDictionary<Int, String> = [1: "a", 2: "b"]

    map[1] = "x"
    map[2] = "y"

    expectEqual(map.count, 2)
    expectEqual(map[1], "x")
    expectEqual(map[2], "y")
  }

  func testSubscriptRemove() {
    var map: TreeDictionary<Int, String> = [1: "a", 2: "b"]

    map[1] = nil
    map[2] = nil

    expectEqual(map.count, 0)
    expectEqual(map[1], nil)
    expectEqual(map[2], nil)
  }

  func testTriggerOverwrite1() {
    var map: TreeDictionary<Int, String> = [1: "a", 2: "b"]

    map.updateValue("x", forKey: 1) // triggers COW
    map.updateValue("y", forKey: 2) // triggers COW

    var res1: TreeDictionary<Int, String> = [:]
    res1.updateValue("a", forKey: 1) // in-place
    res1.updateValue("b", forKey: 2) // in-place

    var res2: TreeDictionary<Int, String> = [:]
    res2[1] = "a" // in-place
    res2[2] = "b" // in-place

    var res3: TreeDictionary<Int, String> = res2
    res3[1] = "x" // triggers COW
    res3[2] = "y" // in-place

    expectEqual(res2.count, 2)
    expectEqual(res2[1], "a")
    expectEqual(res2[2], "b")

    expectEqual(res3.count, 2)
    expectEqual(res3[1], "x")
    expectEqual(res3[2], "y")
  }

  func testTriggerOverwrite2() {
    var res1: TreeDictionary<Collider, String> = [:]
    res1.updateValue("a", forKey: Collider(10, 01)) // in-place
    res1.updateValue("a", forKey: Collider(11, 33)) // in-place
    res1.updateValue("b", forKey: Collider(20, 02)) // in-place

    res1.updateValue("x", forKey: Collider(10, 01)) // in-place
    res1.updateValue("x", forKey: Collider(11, 33)) // in-place
    res1.updateValue("y", forKey: Collider(20, 02)) // in-place

    var res2: TreeDictionary<Collider, String> = res1
    res2.updateValue("a", forKey: Collider(10, 01)) // triggers COW
    res2.updateValue("a", forKey: Collider(11, 33)) // in-place
    res2.updateValue("b", forKey: Collider(20, 02)) // in-place

    expectEqual(res1[Collider(10, 01)], "x")
    expectEqual(res1[Collider(11, 33)], "x")
    expectEqual(res1[Collider(20, 02)], "y")

    expectEqual(res2[Collider(10, 01)], "a")
    expectEqual(res2[Collider(11, 33)], "a")
    expectEqual(res2[Collider(20, 02)], "b")

  }

  func testTriggerOverwrite3() {
    let upperBound = 1_000

    // Populating `map1`
    var map1: TreeDictionary<Collider, String> = [:]
    for index in 0..<upperBound {
      map1[Collider(index)] = "+\(index)"
    }

    // Populating `map2`
    var map2: TreeDictionary<Collider, String> = map1
    for index in 0..<upperBound {
      map2[Collider(index)] = "-\(index)"
    }

    // Testing `map1` and `map2`
    expectEqual(map1.count, upperBound)
    expectEqual(map2.count, upperBound)

    for index in 0..<upperBound {
      expectEqual(map1[Collider(index)], "+\(index)")
      expectEqual(map2[Collider(index)], "-\(index)")
    }

    // Populating and testing `map3`
    var map3: TreeDictionary<Collider, String> = map2
    for index in 0..<upperBound {
      map3[Collider(index)] = nil
      expectEqual(map3.count, inferSize(map3))
    }

    expectEqual(map3.count, 0)
  }

  private func hashPair<Key: Hashable, Value: Hashable>(
    _ key: Key,
    _ value: Value
  ) -> Int {
    var hasher = Hasher()
    hasher.combine(key)
    hasher.combine(value)
    return hasher.finalize()
  }

  func testHashable() {
    let map: TreeDictionary<Int, String> = [1: "a", 2: "b"]

    let hashPair1 = hashPair(1, "a")
    let hashPair2 = hashPair(2, "b")

    var commutativeHasher = Hasher()
    commutativeHasher.combine(hashPair1 ^ hashPair2)

    let expectedHashValue = commutativeHasher.finalize()

    expectEqual(map.hashValue, expectedHashValue)

    var inoutHasher = Hasher()
    map.hash(into: &inoutHasher)

    expectEqual(inoutHasher.finalize(), expectedHashValue)
  }

  func testCollisionNodeNotEqual() {
    let map: TreeDictionary<Collider, Collider> = [:]

    var res12 = map
    res12[Collider(1, 1)] = Collider(1, 1)
    res12[Collider(2, 1)] = Collider(2, 1)

    var res13 = map
    res13[Collider(1, 1)] = Collider(1, 1)
    res13[Collider(3, 1)] = Collider(3, 1)

    var res31 = map
    res31[Collider(3, 1)] = Collider(3, 1)
    res31[Collider(1, 1)] = Collider(1, 1)

    expectEqual(res13, res31)
    expectNotEqual(res13, res12)
    expectNotEqual(res31, res12)
  }

  func testCompactionWhenDeletingFromHashCollisionNode1() {
    let map: TreeDictionary<Collider, Collider> = [:]


    var res1 = map
    res1[Collider(11, 1)] = Collider(11, 1)
    res1[Collider(12, 1)] = Collider(12, 1)

    expectTrue(res1.contains(Collider(11, 1)))
    expectTrue(res1.contains(Collider(12, 1)))

    expectEqual(res1.count, 2)
    expectEqual(TreeDictionary([
      Collider(11, 1): Collider(11, 1),
      Collider(12, 1): Collider(12, 1)
    ]), res1)


    var res2 = res1
    res2[Collider(12, 1)] = nil

    expectTrue(res2.contains(Collider(11, 1)))
    expectFalse(res2.contains(Collider(12, 1)))

    expectEqual(res2.count, 1)
    expectEqual(
      TreeDictionary([
        Collider(11, 1): Collider(11, 1)
      ]),
      res2)


    var res3 = res1
    res3[Collider(11, 1)] = nil

    expectFalse(res3.contains(Collider(11, 1)))
    expectTrue(res3.contains(Collider(12, 1)))

    expectEqual(res3.count, 1)
    expectEqual(
      TreeDictionary([Collider(12, 1): Collider(12, 1)]),
      res3)


    var resX = res1
    resX[Collider(32769)] = Collider(32769)
    resX[Collider(12, 1)] = nil

    expectTrue(resX.contains(Collider(11, 1)))
    expectFalse(resX.contains(Collider(12, 1)))
    expectTrue(resX.contains(Collider(32769)))

    expectEqual(resX.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(11, 1): Collider(11, 1),
        Collider(32769): Collider(32769)]),
      resX)


    var resY = res1
    resY[Collider(32769)] = Collider(32769)
    resY[Collider(32769)] = nil

    expectTrue(resY.contains(Collider(11, 1)))
    expectTrue(resY.contains(Collider(12, 1)))
    expectFalse(resY.contains(Collider(32769)))

    expectEqual(resY.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(11, 1): Collider(11, 1),
        Collider(12, 1): Collider(12, 1)]),
      resY)
  }

  func testCompactionWhenDeletingFromHashCollisionNode2() {
    let map: TreeDictionary<Collider, Collider> = [:]


    var res1 = map
    res1[Collider(32769_1, 32769)] = Collider(32769_1, 32769)
    res1[Collider(32769_2, 32769)] = Collider(32769_2, 32769)

    expectTrue(res1.contains(Collider(32769_1, 32769)))
    expectTrue(res1.contains(Collider(32769_2, 32769)))

    expectEqual(res1.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res1)


    var res2 = res1
    res2[Collider(1)] = Collider(1)

    expectTrue(res2.contains(Collider(1)))
    expectTrue(res2.contains(Collider(32769_1, 32769)))
    expectTrue(res2.contains(Collider(32769_2, 32769)))

    expectEqual(res2.count, 3)
    expectEqual(
      TreeDictionary([
        Collider(1): Collider(1),
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res2)


    var res3 = res2
    res3[Collider(32769_2, 32769)] = nil

    expectTrue(res3.contains(Collider(1)))
    expectTrue(res3.contains(Collider(32769_1, 32769)))

    expectEqual(res3.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(1): Collider(1),
        Collider(32769_1, 32769): Collider(32769_1, 32769)]),
      res3)
  }

  func testCompactionWhenDeletingFromHashCollisionNode3() {
    let map: TreeDictionary<Collider, Collider> = [:]


    var res1 = map
    res1[Collider(32769_1, 32769)] = Collider(32769_1, 32769)
    res1[Collider(32769_2, 32769)] = Collider(32769_2, 32769)

    expectTrue(res1.contains(Collider(32769_1, 32769)))
    expectTrue(res1.contains(Collider(32769_2, 32769)))

    expectEqual(res1.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res1)


    var res2 = res1
    res2[Collider(1)] = Collider(1)

    expectTrue(res2.contains(Collider(1)))
    expectTrue(res2.contains(Collider(32769_1, 32769)))
    expectTrue(res2.contains(Collider(32769_2, 32769)))

    expectEqual(res2.count, 3)
    expectEqual(
      TreeDictionary([
        Collider(1): Collider(1),
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res2)


    var res3 = res2
    res3[Collider(1)] = nil

    expectTrue(res3.contains(Collider(32769_1, 32769)))
    expectTrue(res3.contains(Collider(32769_2, 32769)))

    expectEqual(res3.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res3)


    expectEqual(res1, res3)
  }

  func testCompactionWhenDeletingFromHashCollisionNode4() {
    let map: TreeDictionary<Collider, Collider> = [:]


    var res1 = map
    res1[Collider(32769_1, 32769)] = Collider(32769_1, 32769)
    res1[Collider(32769_2, 32769)] = Collider(32769_2, 32769)

    expectTrue(res1.contains(Collider(32769_1, 32769)))
    expectTrue(res1.contains(Collider(32769_2, 32769)))

    expectEqual(res1.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res1)


    var res2 = res1
    res2[Collider(5)] = Collider(5)

    expectTrue(res2.contains(Collider(5)))
    expectTrue(res2.contains(Collider(32769_1, 32769)))
    expectTrue(res2.contains(Collider(32769_2, 32769)))

    expectEqual(res2.count, 3)
    expectEqual(
      TreeDictionary([
        Collider(5): Collider(5),
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res2)


    var res3 = res2
    res3[Collider(5)] = nil

    expectTrue(res3.contains(Collider(32769_1, 32769)))
    expectTrue(res3.contains(Collider(32769_2, 32769)))

    expectEqual(res3.count, 2)
    expectEqual(
      TreeDictionary([
        Collider(32769_1, 32769): Collider(32769_1, 32769),
        Collider(32769_2, 32769): Collider(32769_2, 32769)]),
      res3)


    expectEqual(res1, res3)
  }

  func testCompactionWhenDeletingFromHashCollisionNode5() {
    let map: TreeDictionary<Collider, Collider> = [:]


    var res1 = map
    res1[Collider(1)] = Collider(1)
    res1[Collider(1026)] = Collider(1026)
    res1[Collider(32770_1, 32770)] = Collider(32770_1, 32770)
    res1[Collider(32770_2, 32770)] = Collider(32770_2, 32770)

    expectTrue(res1.contains(Collider(1)))
    expectTrue(res1.contains(Collider(1026)))
    expectTrue(res1.contains(Collider(32770_1, 32770)))
    expectTrue(res1.contains(Collider(32770_2, 32770)))

    expectEqual(res1.count, 4)
    expectEqual(
      TreeDictionary([
        Collider(1): Collider(1),
        Collider(1026): Collider(1026),
        Collider(32770_1, 32770): Collider(32770_1, 32770),
        Collider(32770_2, 32770): Collider(32770_2, 32770)]),
      res1)


    var res2 = res1
    res2[Collider(1026)] = nil

    expectTrue(res2.contains(Collider(1)))
    expectFalse(res2.contains(Collider(1026)))
    expectTrue(res2.contains(Collider(32770_1, 32770)))
    expectTrue(res2.contains(Collider(32770_2, 32770)))

    expectEqual(res2.count, 3)
    expectEqual(
      TreeDictionary([
        Collider(1): Collider(1),
        Collider(32770_1, 32770): Collider(32770_1, 32770),
        Collider(32770_2, 32770): Collider(32770_2, 32770)]),
      res2)
  }

  func inferSize<Key, Value>(_ map: TreeDictionary<Key, Value>) -> Int {
    var size = 0

    for _ in map {
      size += 1
    }

    return size
  }

  func testIteratorEnumeratesAllIfCollision() {
    let upperBound = 1_000

    // '+' prefixed values
    var map1: TreeDictionary<Collider, String> = [:]
    for index in 0..<upperBound {
      map1[Collider(index, 1)] = "+\(index)"
      expectTrue(map1.contains(Collider(index, 1)))
      expectTrue(map1[Collider(index, 1)] == "+\(index)")
    }
    expectEqual(map1.count, upperBound)
    doIterate(map1)

    // '-' prefixed values
    var map2: TreeDictionary<Collider, String> = map1
    for index in 0..<upperBound {
      let key = Collider(index, 1)
      map2[key] = "-\(index)"
      expectTrue(map2.contains(key))
      expectEqual(map2[key], "-\(index)")
    }
    expectEqual(map2.count, upperBound)
    doIterate(map2)

    // empty map
    var map3: TreeDictionary<Collider, String> = map1
    for index in 0..<upperBound {
      map3[Collider(index, 1)] = nil
    }
    expectEqual(map3.count, 0)
    doIterate(map3)
  }

  func testIteratorEnumeratesAllIfNoCollision() {
    let upperBound = 1_000

    var map1: TreeDictionary<Int, String> = [:]
    for index in 0..<upperBound {
      map1[index] = "+\(index)"
    }

    var count = 0
    for _ in map1 {
      count = count + 1
    }

    expectEqual(map1.count, count)

    doIterate(map1)
  }

  @inline(never)
  func doIterate<Key: Hashable, Value>(
    _ map1: TreeDictionary<Key, Value>
  ) {
    var count = 0
    for _ in map1 {
      count = count + 1
    }
    expectEqual(map1.count, count)
  }

  func testIteratorEnumeratesAll() {
    let map1: TreeDictionary<Collider, String> = [
      Collider(11, 1): "a",
      Collider(12, 1): "a",
      Collider(32769): "b"
    ]

    var map2: TreeDictionary<Collider, String> = [:]
    for (key, value) in map1 {
      map2[key] = value
    }

    expectEqual(map1, map2)
  }

  func test_indexForKey_hashCollision() {
    let a = Collider(1, "1000")
    let b = Collider(2, "1000")
    let c = Collider(3, "1001")
    let map: TreeDictionary<Collider, String> = [
      a: "a",
      b: "b",
      c: "c",
    ]

    let indices = Array(map.indices)

    typealias Index = TreeDictionary<Collider, String>.Index
    expectEqual(map.index(forKey: a), indices[1])
    expectEqual(map.index(forKey: b), indices[2])
    expectEqual(map.index(forKey: c), indices[0])
    expectNil(map.index(forKey: Collider(4, "1000")))
  }

  func test_indexForKey() {
    let input = 0 ..< 10_000
    let d = TreeDictionary(
      uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
    for key in input {
      expectNotNil(d.index(forKey: key)) { index in
        expectNotNil(d[index]) { item in
          expectEqual(item.key, key)
          expectEqual(item.value, 2 * key)
        }
      }
    }
  }

  func test_indexForKey_exhaustIndices() {
    var map: TreeDictionary<Collider, Int> = [:]

    let range = 0 ..< 10_000

    for value in range {
      map[Collider(value)] = value
    }

    var expectedPositions = Set(map.indices)

    for expectedValue in range {
      expectNotNil(map.index(forKey: Collider(expectedValue))) { index in
        let actualValue = map[index].value
        expectEqual(expectedValue, actualValue)
        expectedPositions.remove(index)
      }
    }

    expectTrue(expectedPositions.isEmpty)
  }
}
