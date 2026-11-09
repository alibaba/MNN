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

extension TreeSet: SetAPIExtras {}

class TreeSetTests: CollectionTestCase {
  func test_init_empty() {
    let set = TreeSet<Int>()
    expectEqual(set.count, 0)
    expectTrue(set.isEmpty)
    expectEqualElements(set, [])
  }

  func test_BidirectionalCollection_fixtures() {
    withEachFixture { fixture in
      withLifetimeTracking { tracker in
        let (set, ref) = tracker.shareableSet(for: fixture)
        checkCollection(set, expectedContents: ref, by: ==)
        _checkBidirectionalCollection_indexOffsetBy(
          set, expectedContents: ref, by: ==)
      }
    }
  }

  func test_BidirectionalCollection_random100() {
    let s = TreeSet<Int>(0 ..< 100)
    let ref = Array(s)
    checkCollection(s, expectedContents: ref)
    _checkBidirectionalCollection_indexOffsetBy(
      s, expectedContents: ref, by: ==)
  }

  func test_basics() {
    var set: TreeSet<HashableBox<Int>> = []

    let a1 = HashableBox(1)
    let a2 = HashableBox(1)

    var r = set.insert(a1)
    expectTrue(r.inserted)
    expectIdentical(r.memberAfterInsert, a1)
    expectIdentical(set.first, a1)
    expectTrue(set.contains(a1))
    expectTrue(set.contains(a2))

    r = set.insert(a2)
    expectFalse(r.inserted)
    expectIdentical(r.memberAfterInsert, a1)
    expectIdentical(set.first, a1)
    expectTrue(set.contains(a1))
    expectTrue(set.contains(a2))

    var old = set.update(with: a2)
    expectIdentical(old, a1)
    expectIdentical(set.first, a2)
    expectTrue(set.contains(a1))
    expectTrue(set.contains(a2))

    old = set.remove(a1)
    expectIdentical(old, a2)
    expectNil(set.first)
    expectFalse(set.contains(a1))
    expectFalse(set.contains(a2))

    old = set.update(with: a1)
    expectNil(old)
    expectIdentical(set.first, a1)
    expectTrue(set.contains(a1))
    expectTrue(set.contains(a2))
  }

  func test_descriptions() {
    let empty: TreeSet<Int> = []
    expectEqual(empty.description, "[]")
    expectEqual(empty.debugDescription, "[]")

    let a: TreeSet = ["a"]
    expectEqual(a.description, #"["a"]"#)
    expectEqual(a.debugDescription, #"["a"]"#)
  }

  func test_index_descriptions() {
    let a: TreeSet = [
      RawCollider(1, "1"),
      RawCollider(2, "21"),
      RawCollider(3, "22"),
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

  func test_index_hashing() {
    let s = TreeSet(0 ..< 100)
    checkHashable(s.indices, equalityOracle: ==)
  }

  func test_insert_fixtures() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var s: TreeSet<LifetimeTracked<RawCollider>> = []
          var ref: Set<LifetimeTracked<RawCollider>> = []
          withEvery("i", in: 0 ..< fixture.count) { i in
            withHiddenCopies(if: isShared, of: &s) { s in
              let item = fixture.itemsInInsertionOrder[i]
              let key1 = tracker.instance(for: item)
              let r = s.insert(key1)
              expectTrue(r.inserted)
              expectEqual(r.memberAfterInsert, key1)

              let key2 = tracker.instance(for: item)
              ref.insert(key2)
              expectEqualSets(s, ref)
            }
          }
        }
      }
    }
  }

  func test_remove_at() {
    withEachFixture { fixture in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          withEvery("offset", in: 0 ..< fixture.count) { offset in
            let f = tracker.shareableSet(for: fixture)
            var s = f.map
            var ref = Set(f.ref)
            withHiddenCopies(if: isShared, of: &s) { s in
              let i = s.index(s.startIndex, offsetBy: offset)
              let old = s.remove(at: i)
              expectNotNil(ref.remove(old))
              expectEqualSets(s, ref)
            }
          }
        }
      }
    }
  }

  func test_intersection_Self_basics() {
    let a = RawCollider(1, "A")
    let b = RawCollider(2, "B")
    let c = RawCollider(3, "C")

    let s0: TreeSet<RawCollider> = []
    let s1: TreeSet = [a, b]
    let s2: TreeSet = [a, c]

    expectEqualSets(s0.intersection(s0), [])
    expectEqualSets(s1.intersection(s0), [])
    expectEqualSets(s0.intersection(s1), [])

    expectEqualSets(s1.intersection(s1), [a, b])
    expectEqualSets(s1.intersection(s2), [a])
    expectEqualSets(s2.intersection(s1), [a])

    let ab = RawCollider(4, "AB")
    let ac = RawCollider(5, "AC")
    let s3: TreeSet = [ab, ac]
    let s4: TreeSet = [a, ab]
    expectEqualSets(s1.intersection(s3), [])
    expectEqualSets(s2.intersection(s3), [])
    expectEqualSets(s1.intersection(s4), [a])
    expectEqualSets(s2.intersection(s4), [a])
    expectEqualSets(s3.intersection(s1), [])
    expectEqualSets(s3.intersection(s2), [])
    expectEqualSets(s4.intersection(s1), [a])
    expectEqualSets(s4.intersection(s2), [a])

    let ad = RawCollider(6, "AD")
    let ae = RawCollider(7, "AE")
    let s5: TreeSet = [a, ab, ad, ae]
    expectEqualSets(s1.intersection(s5), [a])
    expectEqualSets(s2.intersection(s5), [a])
    expectEqualSets(s3.intersection(s5), [ab])
    expectEqualSets(s4.intersection(s5), [a, ab])

    expectEqualSets(s5.intersection(s1), [a])
    expectEqualSets(s5.intersection(s2), [a])
    expectEqualSets(s5.intersection(s3), [ab])
    expectEqualSets(s5.intersection(s4), [a, ab])

    let af1 = RawCollider(8, "AF")
    let af2 = RawCollider(9, "AF")
    let s6: TreeSet = [af1, af2]
    expectEqualSets(s1.intersection(s6), [])
    expectEqualSets(s2.intersection(s6), [])
    expectEqualSets(s3.intersection(s6), [])
    expectEqualSets(s4.intersection(s6), [])
    expectEqualSets(s5.intersection(s6), [])

    expectEqualSets(s6.intersection(s1), [])
    expectEqualSets(s6.intersection(s2), [])
    expectEqualSets(s6.intersection(s3), [])
    expectEqualSets(s6.intersection(s4), [])
    expectEqualSets(s6.intersection(s5), [])

    let af3 = RawCollider(10, "AF")
    let s7: TreeSet = [af1, af3]
    expectEqualSets(s1.intersection(s7), [])
    expectEqualSets(s2.intersection(s7), [])
    expectEqualSets(s3.intersection(s7), [])
    expectEqualSets(s4.intersection(s7), [])
    expectEqualSets(s5.intersection(s7), [])
    expectEqualSets(s6.intersection(s7), [af1])

    expectEqualSets(s7.intersection(s1), [])
    expectEqualSets(s7.intersection(s2), [])
    expectEqualSets(s7.intersection(s3), [])
    expectEqualSets(s7.intersection(s4), [])
    expectEqualSets(s7.intersection(s5), [])
    expectEqualSets(s7.intersection(s6), [af1])

    let s8: TreeSet = [a, af1]
    expectEqualSets(s1.intersection(s8), [a])
    expectEqualSets(s2.intersection(s8), [a])
    expectEqualSets(s3.intersection(s8), [])
    expectEqualSets(s4.intersection(s8), [a])
    expectEqualSets(s5.intersection(s8), [a])
    expectEqualSets(s6.intersection(s8), [af1])
    expectEqualSets(s7.intersection(s8), [af1])

    expectEqualSets(s8.intersection(s1), [a])
    expectEqualSets(s8.intersection(s2), [a])
    expectEqualSets(s8.intersection(s3), [])
    expectEqualSets(s8.intersection(s4), [a])
    expectEqualSets(s8.intersection(s5), [a])
    expectEqualSets(s8.intersection(s6), [af1])
    expectEqualSets(s8.intersection(s7), [af1])

    let afa1 = RawCollider(11, "AFA")
    let afa2 = RawCollider(12, "AFA")
    let s9: TreeSet = [afa1, afa2]
    expectEqualSets(s9.intersection(s6), [])
    expectEqualSets(s6.intersection(s9), [])
  }

  func test_isEqual_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectTrue(x.isEqualSet(to: x))
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = (u == v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isEqualSet(to: b)
        }

        expectEqual(x.isEqualSet(to: y), reference)
        expectEqual(x.isEqualSet(to: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isEqualSet(to: v), reference)
        expectEqual(x.isEqualSet(to: b), reference)
        expectEqual(x.isEqualSet(to: b + b), reference)
      }
    }
  }

  func test_isSubset_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectTrue(x.isSubset(of: x))
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.isSubset(of: v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isSubset(of: b)
        }

        expectEqual(x.isSubset(of: y), reference)
        expectEqual(x.isSubset(of: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isSubset(of: v), reference)
        expectEqual(x.isSubset(of: b), reference)
        expectEqual(x.isSubset(of: b + b), reference)
      }
    }
  }

  func test_isSuperset_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectTrue(x.isSuperset(of: x))
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.isSuperset(of: v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isSuperset(of: b)
        }

        expectEqual(x.isSuperset(of: y), reference)
        expectEqual(x.isSuperset(of: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isSuperset(of: v), reference)
        expectEqual(x.isSuperset(of: b), reference)
        expectEqual(x.isSuperset(of: b + b), reference)
      }
    }
  }

  func test_isStrictSubset_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectFalse(x.isStrictSubset(of: x))
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.isStrictSubset(of: v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isStrictSubset(of: b)
        }

        expectEqual(x.isStrictSubset(of: y), reference)
        expectEqual(x.isStrictSubset(of: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isStrictSubset(of: v), reference)
        expectEqual(x.isStrictSubset(of: b), reference)
        expectEqual(x.isStrictSubset(of: b + b), reference)
      }
    }
  }

  func test_isStrictSuperset_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectFalse(x.isStrictSuperset(of: x))
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.isStrictSuperset(of: v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isStrictSuperset(of: b)
        }

        expectEqual(x.isStrictSuperset(of: y), reference)
        expectEqual(x.isStrictSuperset(of: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isStrictSuperset(of: v), reference)
        expectEqual(x.isStrictSuperset(of: b), reference)
        expectEqual(x.isStrictSuperset(of: b + b), reference)
      }
    }
  }

  func test_isDisjoint_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectEqual(x.isDisjoint(with: x), x.isEmpty)
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })
        let reference = u.isDisjoint(with: v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> Bool
        where S.Element == RawCollider {
          a.isDisjoint(with: b)
        }

        expectEqual(x.isDisjoint(with: y), reference)
        expectEqual(x.isDisjoint(with: z.keys), reference)
        expectEqual(checkSequence(x, y), reference)
        expectEqual(x.isDisjoint(with: v), reference)
        expectEqual(x.isDisjoint(with: b), reference)
        expectEqual(x.isDisjoint(with: b + b), reference)
      }
    }
  }

  func test_intersection_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectEqualSets(x.intersection(x), u)
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.intersection(v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> TreeSet<RawCollider>
        where S.Element == RawCollider {
          a.intersection(b)
        }

        expectEqualSets(x.intersection(y), reference)
        expectEqualSets(x.intersection(z.keys), reference)
        expectEqualSets(checkSequence(x, y), reference)
        expectEqualSets(x.intersection(v), reference)
        expectEqualSets(x.intersection(b), reference)
        expectEqualSets(x.intersection(b + b), reference)
      }
    }
  }

  func test_subtracting_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectEqualSets(x.subtracting(x), [])
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.subtracting(v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> TreeSet<RawCollider>
        where S.Element == RawCollider {
          a.subtracting(b)
        }

        expectEqualSets(x.subtracting(y), reference)
        expectEqualSets(x.subtracting(z.keys), reference)
        expectEqualSets(checkSequence(x, y), reference)
        expectEqualSets(x.subtracting(v), reference)
        expectEqualSets(x.subtracting(b), reference)
        expectEqualSets(x.subtracting(b + b), reference)
      }
    }
  }

  func test_filter_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      withEverySubset("b", of: a) { b in
        let v = Set(b)
        expectEqualSets(x.filter { v.contains($0) }, v)
      }
    }
  }

  func test_removeAll_where_exhaustive() {
    withEvery("isShared", in: [false, true]) { isShared in
      withEverySubset("a", of: testItems) { a in
        withEverySubset("b", of: a) { b in
          var x = TreeSet(a)
          let v = Set(b)
          withHiddenCopies(if: isShared, of: &x) { x in
            x.removeAll { !v.contains($0) }
            expectEqualSets(x, v)
          }
        }
      }
    }
  }

  func test_union_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectEqualSets(x.union(x), u)
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.union(v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> TreeSet<RawCollider>
        where S.Element == RawCollider {
          a.union(b)
        }

        expectEqualSets(x.union(y), reference)
        expectEqualSets(x.union(z.keys), reference)
        expectEqualSets(checkSequence(x, y), reference)
        expectEqualSets(x.union(v), reference)
        expectEqualSets(x.union(b), reference)
        expectEqualSets(x.union(b + b), reference)
      }
    }
  }

  func test_symmetricDifference_exhaustive() {
    withEverySubset("a", of: testItems) { a in
      let x = TreeSet(a)
      let u = Set(a)
      expectEqualSets(x.symmetricDifference(x), [])
      withEverySubset("b", of: testItems) { b in
        let y = TreeSet(b)
        let v = Set(b)
        let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

        let reference = u.symmetricDifference(v)

        func checkSequence<S: Sequence>(
          _ a: TreeSet<RawCollider>,
          _ b: S
        ) -> TreeSet<RawCollider>
        where S.Element == RawCollider {
          a.symmetricDifference(b)
        }

        expectEqualSets(x.symmetricDifference(y), reference)
        expectEqualSets(x.symmetricDifference(z.keys), reference)
        expectEqualSets(checkSequence(x, y), reference)
        expectEqualSets(x.symmetricDifference(v), reference)
        expectEqualSets(x.symmetricDifference(b), reference)
        expectEqualSets(x.symmetricDifference(b + b), reference)
      }
    }
  }

  func test_mutating_binary_set_operations() {
    let a = [1, 2, 3, 4]
    let b = [0, 2, 4, 6]

    let x = TreeSet(a)
    let y = TreeSet(b)
    let u = Set(a)
    let v = Set(b)
    let z = TreeDictionary(uniqueKeysWithValues: b.map { ($0, $0) })

    func check(
      _ reference: Set<Int>,
      _ body: (inout TreeSet<Int>) -> Void,
      file: StaticString = #file,
      line: UInt = #line
    ) {
      var set = x
      body(&set)
      expectEqualSets(set, reference, file: file, line: line)
    }

    do {
      let reference = u.intersection(v)
      check(reference) { $0.formIntersection(y) }
      check(reference) { $0.formIntersection(z.keys) }
      check(reference) { $0.formIntersection(b) }
      check(reference) { $0.formIntersection(b + b) }
    }

    do {
      let reference = u.union(v)
      check(reference) { $0.formUnion(y) }
      check(reference) { $0.formUnion(z.keys) }
      check(reference) { $0.formUnion(b) }
      check(reference) { $0.formUnion(b + b) }
    }

    do {
      let reference = u.symmetricDifference(v)
      check(reference) { $0.formSymmetricDifference(y) }
      check(reference) { $0.formSymmetricDifference(z.keys) }
      check(reference) { $0.formSymmetricDifference(b) }
      check(reference) { $0.formSymmetricDifference(b + b) }
    }

    do {
      let reference = u.subtracting(v)
      check(reference) { $0.subtract(y) }
      check(reference) { $0.subtract(z.keys) }
      check(reference) { $0.subtract(b) }
      check(reference) { $0.subtract(b + b) }
    }
  }

  func test_update_at() {
    withEverySubset("a", of: testItems) { a in
      withEvery("offset", in: 0 ..< a.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var x = TreeSet(tracker.instances(for: a))
            let i = x.firstIndex { $0.payload == a[offset] }!
            let replacement = tracker.instance(for: a[offset])
            withHiddenCopies(if: isShared, of: &x) { x in
              let old = x.update(replacement, at: i)
              expectEqual(old, replacement)
              expectNotIdentical(old, replacement)
            }
          }
        }
      }
    }
  }

  func test_Hashable() {
    let classes: [[TreeSet<String>]] = [
      [
        []
      ],
      [
        ["a"]
      ],
      [
        ["b"]
      ],
      [
        ["c"]
      ],
      [
        ["d"]
      ],
      [
        ["e"]
      ],
      [
        ["f"], ["f"],
      ],
      [
        ["g"], ["g"],
      ],
      [
        ["h"], ["h"],
      ],
      [
        ["i"], ["i"],
      ],
      [
        ["j"], ["j"],
      ],
      [
        ["a", "b"], ["b", "a"],
      ],
      [
        ["a", "d"], ["d", "a"],
      ],
      [
        ["a", "b", "c"], ["a", "c", "b"],
        ["b", "a", "c"], ["b", "c", "a"],
        ["c", "a", "b"], ["c", "b", "a"],
      ],
      [
        ["a", "d", "e"], ["a", "e", "d"],
        ["d", "a", "e"], ["d", "e", "a"],
        ["e", "a", "d"], ["e", "d", "a"],
      ],
    ]
    checkHashable(equivalenceClasses: classes)
  }

  func test_Codable() throws {
    let s1: TreeSet<Int> = []
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalEncoder.encode(s1), v1)

    let s2: TreeSet<Int> = [3]
    let v2: MinimalEncoder.Value = .array([.int(3)])
    expectEqual(try MinimalEncoder.encode(s2), v2)

    let s3: TreeSet<Int> = [0, 1, 2, 3]
    let v3: MinimalEncoder.Value = .array(s3.map { .int($0) })
    expectEqual(try MinimalEncoder.encode(s3), v3)

    let s4 = TreeSet<Int>(0 ..< 100)
    let v4: MinimalEncoder.Value = .array(s4.map { .int($0) })
    expectEqual(try MinimalEncoder.encode(s4), v4)
  }

  func test_Decodable() throws {
    let s1: TreeSet<Int> = []
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalDecoder.decode(v1, as: TreeSet<Int>.self), s1)

    let s2: TreeSet<Int> = [3]
    let v2: MinimalEncoder.Value = .array([.int(3)])
    expectEqual(try MinimalDecoder.decode(v2, as: TreeSet<Int>.self), s2)

    let s3: TreeSet<Int> = [0, 1, 2, 3]
    let v3: MinimalEncoder.Value = .array([.int(0), .int(1), .int(2), .int(3)])
    expectEqual(try MinimalDecoder.decode(v3, as: TreeSet<Int>.self), s3)

    let s4 = TreeSet<Int>(0 ..< 100)
    let v4: MinimalEncoder.Value = .array((0 ..< 100).map { .int($0) })
    expectEqual(try MinimalDecoder.decode(v4, as: TreeSet<Int>.self), s4)

    let v5: MinimalEncoder.Value = .array([.int(0), .int(1), .int(0)])
    expectThrows(
      try MinimalDecoder.decode(v5, as: TreeSet<Int>.self)
    ) { error in
      expectNotNil(error as? DecodingError) { error in
        guard case .dataCorrupted(let context) = error else {
          expectFailure("Unexpected error \(error)")
          return
        }
        expectEqual(context.debugDescription,
                    "Decoded elements aren't unique (first duplicate at offset 2)")
      }
    }

    let v6: MinimalEncoder.Value = .array([.int16(42)])
    expectThrows(
      try MinimalDecoder.decode(v6, as: TreeSet<Int>.self)
    ) { error in
      expectNotNil(error as? DecodingError) { error in
        guard case .typeMismatch(_, _) = error else {
          expectFailure("Unexpected error \(error)")
          return
        }
      }
    }
  }

  func test_CustomReflectable() {
    let s: TreeSet = [0, 1, 2, 3]
    let mirror = Mirror(reflecting: s)
    expectEqual(mirror.displayStyle, .set)
    expectNil(mirror.superclassMirror)
    expectTrue(mirror.children.compactMap { $0.label }.isEmpty)
    expectEqualElements(mirror.children.map { $0.value as? Int }, s.map { $0 })
  }
}
