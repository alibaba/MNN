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

import XCTest
#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import _CollectionsTestSupport
import BitCollections
import OrderedCollections
#endif

extension BitSet.Counted: SetAPIExtras {
  public mutating func update(_ member: Int, at index: Index) -> Int {
    fatalError("Not this one though")
  }
}

extension BitSet.Counted: SortedCollectionAPIChecker {}

final class BitSetCountedTests: CollectionTestCase {
  func test_union() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 3 ..< 7
    let exp = 1 ..< 7
    expectEqualElements(a.union(BitSet.Counted(b)), exp)
    expectEqualElements(a.union(BitSet(b)), exp)
    expectEqualElements(a.union(b), exp)
    expectEqualElements(a.union(Set(b)), exp)
  }

  func test_intersection() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 3 ..< 7
    let exp = 3 ..< 5
    expectEqualElements(a.intersection(BitSet.Counted(b)), exp)
    expectEqualElements(a.intersection(BitSet(b)), exp)
    expectEqualElements(a.intersection(b), exp)
    expectEqualElements(a.intersection(Set(b)), exp)
  }

  func test_symmetricDifference() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 3 ..< 7
    let exp = [1, 2, 5, 6]
    expectEqualElements(a.symmetricDifference(BitSet.Counted(b)), exp)
    expectEqualElements(a.symmetricDifference(BitSet(b)), exp)
    expectEqualElements(a.symmetricDifference(b), exp)
    expectEqualElements(a.symmetricDifference(Set(b)), exp)
  }

  func test_subtracting() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 3 ..< 7
    let exp = [1, 2]
    expectEqualElements(a.subtracting(BitSet.Counted(b)), exp)
    expectEqualElements(a.subtracting(BitSet(b)), exp)
    expectEqualElements(a.subtracting(b), exp)
    expectEqualElements(a.subtracting(Set(b)), exp)
  }

  func test_formUnion() {
    func check<S: Sequence>(
      _ expected: S,
      _ body: (inout BitSet.Counted) -> Void,
      file: StaticString = #file,
      line: UInt = #line
    ) where S.Element == Int {
      var a: BitSet.Counted = [1, 2, 3, 4]

      body(&a)
      expectEqualElements(a, expected, file: file, line: line)
    }

    let b = 3 ..< 7
    let exp = 1 ..< 7
    check(exp) { $0.formUnion(BitSet.Counted(b)) }
    check(exp) { $0.formUnion(BitSet(b)) }
    check(exp) { $0.formUnion(b) }
    check(exp) { $0.formUnion(Set(b)) }
  }

  func test_formIntersection() {
    func check<S: Sequence>(
      _ expected: S,
      _ body: (inout BitSet.Counted) -> Void,
      file: StaticString = #file,
      line: UInt = #line
    ) where S.Element == Int {
      var a: BitSet.Counted = [1, 2, 3, 4]

      body(&a)
      expectEqualElements(a, expected, file: file, line: line)
    }

    let b = 3 ..< 7
    let exp = 3 ..< 5
    check(exp) { $0.formIntersection(BitSet.Counted(b)) }
    check(exp) { $0.formIntersection(BitSet(b)) }
    check(exp) { $0.formIntersection(b) }
    check(exp) { $0.formIntersection(Set(b)) }
  }

  func test_formSymmetricDifference() {
    func check<S: Sequence>(
      _ expected: S,
      _ body: (inout BitSet.Counted) -> Void,
      file: StaticString = #file,
      line: UInt = #line
    ) where S.Element == Int {
      var a: BitSet.Counted = [1, 2, 3, 4]

      body(&a)
      expectEqualElements(a, expected, file: file, line: line)
    }

    let b = 3 ..< 7
    let exp = [1, 2, 5, 6]
    check(exp) { $0.formSymmetricDifference(BitSet.Counted(b)) }
    check(exp) { $0.formSymmetricDifference(BitSet(b)) }
    check(exp) { $0.formSymmetricDifference(b) }
    check(exp) { $0.formSymmetricDifference(Set(b)) }
  }

  func test_subtract() {
    func check<S: Sequence>(
      _ expected: S,
      _ body: (inout BitSet.Counted) -> Void,
      file: StaticString = #file,
      line: UInt = #line
    ) where S.Element == Int {
      var a: BitSet.Counted = [1, 2, 3, 4]

      body(&a)
      expectEqualElements(a, expected, file: file, line: line)
    }

    let b = 3 ..< 7
    let exp = [1, 2]
    check(exp) { $0.subtract(BitSet.Counted(b)) }
    check(exp) { $0.subtract(BitSet(b)) }
    check(exp) { $0.subtract(b) }
    check(exp) { $0.subtract(Set(b)) }
  }

  func test_isEqual() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 1 ..< 5
    expectTrue(a.isEqualSet(to: BitSet.Counted(b)))
    expectTrue(a.isEqualSet(to: BitSet(b)))
    expectTrue(a.isEqualSet(to: b))
    expectTrue(a.isEqualSet(to: Set(b)))

    let c = 2 ..< 7
    expectFalse(a.isEqualSet(to: BitSet.Counted(c)))
    expectFalse(a.isEqualSet(to: BitSet(c)))
    expectFalse(a.isEqualSet(to: c))
    expectFalse(a.isEqualSet(to: Set(c)))
  }

  func test_isSubset() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 1 ..< 6
    expectTrue(a.isSubset(of: BitSet.Counted(b)))
    expectTrue(a.isSubset(of: BitSet(b)))
    expectTrue(a.isSubset(of: b))
    expectTrue(a.isSubset(of: Set(b)))

    let c = 2 ..< 4
    expectFalse(a.isSubset(of: BitSet.Counted(c)))
    expectFalse(a.isSubset(of: BitSet(c)))
    expectFalse(a.isSubset(of: c))
    expectFalse(a.isSubset(of: Set(c)))
  }

  func test_isSuperset() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 2 ..< 5
    expectTrue(a.isSuperset(of: BitSet.Counted(b)))
    expectTrue(a.isSuperset(of: BitSet(b)))
    expectTrue(a.isSuperset(of: b))
    expectTrue(a.isSuperset(of: Set(b)))

    let c = 2 ..< 8
    expectFalse(a.isSuperset(of: BitSet.Counted(c)))
    expectFalse(a.isSuperset(of: BitSet(c)))
    expectFalse(a.isSuperset(of: c))
    expectFalse(a.isSuperset(of: Set(c)))
  }

  func test_isStrictSubset() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 1 ..< 6
    expectTrue(a.isStrictSubset(of: BitSet.Counted(b)))
    expectTrue(a.isStrictSubset(of: BitSet(b)))
    expectTrue(a.isStrictSubset(of: b))
    expectTrue(a.isStrictSubset(of: Set(b)))

    let c = 1 ..< 5
    expectFalse(a.isStrictSubset(of: BitSet.Counted(c)))
    expectFalse(a.isStrictSubset(of: BitSet(c)))
    expectFalse(a.isStrictSubset(of: c))
    expectFalse(a.isStrictSubset(of: Set(c)))
  }

  func test_isStrictSuperset() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 2 ..< 5
    expectTrue(a.isStrictSuperset(of: BitSet.Counted(b)))
    expectTrue(a.isStrictSuperset(of: BitSet(b)))
    expectTrue(a.isStrictSuperset(of: b))
    expectTrue(a.isStrictSuperset(of: Set(b)))

    let c = 1 ..< 5
    expectFalse(a.isStrictSuperset(of: BitSet.Counted(c)))
    expectFalse(a.isStrictSuperset(of: BitSet(c)))
    expectFalse(a.isStrictSuperset(of: c))
    expectFalse(a.isStrictSuperset(of: Set(c)))
  }

  func test_isDisjoint() {
    let a: BitSet.Counted = [1, 2, 3, 4]

    let b = 5 ..< 10
    expectTrue(a.isDisjoint(with: BitSet.Counted(b)))
    expectTrue(a.isDisjoint(with: BitSet(b)))
    expectTrue(a.isDisjoint(with: b))
    expectTrue(a.isDisjoint(with: Set(b)))

    let c = 4 ..< 10
    expectFalse(a.isDisjoint(with: BitSet.Counted(c)))
    expectFalse(a.isDisjoint(with: BitSet(c)))
    expectFalse(a.isDisjoint(with: c))
    expectFalse(a.isDisjoint(with: Set(c)))
  }
}
