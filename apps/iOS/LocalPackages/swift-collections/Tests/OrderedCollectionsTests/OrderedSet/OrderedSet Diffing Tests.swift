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

import XCTest
#if COLLECTIONS_SINGLE_MODULE
import Collections
#else
import OrderedCollections
import _CollectionsTestSupport
#endif

class MeasuringHashable: Hashable {
  static var equalityChecks = 0
  static func == (lhs: MeasuringHashable, rhs: MeasuringHashable) -> Bool {
    MeasuringHashable.equalityChecks += 1
    return lhs._inner == rhs._inner
  }

  static var hashChecks = 0
  func hash(into hasher: inout Hasher) {
    MeasuringHashable.hashChecks += 1
    _inner.hash(into: &hasher)
  }

  let _inner: AnyHashable

  init<T: Hashable>(_ wrapped: T) {
    _inner = AnyHashable(wrapped)
  }
}

@available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *)
class OrderedSetDiffingTests: CollectionTestCase {

  func _validatePerformance<T: Hashable>(from a: OrderedSet<T>, to b: OrderedSet<T>) {
    MeasuringHashable.equalityChecks = 0
    MeasuringHashable.hashChecks = 0

    let _ = OrderedSet(a.map({MeasuringHashable($0)})).difference(from: OrderedSet(b.map({MeasuringHashable($0)})))
    let n = a.count + b.count

    /* Expect linear performance, which we can fence in testing as
     * "less than nlogn" since diffing generally tends to be at least n**2
     */
    expectLessThan(MeasuringHashable.equalityChecks, n * n.bitWidth)
    expectLessThan(MeasuringHashable.hashChecks, n * n.bitWidth)
  }

  func _validate<T: Hashable>(from a: Array<T>, to b: Array<T>, mutations: Int? = nil) {
    _validate(from: OrderedSet(a), to: OrderedSet(b), mutations: mutations)
  }

  func _validate<T: Hashable>(from a: OrderedSet<T>, to b: OrderedSet<T>, mutations: Int? = nil) {
    _validatePerformance(from: a, to: b)
    _validatePerformance(from: b, to: a)

    let d = b.difference(from: a)
    let e = a.difference(from: b)

    if let mutations = mutations {
      if mutations >= 0 {
        expectEqual(d.count, mutations)
        expectEqual(d.count, e.count)
      }
    } else {
      expectEqual(d.count, e.count)
    }
    expectEqual(a.applying(d), b)
    expectEqual(b.applying(d.inverse()), a)
    expectEqual(a.applying(e.inverse()), b)
    expectEqual(b.applying(e), a)
  }

  func test_equal() {
    let a = [1, 2, 3, 4, 5]
    _validate(from: a, to: a, mutations: 0)
  }

  func test_moveTrap() {
    let a = [1, 2, 3]
    let b = [3, 2]
    _validate(from: a, to: b, mutations: 3)
  }

  func test_minimalish() {
    let a = ["g", "1", "2",      "w", "o", "a", "e", "t", "4",               "8"]
    let b = [     "1", "2", "3",                "e",      "4", "5", "6", "7"]
    _validate(from: a, to: b, mutations: 10)
  }

  func test_arrowDefeatingMinimalDiff() {
    // Arrow diff only finds one match ("n") instead of two ("o", "c")
    let a = [               "o", "c", "n"]
    let b = ["n", "d", "y", "o", "c"]
    _validate(from: a, to: b, mutations: 4)
  }

  func test_disparate() {
    let a = OrderedSet(0..<1000)
    let b = OrderedSet(1000..<2000)
    _validate(from: a, to: b, mutations: 2000)
  }

  func test_reversed() {
    let a = OrderedSet(0..<1000)
    let b = OrderedSet((0..<1000).reversed())
    _validate(from: a, to: b, mutations: 1998)
  }

  func test_fuzzerDiscovered00() {
    //                  X     X        X
    let a = [3, 5, 8, 0, 7, 9]
    let b = [8, 5, 9, 7, 0, 6]
    //                     X        X  X
    _validate(from: a, to: b)
  }

  /* Good example of greedy match seeking resulting in a non-minimal diff.
   * Luckily, the API contract doesn't promise the shortest edit script, just an
   * accurate one.
   */
  func test_fuzzerDiscovered01() {
    // b → a X  X  X  X  X        X
    // a → b X  X  X  X  X  X     X
    let a = [3, 7, 2, 5, 9, 1, 8, 4]
    let b = [4, 2, 1, 7, 8, 6]
    // a → b X  X  X  X     X
    // b → a X  X     X     X
    _validate(from: a, to: b, mutations: -1)
  }

  func test_fuzz() {
    for _ in 0..<1000 {
      let a = (0..<10).map { _ in Int.random(in: 0..<10) }
      let b = (0..<10).map { _ in Int.random(in: 0..<10) }
      _validate(from: OrderedSet(a), to: OrderedSet(b), mutations: -1)
    }
  }
}
