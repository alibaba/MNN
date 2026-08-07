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
@_spi(Testing) import Collections
#else
@_spi(Testing) import OrderedCollections
import _CollectionsTestSupport
#endif

class OrderedDictionaryElementsTests: CollectionTestCase {
  func test_elements_getter() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        expectEqualElements(d.elements, reference)
      }
    }
  }

  func test_elements_modify() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)

        var d2 = OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>>()

        swap(&d.elements, &d2.elements)

        expectEqualElements(d, [])
        expectEqualElements(d2, reference)
      }
    }
  }

  func test_keys_values() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        let keys = reference.map { $0.key }
        var values = reference.map { $0.value }
        expectEqualElements(d.elements.keys, keys)
        expectEqualElements(d.elements.values, values)

        values.reverse()
        d.elements.values.reverse()

        for i in 0 ..< count { reference[i].value = values[i] }
        expectEqualElements(d.elements.values, values)
        expectEqualElements(d.elements, reference)
        expectEqualElements(d, reference)
      }
    }
  }

  func test_index_forKey() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          expectEqual(d.elements.index(forKey: reference[offset].key), offset)
        }
        expectNil(d.elements.index(forKey: tracker.instance(for: -1)))
        expectNil(d.elements.index(forKey: tracker.instance(for: count)))
      }
    }
  }

  func test_RandomAccessCollection() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        checkBidirectionalCollection(
          d.elements, expectedContents: reference,
          by: { $0 == $1 })
      }
    }
  }

  func test_CustomStringConvertible() {
    let a: OrderedDictionary<Int, Int> = [:]
    expectEqual(a.elements.description, "[:]")

    let b: OrderedDictionary<Int, Int> = [0: 1]
    expectEqual(b.elements.description, "[0: 1]")

    let c: OrderedDictionary<Int, Int> = [0: 1, 2: 3, 4: 5]
    expectEqual(c.elements.description, "[0: 1, 2: 3, 4: 5]")
  }

  func test_CustomDebugStringConvertible() {
    let a: OrderedDictionary<Int, Int> = [:]
    expectEqual(a.elements.debugDescription, "[:]")

    let b: OrderedDictionary<Int, Int> = [0: 1]
    expectEqual(b.elements.debugDescription, "[0: 1]")

    let c: OrderedDictionary<Int, Int> = [0: 1, 2: 3, 4: 5]
    expectEqual(c.elements.debugDescription, "[0: 1, 2: 3, 4: 5]")
  }

  func test_SubSequence_descriptions() {
    let d: OrderedDictionary = [
      "a": 1,
      "b": 2
    ]

    let s = d.elements[0 ..< 1]

    expectEqual(s.description, #"["a": 1]"#)
    expectEqual(s.debugDescription, #"["a": 1]"#)
  }

  func test_customReflectable() {
    do {
      let d: OrderedDictionary<Int, Int> = [1: 2, 3: 4, 5: 6]
      let mirror = Mirror(reflecting: d.elements)
      expectEqual(mirror.displayStyle, .collection)
      expectNil(mirror.superclassMirror)
      expectTrue(mirror.children.compactMap { $0.label }.isEmpty) // No label
      expectEqualElements(
        mirror.children.compactMap { $0.value as? (key: Int, value: Int) },
        d.map { $0 })
    }
  }

  func test_Equatable_Hashable() {
    let samples: [[OrderedDictionary<Int, Int>]] = [
      [[:], [:]],
      [[1: 100], [1: 100]],
      [[2: 200], [2: 200]],
      [[3: 300], [3: 300]],
      [[100: 1], [100: 1]],
      [[1: 1], [1: 1]],
      [[100: 100], [100: 100]],
      [[1: 100, 2: 200], [1: 100, 2: 200]],
      [[2: 200, 1: 100], [2: 200, 1: 100]],
      [[1: 100, 2: 200, 3: 300], [1: 100, 2: 200, 3: 300]],
      [[2: 200, 1: 100, 3: 300], [2: 200, 1: 100, 3: 300]],
      [[3: 300, 2: 200, 1: 100], [3: 300, 2: 200, 1: 100]],
      [[3: 300, 1: 100, 2: 200], [3: 300, 1: 100, 2: 200]]
    ]
    checkHashable(equivalenceClasses: samples.map { $0.map { $0.elements }})
  }

  func test_swapAt() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("i", in: 0 ..< count) { i in
        withEvery("j", in: 0 ..< count) { j in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
              reference.swapAt(i, j)
              withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
                d.elements.swapAt(i, j)
                expectEquivalentElements(
                  d, reference,
                  by: { $0.key == $1.key && $0.value == $1.value })
                expectEqual(d[reference[i].key], reference[i].value)
                expectEqual(d[reference[j].key], reference[j].value)
              }
            }
          }
        }
      }
    }
  }

  func test_partition() {
    withEvery("seed", in: 0 ..< 10) { seed in
      withEvery("count", in: 0 ..< 30) { count in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var rng = RepeatableRandomNumberGenerator(seed: seed)
            var (d, reference) = tracker.orderedDictionary(
              keys: (0 ..< count).shuffled(using: &rng))
            let expectedPivot = reference.partition { $0.key.payload < count / 2 }
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let actualPivot = d.elements.partition { $0.key.payload < count / 2 }
              expectEqual(actualPivot, expectedPivot)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_sort() {
    withEvery("seed", in: 0 ..< 10) { seed in
      withEvery("count", in: 0 ..< 30) { count in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var rng = RepeatableRandomNumberGenerator(seed: seed)
            var (d, reference) = tracker.orderedDictionary(
              keys: (0 ..< count).shuffled(using: &rng))
            reference.sort(by: { $0.key < $1.key })
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.sort()
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_sort_by() {
    withEvery("seed", in: 0 ..< 10) { seed in
      withEvery("count", in: 0 ..< 30) { count in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var rng = RepeatableRandomNumberGenerator(seed: seed)
            var (d, reference) = tracker.orderedDictionary(
              keys: (0 ..< count).shuffled(using: &rng))
            reference.sort(by: { $0.key > $1.key })
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.sort(by: { $0.key > $1.key })
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_shuffle() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withEvery("seed", in: 0 ..< 10) { seed in
          var d = OrderedDictionary(
            uniqueKeys: 0 ..< count,
            values: 100 ..< 100 + count)
          var items = (0 ..< count).map { (key: $0, value: 100 + $0) }
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            expectEqualElements(d.elements, items)

            var rng1 = RepeatableRandomNumberGenerator(seed: seed)
            items.shuffle(using: &rng1)

            var rng2 = RepeatableRandomNumberGenerator(seed: seed)
            d.elements.shuffle(using: &rng2)

            items.sort(by: { $0.key < $1.key })
            d.elements.sort()
            expectEqualElements(d, items)
          }
        }
      }
      if count >= 2 {
        // Check that shuffling with the system RNG does permute the elements.
        var d = OrderedDictionary(
          uniqueKeys: 0 ..< count,
          values: 100 ..< 100 + count)
        let original = d
        var success = false
        for _ in 0 ..< 1000 {
          d.elements.shuffle()
          if !d.elementsEqual(
            original,
            by: { $0.key == $1.key && $0.value == $1.value}
          ) {
            success = true
            break
          }
        }
        expectTrue(success)
      }
    }
  }

  func test_reverse() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            reference.reverse()
            d.elements.reverse()
            expectEqualElements(d, reference)
          }
        }
      }
    }
  }

  func test_removeAll() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, _) = tracker.orderedDictionary(keys: 0 ..< count)
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            d.elements.removeAll()
            expectEqual(d.keys.__unstable.scale, 0)
            expectEqualElements(d, [])
          }
        }
      }
    }
  }

  func test_removeAll_keepCapacity() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, _) = tracker.orderedDictionary(keys: 0 ..< count)
          let origScale = d.keys.__unstable.scale
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            d.elements.removeAll(keepingCapacity: true)
            expectEqual(d.keys.__unstable.scale, origScale)
            expectEqualElements(d, [])
          }
        }
      }
    }
  }

  func test_remove_at() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let actual = d.elements.remove(at: offset)
              let expected = reference.remove(at: offset)
              expectEqual(actual.key, expected.key)
              expectEqual(actual.value, expected.value)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_removeSubrange() {
    withEvery("count", in: 0 ..< 30) { count in
      withEveryRange("range", in: 0 ..< count) { range in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.removeSubrange(range)
              reference.removeSubrange(range)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_removeSubrange_rangeExpression() {
    let d = OrderedDictionary(uniqueKeys: 0 ..< 30, values: 100 ..< 130)
    let item = (0 ..< 30).map { (key: $0, value: 100 + $0) }

    var d1 = d
    d1.elements.removeSubrange(...10)
    expectEqualElements(d1, item[11...])

    var d2 = d
    d2.elements.removeSubrange(..<10)
    expectEqualElements(d2, item[10...])

    var d3 = d
    d3.elements.removeSubrange(10...)
    expectEqualElements(d3, item[0 ..< 10])
  }

  func test_removeLast() {
    withEvery("isShared", in: [false, true]) { isShared in
      withLifetimeTracking { tracker in
        var (d, reference) = tracker.orderedDictionary(keys: 0 ..< 30)
        withEvery("i", in: 0 ..< d.count) { i in
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            let actual = d.elements.removeLast()
            let expected = reference.removeLast()
            expectEqual(actual.key, expected.key)
            expectEqual(actual.value, expected.value)
            expectEqualElements(d, reference)
          }
        }
      }
    }
  }

  func test_removeFirst() {
    withEvery("isShared", in: [false, true]) { isShared in
      withLifetimeTracking { tracker in
        var (d, reference) = tracker.orderedDictionary(keys: 0 ..< 30)
        withEvery("i", in: 0 ..< d.count) { i in
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            let actual = d.elements.removeFirst()
            let expected = reference.removeFirst()
            expectEqual(actual.key, expected.key)
            expectEqual(actual.value, expected.value)
            expectEqualElements(d, reference)
          }
        }
      }
    }
  }

  func test_removeLast_n() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("suffix", in: 0 ..< count) { suffix in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.removeLast(suffix)
              reference.removeLast(suffix)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_removeFirst_n() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("prefix", in: 0 ..< count) { prefix in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.removeFirst(prefix)
              reference.removeFirst(prefix)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_removeAll_where() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("n", in: [2, 3, 4]) { n in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.elements.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
              reference.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_slice_keys() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        withEveryRange("range", in: 0 ..< count) { range in
          expectEqual(d.elements[range].keys, d.keys[range])
          expectEqualElements(d.elements[range].keys, reference[range].map { $0.key })
        }
      }
    }
  }

  func test_slice_values() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        withEveryRange("range", in: 0 ..< count) { range in
          expectEqualElements(d.elements[range].values, d.values[range])
          expectEqualElements(d.elements[range].values, reference[range].map { $0.value })
        }
      }
    }
  }

  func test_slice_index_forKey() {
    withEvery("count", in: 0 ..< 30) { count in
      withEveryRange("range", in: 0 ..< count) { range in
        withLifetimeTracking { tracker in
          let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          withEvery("offset", in: 0 ..< count) { offset in
            let actual = d.elements[range].index(forKey: reference[offset].key)
            let expected = range.contains(offset) ? offset : nil
            expectEqual(actual, expected)
          }
          expectNil(d.elements[range].index(forKey: tracker.instance(for: -1)))
          expectNil(d.elements[range].index(forKey: tracker.instance(for: count)))
        }
      }
    }
  }
}
