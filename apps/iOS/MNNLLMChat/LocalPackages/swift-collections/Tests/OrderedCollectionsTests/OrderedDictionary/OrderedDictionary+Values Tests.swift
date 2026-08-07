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
@_spi(Testing) import OrderedCollections
import _CollectionsTestSupport
#endif

class OrderedDictionaryValueTests: CollectionTestCase {
  func test_values_getter_equal() {
    let left: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    let right: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    expectEqual(left.values, left.values) // Identity fast path
    expectEqual(left.values, right.values) // Linear algorithm
  }
  
  func test_values_getter_not_equal() {
    let left: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    let right: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
      "five": 5,
    ]
    expectNotEqual(left.values, right.values)
  }
  
  func test_values_getter_equal_elements() {
    let d: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    expectEqualElements(d.values, [1, 2, 3, 4])
  }

  func test_descriptions() {
    let d: OrderedDictionary = [
      "a": 1,
      "b": 2
    ]

    expectEqual(d.values.description, "[1, 2]")
    expectEqual(d.values.debugDescription, "[1, 2]")
  }

  func test_values_RandomAccessCollection() {
    withEvery("count", in: 0 ..< 30) { count in
      let keys = 0 ..< count
      let values = keys.map { $0 + 100 }
      let d = OrderedDictionary(uniqueKeys: keys, values: values)
      checkBidirectionalCollection(d.values, expectedContents: values)
    }
  }

  func test_values_subscript_assignment() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d.values[offset] = replacement
              reference[offset].value = replacement
              expectEqualElements(d.values, reference.map { $0.value })
              expectEqual(d[reference[offset].key], reference[offset].value)
            }
          }
        }
      }
    }
  }

  func test_values_subscript_inPlaceMutation() {
    func checkMutation(
      _ item: inout LifetimeTracked<Int>,
      tracker: LifetimeTracker,
      delta: Int,
      file: StaticString = #file,
      line: UInt = #line
    ) {
      expectTrue(isKnownUniquelyReferenced(&item))
      item = tracker.instance(for: item.payload + delta)
    }

    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          // Discard reference values and recreate them from scratch to make
          // sure values are all uniquely referenced.
          for i in 0 ..< count {
            reference[i].value = tracker.instance(for: reference[i].value.payload)
          }

          checkMutation(&reference[offset].value, tracker: tracker, delta: 10)
          checkMutation(&d.values[offset], tracker: tracker, delta: 10)

          expectEqualElements(d.values, reference.map { $0.value })
          expectEqual(d[reference[offset].key], reference[offset].value)
        }
      }
    }
  }

  func test_swapAt() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("i", in: 0 ..< count) { i in
        withEvery("j", in: 0 ..< count) { j in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
              let t = reference[i].value
              reference[i].value = reference[j].value
              reference[j].value = t
              withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
                d.values.swapAt(i, j)
                expectEqualElements(d.values, reference.map { $0.value })
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
            var values = reference.map { $0.value }
            let expectedPivot = values.partition { $0.payload < 100 + count / 2 }
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let actualPivot = d.values.partition { $0.payload < 100 + count / 2 }
              expectEqual(actualPivot, expectedPivot)
              expectEqualElements(d.values, values)
              withEvery("i", in: 0 ..< count) { i in
                expectEqual(d[reference[i].key], values[i])
              }
            }
          }
        }
      }
    }
  }

  func test_withUnsafeBufferPointer() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          typealias R = [LifetimeTracked<Int>]
          let actual =
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d -> R in
              d.values.withUnsafeBufferPointer { buffer -> R in
                Array(buffer)
              }
            }
          expectEqualElements(actual, reference.map { $0.value })
        }
      }
    }
  }

  func test_withUnsafeMutableBufferPointer() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          let replacement = tracker.instances(for: (0 ..< count).map { -$0 })
          typealias R = [LifetimeTracked<Int>]
          let actual =
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d -> R in
              d.values.withUnsafeMutableBufferPointer { buffer -> R in
                let result = Array(buffer)
                expectEqual(buffer.count, replacement.count, trapping: true)
                for i in 0 ..< replacement.count {
                  buffer[i] = replacement[i]
                }
                return result
              }
            }
          expectEqualElements(actual, reference.map { $0.value })
          expectEqualElements(d.values, replacement)
          withEvery("i", in: 0 ..< count) { i in
            expectEqual(d[reference[i].key], replacement[i])
          }
        }
      }
    }
  }

  func test_withContiguousMutableStorageIfAvailable() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          let replacement = tracker.instances(for: (0 ..< count).map { -$0 })
          typealias R = [LifetimeTracked<Int>]
          let actual =
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d -> R? in
              d.values.withContiguousMutableStorageIfAvailable { buffer -> R in
                let result = Array(buffer)
                expectEqual(buffer.count, replacement.count, trapping: true)
                for i in 0 ..< replacement.count {
                  buffer[i] = replacement[i]
                }
                return result
              }
            }
          if let actual = actual {
            expectEqualElements(actual, reference.map { $0.value })
            expectEqualElements(d.values, replacement)
            withEvery("i", in: 0 ..< count) { i in
              expectEqual(d[reference[i].key], replacement[i])
            }
          } else {
            expectFailure("OrderedDictionary.Value isn't contiguous?")
          }
        }
      }
    }
  }
}
