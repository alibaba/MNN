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

extension OrderedDictionary: DictionaryAPIExtras {}

class OrderedDictionaryTests: CollectionTestCase {
  func test_empty() {
    let d = OrderedDictionary<String, Int>()
    expectEqualElements(d, [])
    expectEqual(d.count, 0)
  }

  func test_init_minimumCapacity() {
    let d = OrderedDictionary<String, Int>(minimumCapacity: 1000)
    expectGreaterThanOrEqual(d.keys.__unstable.capacity, 1000)
    expectGreaterThanOrEqual(d.values.elements.capacity, 1000)
    expectEqual(d.keys.__unstable.reservedScale, 0)
  }

  func test_init_minimumCapacity_persistent() {
    let d = OrderedDictionary<String, Int>(minimumCapacity: 1000, persistent: true)
    expectGreaterThanOrEqual(d.keys.__unstable.capacity, 1000)
    expectGreaterThanOrEqual(d.values.elements.capacity, 1000)
    expectNotEqual(d.keys.__unstable.reservedScale, 0)
  }

  func test_uniqueKeysWithValues_Dictionary() {
    let items: Dictionary<String, Int> = [
      "zero": 0,
      "one": 1,
      "two": 2,
      "three": 3,
    ]
    let d = OrderedDictionary(uncheckedUniqueKeysWithValues: items)
    expectEqualElements(d, items)
  }

  func test_uniqueKeysWithValues_labeled_tuples() {
    let items: KeyValuePairs<String, Int> = [
      "zero": 0,
      "one": 1,
      "two": 2,
      "three": 3,
    ]
    let d = OrderedDictionary(uncheckedUniqueKeysWithValues: items)
    expectEqualElements(d, items)
  }

  func test_uniqueKeysWithValues_unlabeled_tuples() {
    let items: [(String, Int)] = [
      ("zero", 0),
      ("one", 1),
      ("two", 2),
      ("three", 3),
    ]
    let d = OrderedDictionary(uncheckedUniqueKeysWithValues: items)
    expectEqualElements(d, items)
  }

  func test_uniqueKeys_values() {
    let d = OrderedDictionary(
    uncheckedUniqueKeys: ["zero", "one", "two", "three"],
      values: [0, 1, 2, 3])
    expectEqualElements(d, [
      (key: "zero", value: 0),
      (key: "one", value: 1),
      (key: "two", value: 2),
      (key: "three", value: 3),
    ])
  }

  func test_uniquing_initializer_labeled_tuples() {
    let items: KeyValuePairs<String, Int> = [
      "a": 1,
      "b": 1,
      "c": 1,
      "a": 2,
      "a": 2,
      "b": 1,
      "d": 3,
    ]
    let d = OrderedDictionary(items, uniquingKeysWith: +)
    expectEqualElements(d, [
      (key: "a", value: 5),
      (key: "b", value: 2),
      (key: "c", value: 1),
      (key: "d", value: 3)
    ])
  }

  func test_uniquing_initializer_unlabeled_tuples() {
    let items: [(String, Int)] = [
      ("a", 1),
      ("b", 1),
      ("c", 1),
      ("a", 2),
      ("a", 2),
      ("b", 1),
      ("d", 3),
    ]
    let d = OrderedDictionary(items, uniquingKeysWith: +)
    expectEqualElements(d, [
      (key: "a", value: 5),
      (key: "b", value: 2),
      (key: "c", value: 1),
      (key: "d", value: 3)
    ])
  }

  func test_grouping_initializer() {
    let items: [String] = [
      "one", "two", "three", "four", "five",
      "six", "seven", "eight", "nine", "ten"
    ]
    let d = OrderedDictionary<Int, ContiguousArray<String>>(
      grouping: items, by: { $0.count })
    expectEqualElements(d, [
      (key: 3, value: ["one", "two", "six", "ten"]),
      (key: 5, value: ["three", "seven", "eight"]),
      (key: 4, value: ["four", "five", "nine"]),
    ] as [(key: Int, value: ContiguousArray<String>)])
  }

  func test_grouping_initializer_inference() {
    struct StatEvent: Equatable {
       var name: String
       var date: String
       var hours: Int
    }

    let statEvents = [
       StatEvent(name: "lunch", date: "01-01-2015", hours: 1),
       StatEvent(name: "dinner", date: "01-01-2015", hours: 1),
       StatEvent(name: "dinner", date: "01-01-2015", hours: 1),
       StatEvent(name: "lunch", date: "01-01-2015", hours: 1),
       StatEvent(name: "dinner", date: "01-01-2015", hours: 1)
    ]

    // We expect this to be inferred to `OrderedDictionary<String, [StatEvent]>`
    let d = OrderedDictionary(grouping: statEvents, by: { $0.name })
    let expected = [
      (key: "lunch", value: [statEvents[0], statEvents[3]]),
      (key: "dinner", value: [statEvents[1], statEvents[2], statEvents[4]]),
    ]
    expectEqualElements(d, expected)
  }


  func test_uncheckedUniqueKeysWithValues_labeled_tuples() {
    let items: KeyValuePairs<String, Int> = [
      "zero": 0,
      "one": 1,
      "two": 2,
      "three": 3,
    ]
    let d = OrderedDictionary(uncheckedUniqueKeysWithValues: items)
    expectEqualElements(d, items)
  }

  func test_uncheckedUniqueKeysWithValues_unlabeled_tuples() {
    let items: [(String, Int)] = [
      ("zero", 0),
      ("one", 1),
      ("two", 2),
      ("three", 3),
    ]
    let d = OrderedDictionary(uncheckedUniqueKeysWithValues: items)
    expectEqualElements(d, items)
  }

  func test_uncheckedUniqueKeys_values() {
    let d = OrderedDictionary(
    uncheckedUniqueKeys: ["zero", "one", "two", "three"],
      values: [0, 1, 2, 3])
    expectEqualElements(d, [
      (key: "zero", value: 0),
      (key: "one", value: 1),
      (key: "two", value: 2),
      (key: "three", value: 3),
    ])
  }

  func test_ExpressibleByDictionaryLiteral() {
    let d0: OrderedDictionary<String, Int> = [:]
    expectTrue(d0.isEmpty)

    let d1: OrderedDictionary<String, Int> = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    expectEqualElements(d1.map { $0.key }, ["one", "two", "three", "four"])
    expectEqualElements(d1.map { $0.value }, [1, 2, 3, 4])
  }

  func test_keys() {
    let d: OrderedDictionary = [
      "one": 1,
      "two": 2,
      "three": 3,
      "four": 4,
    ]
    expectEqual(d.keys, ["one", "two", "three", "four"] as OrderedSet)
  }

  func test_counts() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, _) = tracker.orderedDictionary(keys: 0 ..< count)
        expectEqual(d.isEmpty, count == 0)
        expectEqual(d.count, count)
        expectEqual(d.underestimatedCount, count)
      }
    }
  }

  func test_index_forKey() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          expectEqual(d.index(forKey: reference[offset].key), offset)
        }
        expectNil(d.index(forKey: tracker.instance(for: -1)))
        expectNil(d.index(forKey: tracker.instance(for: count)))
      }
    }
  }

  func test_subscript_getter() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        withEvery("offset", in: 0 ..< count) { offset in
          expectEqual(d[reference[offset].key], reference[offset].value)
        }
        expectNil(d[tracker.instance(for: -1)])
        expectNil(d[tracker.instance(for: count)])
      }
    }
  }

  func test_subscript_setter_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d[reference[offset].key] = replacement
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_remove() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d[reference[offset].key] = nil
              reference.remove(at: offset)
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              d[keys[offset]] = values[offset]
              expectEqualElements(d, reference.prefix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_subscript_setter_noop() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          let key = tracker.instance(for: -1)
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            d[key] = nil
          }
          expectEqualElements(d, reference)
        }
      }
    }
  }

  func mutate<T, R>(
    _ value: inout T,
    _ body: (inout T) throws -> R
  ) rethrows -> R {
    try body(&value)
  }

  func test_subscript_modify_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              mutate(&d[reference[offset].key]) { $0 = replacement }
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_remove() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = reference[offset].key
              mutate(&d[key]) { v in
                expectEqual(v, reference[offset].value)
                v = nil
              }
              reference.remove(at: offset)
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              mutate(&d[keys[offset]]) { v in
                expectNil(v)
                v = values[offset]
              }
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.prefix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_subscript_modify_noop() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          let key = tracker.instance(for: -1)
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            mutate(&d[key]) { v in
              expectNil(v)
              v = nil
            }
          }
          expectEqualElements(d, reference)
        }
      }
    }
  }

  func test_defaulted_subscript_getter() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
          let fallback = tracker.instance(for: -1)
          withEvery("offset", in: 0 ..< count) { offset in
            let key = reference[offset].key
            expectEqual(d[key, default: fallback], reference[offset].value)
          }
          expectEqual(
            d[tracker.instance(for: -1), default: fallback],
            fallback)
          expectEqual(
            d[tracker.instance(for: count), default: fallback],
            fallback)
        }
      }
    }
  }

  func test_defaulted_subscript_modify_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            let fallback = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = reference[offset].key
              mutate(&d[key, default: fallback]) { v in
                expectEqual(v, reference[offset].value)
                v = replacement
              }
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_defaulted_subscript_modify_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          let fallback = tracker.instance(for: -1)
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = keys[offset]
              mutate(&d[key, default: fallback]) { v in
                expectEqual(v, fallback)
                v = values[offset]
              }
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.prefix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = reference[offset].key
              let old = d.updateValue(replacement, forKey: key)
              expectEqual(old, reference[offset].value)
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = keys[offset]
              let old = d.updateValue(values[offset], forKey: key)
              expectNil(old)
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.prefix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_insertingAt_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = reference[offset].key
              let (old, index) =
                d.updateValue(replacement, forKey: key, insertingAt: 0)
              expectEqual(old, reference[offset].value)
              expectEqual(index, offset)
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_insertingAt_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = keys[count - 1 - offset]
              let value = values[count - 1 - offset]
              let (old, index) =
                d.updateValue(value, forKey: key, insertingAt: 0)
              expectNil(old)
              expectEqual(index, 0)
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.suffix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_default_closure_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            let fallback = tracker.instance(for: -2)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = reference[offset].key
              d.updateValue(forKey: key, default: fallback) { value in
                expectEqual(value, reference[offset].value)
                value = replacement
              }
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_default_closure_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          let fallback = tracker.instance(for: -2)
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let key = keys[offset]
              d.updateValue(forKey: key, default: fallback) { value in
                expectEqual(value, fallback)
                value = values[offset]
              }
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.prefix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_insertingDefault_at_closure_update() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            let replacement = tracker.instance(for: -1)
            let fallback = tracker.instance(for: -2)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let (key, value) = reference[offset]
              d.updateValue(forKey: key, insertingDefault: fallback, at: 0) { v in
                expectEqual(v, value)
                v = replacement
              }
              reference[offset].value = replacement
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_updateValue_forKey_insertingDefault_at_closure_insert() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          let keys = tracker.instances(for: 0 ..< count)
          let values = tracker.instances(for: (0 ..< count).map { 100 + $0 })
          let reference = zip(keys, values).map { (key: $0.0, value: $0.1) }
          var d: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>> = [:]
          let fallback = tracker.instance(for: -2)
          withEvery("offset", in: 0 ..< count) { offset in
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let (key, value) = reference[count - 1 - offset]
              d.updateValue(forKey: key, insertingDefault: fallback, at: 0) { v in
                expectEqual(v, fallback)
                v = value
              }
              expectEqual(d.count, offset + 1)
              expectEqualElements(d, reference.suffix(offset + 1))
            }
          }
        }
      }
    }
  }

  func test_removeValue_forKey() {
    withEvery("count", in: 0 ..< 30) { count in
      withEvery("offset", in: 0 ..< count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
            withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
              let (key, value) = reference.remove(at: offset)
              let actual = d.removeValue(forKey: key)
              expectEqual(actual, value)

              expectEqualElements(d, reference)
              expectNil(d.removeValue(forKey: key))
            }
          }
        }
      }
    }
  }

  func test_merge_labeled_tuple() {
    var d: OrderedDictionary = [
      "one": 1,
      "two": 1,
      "three": 1,
    ]

    let items: KeyValuePairs = [
      "one": 1,
      "one": 1,
      "three": 1,
      "four": 1,
      "one": 1,
    ]

    d.merge(items, uniquingKeysWith: +)

    expectEqualElements(d, [
      "one": 4,
      "two": 1,
      "three": 2,
      "four": 1,
    ] as KeyValuePairs)
  }

  func test_merge_unlabeled_tuple() {
    var d: OrderedDictionary = [
      "one": 1,
      "two": 1,
      "three": 1,
    ]

    let items: [(String, Int)] = [
      ("one", 1),
      ("one", 1),
      ("three", 1),
      ("four", 1),
      ("one", 1),
    ]

    d.merge(items, uniquingKeysWith: +)

    expectEqualElements(d, [
      "one": 4,
      "two": 1,
      "three": 2,
      "four": 1,
    ] as KeyValuePairs)
  }

  func test_merging_labeled_tuple() {
    let d: OrderedDictionary = [
      "one": 1,
      "two": 1,
      "three": 1,
    ]

    let items: KeyValuePairs = [
      "one": 1,
      "one": 1,
      "three": 1,
      "four": 1,
      "one": 1,
    ]

    let d2 = d.merging(items, uniquingKeysWith: +)

    expectEqualElements(d, [
      "one": 1,
      "two": 1,
      "three": 1,
    ] as KeyValuePairs)

    expectEqualElements(d2, [
      "one": 4,
      "two": 1,
      "three": 2,
      "four": 1,
    ] as KeyValuePairs)
  }

  func test_merging_unlabeled_tuple() {
    let d: OrderedDictionary = [
      "one": 1,
      "two": 1,
      "three": 1,
    ]

    let items: [(String, Int)] = [
      ("one", 1),
      ("one", 1),
      ("three", 1),
      ("four", 1),
      ("one", 1),
    ]

    let d2 = d.merging(items, uniquingKeysWith: +)

    expectEqualElements(d, [
      "one": 1,
      "two": 1,
      "three": 1,
    ] as KeyValuePairs)

    expectEqualElements(d2, [
      "one": 4,
      "two": 1,
      "three": 2,
      "four": 1,
    ] as KeyValuePairs)
  }

  func test_filter() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = OrderedDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.filter { item in
      c += 1
      expectEqual(item.value, 100 * item.key)
      return item.key.isMultiple(of: 2)
    }
    expectEqual(c, 100)
    expectEqualElements(d, items)

    expectEqualElements(d2, (0 ..< 50).compactMap { key in
      return (key: 2 * key, value: 200 * key)
    })
  }

  func test_mapValues() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = OrderedDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.mapValues { value -> String in
      c += 1
      expectTrue(value.isMultiple(of: 100))
      return "\(value)"
    }
    expectEqual(c, 100)
    expectEqualElements(d, items)

    expectEqualElements(d2, (0 ..< 100).compactMap { key in
      (key: key, value: "\(100 * key)")
    })
  }

  func test_compactMapValue() {
    let items = (0 ..< 100).map { ($0, 100 * $0) }
    let d = OrderedDictionary(uniqueKeysWithValues: items)

    var c = 0
    let d2 = d.compactMapValues { value -> String? in
      c += 1
      guard value.isMultiple(of: 200) else { return nil }
      expectTrue(value.isMultiple(of: 100))
      return "\(value)"
    }
    expectEqual(c, 100)
    expectEqualElements(d, items)

    expectEqualElements(d2, (0 ..< 50).map { key in
      (key: 2 * key, value: "\(200 * key)")
    })
  }

  func test_CustomStringConvertible() {
    let a: OrderedDictionary<Int, Int> = [:]
    expectEqual(a.description, "[:]")

    let b: OrderedDictionary<Int, Int> = [0: 1]
    expectEqual(b.description, "[0: 1]")

    let c: OrderedDictionary<Int, Int> = [0: 1, 2: 3, 4: 5]
    expectEqual(c.description, "[0: 1, 2: 3, 4: 5]")
  }

  func test_CustomDebugStringConvertible() {
    let a: OrderedDictionary<Int, Int> = [:]
    expectEqual(a.debugDescription, "[:]")

    let b: OrderedDictionary<Int, Int> = [0: 1]
    expectEqual(b.debugDescription, "[0: 1]")

    let c: OrderedDictionary<Int, Int> = [0: 1, 2: 3, 4: 5]
    expectEqual(c.debugDescription, "[0: 1, 2: 3, 4: 5]")
  }

  func test_CustomReflectable() {
    do {
      let d: OrderedDictionary<Int, Int> = [1: 2, 3: 4, 5: 6]
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
    checkHashable(equivalenceClasses: samples)
  }

  func test_Encodable() throws {
    let d1: OrderedDictionary<Int, Int> = [:]
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalEncoder.encode(d1), v1)

    let d2: OrderedDictionary<Int, Int> = [0: 1]
    let v2: MinimalEncoder.Value = .array([.int(0), .int(1)])
    expectEqual(try MinimalEncoder.encode(d2), v2)

    let d3: OrderedDictionary<Int, Int> = [0: 1, 2: 3]
    let v3: MinimalEncoder.Value =
      .array([.int(0), .int(1), .int(2), .int(3)])
    expectEqual(try MinimalEncoder.encode(d3), v3)

    let d4 = OrderedDictionary(
      uniqueKeys: 0 ..< 100,
      values: (0 ..< 100).map { 100 * $0 })
    let v4: MinimalEncoder.Value =
      .array((0 ..< 100).flatMap { [.int($0), .int(100 * $0)] })
    expectEqual(try MinimalEncoder.encode(d4), v4)
  }

  func test_Decodable() throws {
    typealias OD = OrderedDictionary<Int, Int>
    let d1: OD = [:]
    let v1: MinimalEncoder.Value = .array([])
    expectEqual(try MinimalDecoder.decode(v1, as: OD.self), d1)

    let d2: OD = [0: 1]
    let v2: MinimalEncoder.Value = .array([.int(0), .int(1)])
    expectEqual(try MinimalDecoder.decode(v2, as: OD.self), d2)

    let d3: OD = [0: 1, 2: 3]
    let v3: MinimalEncoder.Value =
      .array([.int(0), .int(1), .int(2), .int(3)])
    expectEqual(try MinimalDecoder.decode(v3, as: OD.self), d3)

    let d4 = OrderedDictionary(
      uniqueKeys: 0 ..< 100,
      values: (0 ..< 100).map { 100 * $0 })
    let v4: MinimalEncoder.Value =
      .array((0 ..< 100).flatMap { [.int($0), .int(100 * $0)] })
    expectEqual(try MinimalDecoder.decode(v4, as: OD.self), d4)

    let v5: MinimalEncoder.Value = .array([.int(0), .int(1), .int(2)])
    expectThrows(try MinimalDecoder.decode(v5, as: OD.self)) { error in
      guard case DecodingError.dataCorrupted(let context) = error else {
        expectFailure("Unexpected error \(error)")
        return
      }
      expectEqual(context.debugDescription,
                  "Unkeyed container reached end before value in key-value pair")

    }

    let v6: MinimalEncoder.Value = .array([.int(0), .int(1), .int(0), .int(2)])
    expectThrows(try MinimalDecoder.decode(v6, as: OD.self)) { error in
      guard case DecodingError.dataCorrupted(let context) = error else {
        expectFailure("Unexpected error \(error)")
        return
      }
      expectEqual(context.debugDescription, "Duplicate key at offset 2")
    }
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
                d.swapAt(i, j)
                expectEqualElements(d, reference)
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
              let actualPivot = d.partition { $0.key.payload < count / 2 }
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
              d.sort()
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
              d.sort(by: { $0.key > $1.key })
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
            expectEqualElements(d, items)

            var rng1 = RepeatableRandomNumberGenerator(seed: seed)
            items.shuffle(using: &rng1)

            var rng2 = RepeatableRandomNumberGenerator(seed: seed)
            d.shuffle(using: &rng2)

            items.sort(by: { $0.key < $1.key })
            d.sort()
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
          d.shuffle()
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
            d.reverse()
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
            d.removeAll()
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
            d.removeAll(keepingCapacity: true)
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
              let actual = d.remove(at: offset)
              let expectedItem = reference.remove(at: offset)
              expectEqual(actual.key, expectedItem.key)
              expectEqual(actual.value, expectedItem.value)
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
              d.removeSubrange(range)
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
    d1.removeSubrange(...10)
    expectEqualElements(d1, item[11...])

    var d2 = d
    d2.removeSubrange(..<10)
    expectEqualElements(d2, item[10...])

    var d3 = d
    d3.removeSubrange(10...)
    expectEqualElements(d3, item[0 ..< 10])
  }

  func test_removeLast() {
    withEvery("isShared", in: [false, true]) { isShared in
      withLifetimeTracking { tracker in
        var (d, reference) = tracker.orderedDictionary(keys: 0 ..< 30)
        withEvery("i", in: 0 ..< d.count) { i in
          withHiddenCopies(if: isShared, of: &d, checker: { $0._checkInvariants() }) { d in
            let actual = d.removeLast()
            let ref = reference.removeLast()
            expectEqual(actual.key, ref.key)
            expectEqual(actual.value, ref.value)
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
            let actual = d.removeFirst()
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
              d.removeLast(suffix)
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
              d.removeFirst(prefix)
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
              d.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
              reference.removeAll(where: { !$0.key.payload.isMultiple(of: n) })
              expectEqualElements(d, reference)
            }
          }
        }
      }
    }
  }

  func test_Sequence() {
    withEvery("count", in: 0 ..< 30) { count in
      withLifetimeTracking { tracker in
        let (d, reference) = tracker.orderedDictionary(keys: 0 ..< count)
        checkSequence(
          { d },
          expectedContents: reference,
          by: { $0.key == $1.0 && $0.value == $1.1 })
      }
    }
  }

  func test_uniqueKeysWithValues_initializer_ambiguity() {
    // https://github.com/apple/swift-collections/issues/125

    let names = ["dylan", "bob", "aaron", "carol"]
    let expected = names.map { (key: $0, value: 0) }

    let d1 = OrderedDictionary(uniqueKeysWithValues: names.map { ($0, 0) })
    expectEqualElements(d1, expected)

    let d2 = OrderedDictionary(
      uniqueKeysWithValues: ["dylan", "bob", "aaron", "carol"].map { ($0, 0) })
    expectEqualElements(d2, expected)

    let d3 = OrderedDictionary(
      uniqueKeysWithValues: names.map { (key: $0, value: 0) })
    expectEqualElements(d3, expected)

    let d4 = OrderedDictionary(
      uniqueKeysWithValues: ["dylan", "bob", "aaron", "carol"].map { (key: $0, value: 0) })
    expectEqualElements(d4, expected)
  }

  func test_uniquingKeysWith_initializer_ambiguity() {
    // https://github.com/apple/swift-collections/issues/125
    let d1 = OrderedDictionary(
      ["a", "b", "a"].map { ($0, 1) },
      uniquingKeysWith: +)
    expectEqualElements(d1, ["a": 2, "b": 1] as KeyValuePairs)

    let d2 = OrderedDictionary(
      ["a", "b", "a"].map { (key: $0, value: 1) },
      uniquingKeysWith: +)
    expectEqualElements(d2, ["a": 2, "b": 1] as KeyValuePairs)
  }

  func test_merge_ambiguity() {
    // https://github.com/apple/swift-collections/issues/125

    var d1: OrderedDictionary = ["a": 1, "b": 2]
    d1.merge(
      ["c", "a"].map { ($0, 1) },
      uniquingKeysWith: +
    )
    expectEqualElements(d1, ["a": 2, "b": 2, "c": 1] as KeyValuePairs)

    var d2: OrderedDictionary = ["a": 1, "b": 2]
    d2.merge(
      ["c", "a"].map { (key: $0, value: 1) },
      uniquingKeysWith: +
    )
    expectEqualElements(d2, ["a": 2, "b": 2, "c": 1] as KeyValuePairs)
  }

  func test_merging_ambiguity() {
    // https://github.com/apple/swift-collections/issues/125

    let d1: OrderedDictionary = ["a": 1, "b": 2]
    let d1m = d1.merging(
      ["c", "a"].map { ($0, 1) },
      uniquingKeysWith: +
    )
    expectEqualElements(d1m, ["a": 2, "b": 2, "c": 1] as KeyValuePairs)

    let d2: OrderedDictionary = ["a": 1, "b": 2]
    let d2m = d2.merging(
      ["c", "a"].map { (key: $0, value: 1) },
      uniquingKeysWith: +
    )
    expectEqualElements(d2m, ["a": 2, "b": 2, "c": 1] as KeyValuePairs)
  }
}

