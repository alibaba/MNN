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
import _CollectionsTestSupport
@_spi(Testing) import DequeModule
#endif

/// Exhaustive tests for `Deque`'s implementations for `RangeReplaceableCollection`
/// requirements.
final class RangeReplaceableCollectionTests: CollectionTestCase {
  // Note: Most of the test below are exhaustively testing the behavior
  // of a particular mutation or query on every possible deque layout
  // of a small set of capacities. This helps catching issues that may only
  // occur within rare constellations of deque state -- such as when the range
  // of occupied slots is wrapped at a particular point.

  func test_emptyInitializer() {
    let deque = Deque<Int>()
    expectTrue(deque.isEmpty)
    expectEqual(deque.count, 0)
    expectEqual(deque.startIndex, deque.endIndex)
    expectEqual(deque.distance(from: deque.startIndex, to: deque.endIndex), 0)
    expectEqual(Array(deque), [])
  }

  func test_singleElement() {
    let deque = Deque([42])
    expectFalse(deque.isEmpty)
    expectEqual(deque.count, 1)
    expectLessThan(deque.startIndex, deque.endIndex)
    expectEqual(deque.index(after: deque.startIndex), deque.endIndex)
    expectEqual(deque.distance(from: deque.startIndex, to: deque.endIndex), 1)
    expectEqual(deque[0], 42)
    expectEqual(Array(deque), [42])
  }

  func test_sequenceInitializer() {
    withEvery("count", in: [0, 1, 2, 10, 100]) { count in
      let ucVariants: [UnderestimatedCountBehavior] = [.precise, .half, .value(min(1, count))]
      withEvery("underestimatedCount", in: ucVariants) { underestimatedCount in
        withLifetimeTracking { tracker in
          let contents = tracker.instances(for: 0 ..< count)
          let d1 = Deque(MinimalSequence(elements: contents, underestimatedCount: underestimatedCount))
          expectEqualElements(d1, contents)
        }
      }
    }
  }

  func test_sequenceInitializer_ContiguousArray() {
    withEvery("count", in: [0, 1, 2, 10, 100]) { count in
      withLifetimeTracking { tracker in
        let contents = ContiguousArray(tracker.instances(for: 0 ..< count))
        let d1 = Deque(contents)
        expectEqualElements(d1, contents)
      }
    }
  }

  func test_sequenceInitializer_bridgedArray() {
    // https://github.com/apple/swift-collections/issues/27
    withEvery("count", in: [0, 1, 2, 10, 100]) { count in
      let contents: [AnyObject] = (0 ..< count).map { _ in NSObject() }
      let array: [AnyObject] = contents.withUnsafeBufferPointer { buffer in
        NSArray(objects: buffer.baseAddress, count: buffer.count) as [AnyObject]
      }
      let deque = Deque(array)
      expectEquivalentElements(deque, contents, by: ===)
    }
  }

  func test_replaceSubrange_withMinimalCollection() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEveryRange("range", in: 0 ..< layout.count) { range in
        withEvery("replacementCount", in: [0, 1, 2, 3, 5, 10]) { replacementCount in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              let extras = tracker.instances(for: layout.count ..< layout.count + replacementCount)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                contents.replaceSubrange(range, with: extras)
                let minimal = MinimalCollection(extras)
                deque.replaceSubrange(range, with: minimal)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_replaceSubrange_withArray() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEveryRange("range", in: 0 ..< layout.count) { range in
        withEvery("replacementCount", in: [0, 1, 2, 3, 5, 10]) { replacementCount in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              let extras = tracker.instances(for: layout.count ..< layout.count + replacementCount)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                contents.replaceSubrange(range, with: extras)
                deque.replaceSubrange(range, with: extras)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_reserveCapacity() {
    // FIXME: Implement
  }

  func test_repeatingInitializer() {
    withEvery("count", in: 0 ..< 10) { count in
      withLifetimeTracking { tracker in
        let item = tracker.instance(for: 0)
        let deque = Deque(repeating: item, count: count)
        expectEqual(Array(deque), Array(repeating: item, count: count))
      }
    }
  }

  func test_appendOne() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          let extra = tracker.instance(for: layout.count)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.append(extra)
            deque.append(extra)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_appendManyFromMinimalSequence() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("appendCount", in: 0 ..< 10) { appendCount in
        withEvery("underestimatedCount", in: [UnderestimatedCountBehavior.precise, .half, .value(min(1, appendCount))]) { underestimatedCount in
          withEvery("isContiguous", in: [false, true]) { isContiguous in
            withEvery("isShared", in: [false, true]) { isShared in
              withLifetimeTracking { tracker in
                var (deque, contents) = tracker.deque(with: layout)
                let extras = tracker.instances(for: layout.count ..< layout.count + appendCount)
                let sequence = MinimalSequence(
                  elements: extras,
                  underestimatedCount: underestimatedCount,
                  isContiguous: isContiguous)
                withHiddenCopies(if: isShared, of: &deque) { deque in
                  contents.append(contentsOf: extras)
                  deque.append(contentsOf: sequence)
                  expectEqualElements(deque, contents)
                }
              }
            }
          }
        }
      }
    }
  }


  func test_appendManyFromMinimalCollection() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("appendCount", in: 0 ..< 10) { appendCount in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extra = tracker.instances(for: layout.count ..< layout.count + appendCount)
            let minimal = MinimalCollection(extra)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.append(contentsOf: extra)
              deque.append(contentsOf: minimal)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_appendManyFromContiguousArray() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("appendCount", in: 0 ..< 10) { appendCount in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extraRange = layout.count ..< layout.count + appendCount
            let extra = ContiguousArray(tracker.instances(for: extraRange))
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.append(contentsOf: extra)
              deque.append(contentsOf: extra)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_appendManyFromBridgedArray() {
    // https://github.com/apple/swift-collections/issues/27
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("appendCount", in: 0 ..< 10) { appendCount in
        withEvery("isShared", in: [false, true]) { isShared in
          var contents: [NSObject] = (0 ..< layout.count).map { _ in NSObject() }
          var deque = Deque(layout: layout, contents: contents)
          let extra: [NSObject] = (0 ..< appendCount)
            .map { _ in NSObject() }
            .withUnsafeBufferPointer { buffer in
              NSArray(objects: buffer.baseAddress, count: buffer.count) as! [NSObject]
            }
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.append(contentsOf: extra)
            deque.append(contentsOf: extra)
            expectEquivalentElements(deque, contents, by: ===)
          }
        }
      }
    }
  }

  func test_insertOneElement() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ... layout.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let extra = tracker.instance(for: layout.count)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.insert(extra, at: offset)
              deque.insert(extra, at: offset)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_insertFromMinimalCollection() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ... layout.count) { offset in
        withEvery("insertCount", in: 0 ..< 10) { insertCount in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              let extras = tracker.instances(for: layout.count ..< layout.count + insertCount)
              let minimal = MinimalCollection(extras)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                contents.insert(contentsOf: extras, at: offset)
                deque.insert(contentsOf: minimal, at: offset)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_insertFromArray() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ... layout.count) { offset in
        withEvery("insertCount", in: 0 ..< 10) { insertCount in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              let extras = tracker.instances(for: layout.count ..< layout.count + insertCount)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                contents.insert(contentsOf: extras, at: offset)
                deque.insert(contentsOf: extras, at: offset)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_removeOne() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ..< layout.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              let expected = contents.remove(at: offset)
              let actual = deque.remove(at: offset)
              expectEqual(actual, expected)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_removeSubrange() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEveryRange("range", in: 0 ..< layout.count) { range in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.removeSubrange(range)
              deque.removeSubrange(range)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_customRemoveLast_one() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            let expected = contents._customRemoveLast()
            let actual = deque._customRemoveLast()
            expectEqual(actual, expected)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_removeLast() {
    // `removeLast`'s implementation in the stdlib is derived from `_customRemoveLast`.
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            let expected = contents.removeLast()
            let actual = deque.removeLast()
            expectEqual(actual, expected)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_popLast() {
    // `popLast`'s implementation in the stdlib is derived from `_customRemoveLast`.
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            let expected = contents.popLast()
            let actual = deque.popLast()
            expectEqual(actual, expected)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_customRemoveLast_many() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("n", in: 0 ... layout.count) { n in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.removeLast(n)
              expectTrue(deque._customRemoveLast(n))
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_removeLast_many() {
    // `removeLast`'s implementation in the stdlib is derived from `_customRemoveLast`.
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("n", in: 0 ... layout.count) { n in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.removeLast(n)
              deque.removeLast(n)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_removeFirst_one() {
    // `removeLast`'s implementation in the stdlib is derived from `_customRemoveLast`.
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            let expected = contents.removeFirst()
            let actual = deque.removeFirst()
            expectEqual(actual, expected)
            expectEqualElements(deque, contents)
          }
        }
      }
    }
  }

  func test_removeFirst_many() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      guard layout.count > 0 else { return }
      withEvery("n", in: 0 ... layout.count) { n in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            withHiddenCopies(if: isShared, of: &deque) { deque in
              contents.removeFirst(n)
              deque.removeFirst(n)
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_removeAll_discardingCapacity() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        guard layout.count > 0 else { return }
        var deque: Deque<LifetimeTracked<Int>> = []
        withLifetimeTracking { tracker in
          var contents: [LifetimeTracked<Int>] = []
          (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.removeAll()
            deque.removeAll()
            expectEqual(deque.count, 0)
            expectEqual(deque._capacity, 0) // This assumes the empty singleton has zero capacity.
          }
        } // All elements must be deinitialized at this point.
        withExtendedLifetime(deque) {}
      }
    }
  }

  func test_removeAll_keepingCapacity() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        guard layout.count > 0 else { return }
        var deque: Deque<LifetimeTracked<Int>> = []
        withLifetimeTracking { tracker in
          var contents: [LifetimeTracked<Int>] = []
          (deque, contents) = tracker.deque(with: layout)
          withHiddenCopies(if: isShared, of: &deque) { deque in
            contents.removeAll(keepingCapacity: true)
            deque.removeAll(keepingCapacity: true)
            expectEqual(deque.count, 0)
            expectEqual(deque._capacity, layout.capacity)
          }
        } // All elements must be deinitialized at this point.
        withExtendedLifetime(deque) {}
      }
    }
  }
}
