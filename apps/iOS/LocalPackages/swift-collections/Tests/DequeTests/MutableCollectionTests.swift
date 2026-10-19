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
import _CollectionsTestSupport
@_spi(Testing) import DequeModule
#endif

final class MutableCollectiontests: CollectionTestCase {
  // Note: Most of the test below are exhaustively testing the behavior
  // of a particular mutation or query on every possible deque layout
  // of a small set of capacities. This helps catching issues that may only
  // occur within rare constellations of deque state -- such as when the range
  // of occupied slots is wrapped at a particular point.

  func test_subscript_assignment() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ..< layout.count) { offset in
        withEvery("isShared", in: [false, true]) { isShared in
          withLifetimeTracking { tracker in
            var (deque, contents) = tracker.deque(with: layout)
            let replacement = tracker.instance(for: layout.count)
            contents[offset] = replacement
            withHiddenCopies(if: isShared, of: &deque) { deque in
              deque[offset] = replacement
              expectEqualElements(deque, contents)
            }
          }
        }
      }
    }
  }

  func test_subscript_inPlaceMutation() {
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

    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("offset", in: 0 ..< layout.count) { offset in
        withLifetimeTracking { tracker in
          var (deque, contents) = tracker.deque(with: layout)
          // Discard `contents` and recreate it from scratch to make sure its
          // elements are uniquely referenced.
          contents = tracker.instances(for: 0 ..< layout.count)

          checkMutation(&contents[offset], tracker: tracker, delta: 100)
          checkMutation(&deque[offset], tracker: tracker, delta: 100)

          expectEqualElements(deque, contents)
        }
      }
    }
  }

  func test_subscript_rangeAssignmentAcrossInstances() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 6]) { layout in
      withEveryRange("targetRange", in: 0 ..< layout.count) { targetRange in
        let replacementCount = 4
        withEvery("replacementStartSlot", in: 0 ..< replacementCount) { replacementStartSlot in
          withEveryRange("sourceRange", in: 0 ..< replacementCount) { sourceRange in
            withEvery("isShared", in: [false, true]) { isShared in
              withLifetimeTracking { tracker in
                var (deque, contents) = tracker.deque(with: layout)
                let replacementLayout = DequeLayout(
                  capacity: replacementCount,
                  startSlot: replacementStartSlot,
                  count: replacementCount,
                  startValue: layout.count)
                let (replacementDeque, replacementArray) = tracker.deque(with: replacementLayout)
                contents[targetRange] = replacementArray[sourceRange]
                withHiddenCopies(if: isShared, of: &deque) { deque in
                  deque[targetRange] = replacementDeque[sourceRange]
                  expectEqualElements(deque, contents)
                }
              }
            }
          }
        }
      }
    }
  }

  func test_subscript_rangeAssignmentWithinTheSameInstance() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEveryRange("targetRange", in: 0 ..< layout.count) { targetRange in
        withEveryRange("sourceRange", in: 0 ..< layout.count) { sourceRange in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              contents[targetRange] = contents[sourceRange]
              withHiddenCopies(if: isShared, of: &deque) { deque in
                deque[targetRange] = deque[sourceRange]
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_swapAt() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("i", in: 0 ..< layout.count) { i in
        withEvery("j", in: 0 ..< layout.count) { j in
          withEvery("isShared", in: [false, true]) { isShared in
            withLifetimeTracking { tracker in
              var (deque, contents) = tracker.deque(with: layout)
              contents.swapAt(i, j)
              withHiddenCopies(if: isShared, of: &deque) { deque in
                deque.swapAt(i, j)
                expectEqualElements(deque, contents)
              }
            }
          }
        }
      }
    }
  }

  func test_withContiguousMutableStorageIfAvailable() {
    withEveryDeque("deque", ofCapacities: [0, 1, 2, 3, 5, 10]) { layout in
      withEvery("isShared", in: [false, true]) { isShared in
        withLifetimeTracking { tracker in
          var (deque, expected) = tracker.deque(with: layout)
          let replacement = tracker.instances(for: 100 ..< 100 + layout.count)
          let actual: [LifetimeTracked<Int>]? = withHiddenCopies(if: isShared, of: &deque) { deque in
            deque.withContiguousMutableStorageIfAvailable { buffer in
              let result = Array(buffer)
              expectEqual(buffer.count, replacement.count, trapping: true)
              for i in 0 ..< replacement.count {
                buffer[i] = replacement[i]
              }
              return result
            }
          }
          if let actual = actual {
            expectFalse(layout.isWrapped)
            expectEqualElements(actual, expected)
            expectEqualElements(deque, replacement)
          } else {
            expectTrue(layout.isWrapped)
          }
        }
      }
    }
  }
}
