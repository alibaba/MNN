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
import HeapModule
#endif

extension Heap {
  func itemsInAscendingOrder() -> [Element] {
    Array(sequence(state: self) { $0.popMin() })
  }
}

extension Heap {
  /// Creates a heap from the given array of elements, which must have already
  /// been heapified to form a binary min-max heap.
  ///
  /// - Precondition: `storage` has already been heapified.
  ///
  /// - Parameter storage: The elements of the heap.
  /// - Postcondition: `unordered.elementsEqual(s)`, where *s* is a sequence
  ///   with the same elements as pre-call `storage`.
  ///
  /// - Complexity: O(1)
  public init(raw storage: [Element]) {
    self.init(storage)
    precondition(self.unordered == storage)
  }
}

final class HeapTests: CollectionTestCase {
  func test_isEmpty() {
    var heap = Heap<Int>()
    expectTrue(heap.isEmpty)

    heap.insert(42)
    expectFalse(heap.isEmpty)

    let _ = heap.popMin()
    expectTrue(heap.isEmpty)
  }

  func test_count() {
    var heap = Heap<Int>()
    expectEqual(heap.count, 0)

    heap.insert(20)
    expectEqual(heap.count, 1)

    heap.insert(40)
    expectEqual(heap.count, 2)

    _ = heap.popMin()
    expectEqual(heap.count, 1)
  }

  func test_descriptions() {
    let a: Heap<Int> = []
    expectTrue(
      a.description.starts(with: "<0 items @"),
      "\(a.description)")
    expectTrue(
      a.debugDescription.starts(with: "<0 items @"),
      "\(a.debugDescription)")

    let b: Heap = [1]
    expectTrue(
      b.description.starts(with: "<1 item @"),
      "\(b.description)")
    expectTrue(
      b.debugDescription.starts(with: "<1 item @"))

    let c: Heap = [1, 2]
    expectTrue(c.description.starts(with: "<2 items @"))
    expectTrue(c.debugDescription.starts(with: "<2 items @"))
  }

  func test_unordered() {
    let heap = Heap<Int>((1...10))
    expectEqual(Set(heap.unordered), Set(1...10))
  }

  struct Task: Comparable {
    let name: String
    let priority: Int

    static func < (lhs: Task, rhs: Task) -> Bool {
      lhs.priority < rhs.priority
    }
  }

  func test_insert() {
    var heap = Heap<Task>()

    expectEqual(heap.count, 0)
    heap.insert(Task(name: "Hello, world", priority: 50))
    expectEqual(heap.count, 1)
  }

  func test_insert_random() {
    let c = 128
    withEvery("seed", in: 0 ..< 5_000) { seed in
      var rng = RepeatableRandomNumberGenerator(seed: seed)
      let input = (0 ..< c).shuffled(using: &rng)
      var heap: Heap<Int> = []
      var i = 0
      withEvery("value", in: input) { value in
        expectEqual(heap.count, i)
        heap.insert(value)
        i += 1
        expectEqual(heap.count, i)
      }
      expectEqualElements(heap.itemsInAscendingOrder(), 0 ..< c)
    }
  }

  func test_insert_contentsOf() {
    var heap = Heap<Int>()
    heap.insert(contentsOf: (1...10).shuffled())
    expectEqual(heap.count, 10)
    expectEqual(heap.popMax(), 10)
    expectEqual(heap.popMin(), 1)

    heap.insert(contentsOf: (21...50).shuffled())
    expectEqual(heap.count, 38)
    expectEqual(heap.max, 50)
    expectEqual(heap.min, 2)

    heap.insert(contentsOf: [-10, -9, -8, -7, -6, -5].shuffled())
    expectEqual(heap.count, 44)
    expectEqual(heap.min, -10)
  }

  func test_insert_contentsOf_withSequenceFunction() {
    func addTwo(_ i: Int) -> Int {
      i + 2
    }

    let evens = sequence(first: 0, next: addTwo(_:)).prefix(20)
    var heap = Heap(evens)
    expectEqual(heap.count, 20)

    heap.insert(contentsOf: sequence(first: 1, next: addTwo(_:)).prefix(20))
    expectEqual(heap.count, 40)

    withEvery("i", in: 0 ..< 40) { i in
      expectNotNil(heap.popMin()) { min in
        expectEqual(min, i)
      }
    }
    expectNil(heap.popMin())
  }

  func test_insert_contentsOf_exhaustive() {
    withEvery("c", in: 0 ..< 15) { c in
      withEverySubset("a", of: 0 ..< c) { a in
        let startInput = (0 ..< c).filter { !a.contains($0) }
        var heap = Heap(startInput)
        heap.insert(contentsOf: a.shuffled())
        expectEqualElements(heap.itemsInAscendingOrder(), 0 ..< c)
      }
    }
  }

  func test_min() {
    var heap = Heap<Int>()
    expectNil(heap.min)

    heap.insert(5)
    expectEqual(5, heap.min)

    heap.insert(12)
    expectEqual(5, heap.min)

    heap.insert(2)
    expectEqual(2, heap.min)

    heap.insert(1)
    expectEqual(1, heap.min)
  }

  func test_max() {
    var heap = Heap<Int>()
    expectNil(heap.max)

    heap.insert(42)
    expectEqual(42, heap.max)

    heap.insert(20)
    expectEqual(42, heap.max)

    heap.insert(63)
    expectEqual(63, heap.max)

    heap.insert(90)
    expectEqual(90, heap.max)
  }

  func test_popMin() {
    var heap = Heap<Int>()
    expectNil(heap.popMin())

    heap.insert(7)
    expectEqual(heap.popMin(), 7)

    heap.insert(12)
    heap.insert(9)
    expectEqual(heap.popMin(), 9)

    heap.insert(13)
    heap.insert(1)
    heap.insert(4)
    expectEqual(heap.popMin(), 1)

    for i in (1...20).shuffled() {
      heap.insert(i)
    }

    expectEqual(heap.popMin(), 1)
    expectEqual(heap.popMin(), 2)
    expectEqual(heap.popMin(), 3)
    expectEqual(heap.popMin(), 4)
    expectEqual(heap.popMin(), 4)  // One 4 was still in the heap from before
    expectEqual(heap.popMin(), 5)
    expectEqual(heap.popMin(), 6)
    expectEqual(heap.popMin(), 7)
    expectEqual(heap.popMin(), 8)
    expectEqual(heap.popMin(), 9)
    expectEqual(heap.popMin(), 10)
    expectEqual(heap.popMin(), 11)
    expectEqual(heap.popMin(), 12)
    expectEqual(heap.popMin(), 12)  // One 12 was still in the heap from before
    expectEqual(heap.popMin(), 13)
    expectEqual(heap.popMin(), 13)  // One 13 was still in the heap from before
    expectEqual(heap.popMin(), 14)
    expectEqual(heap.popMin(), 15)
    expectEqual(heap.popMin(), 16)
    expectEqual(heap.popMin(), 17)
    expectEqual(heap.popMin(), 18)
    expectEqual(heap.popMin(), 19)
    expectEqual(heap.popMin(), 20)

    expectNil(heap.popMin())
  }

  func test_popMax() {
    var heap = Heap<Int>()
    expectNil(heap.popMax())

    heap.insert(7)
    expectEqual(heap.popMax(), 7)

    heap.insert(12)
    heap.insert(9)
    expectEqual(heap.popMax(), 12)

    heap.insert(13)
    heap.insert(1)
    heap.insert(4)
    expectEqual(heap.popMax(), 13)

    for i in (1...20).shuffled() {
      heap.insert(i)
    }

    expectEqual(heap.popMax(), 20)
    expectEqual(heap.popMax(), 19)
    expectEqual(heap.popMax(), 18)
    expectEqual(heap.popMax(), 17)
    expectEqual(heap.popMax(), 16)
    expectEqual(heap.popMax(), 15)
    expectEqual(heap.popMax(), 14)
    expectEqual(heap.popMax(), 13)
    expectEqual(heap.popMax(), 12)
    expectEqual(heap.popMax(), 11)
    expectEqual(heap.popMax(), 10)
    expectEqual(heap.popMax(), 9)
    expectEqual(heap.popMax(), 9)  // One 9 was still in the heap from before
    expectEqual(heap.popMax(), 8)
    expectEqual(heap.popMax(), 7)
    expectEqual(heap.popMax(), 6)
    expectEqual(heap.popMax(), 5)
    expectEqual(heap.popMax(), 4)
    expectEqual(heap.popMax(), 4)  // One 4 was still in the heap from before
    expectEqual(heap.popMax(), 3)
    expectEqual(heap.popMax(), 2)
    expectEqual(heap.popMax(), 1)
    expectEqual(heap.popMax(), 1)  // One 1 was still in the heap from before

    expectNil(heap.popMax())
  }

  func test_removeMin() {
    var heap = Heap<Int>((1...20).shuffled())

    expectEqual(heap.removeMin(), 1)
    expectEqual(heap.removeMin(), 2)
    expectEqual(heap.removeMin(), 3)
    expectEqual(heap.removeMin(), 4)
    expectEqual(heap.removeMin(), 5)
    expectEqual(heap.removeMin(), 6)
    expectEqual(heap.removeMin(), 7)
    expectEqual(heap.removeMin(), 8)
    expectEqual(heap.removeMin(), 9)
    expectEqual(heap.removeMin(), 10)
    expectEqual(heap.removeMin(), 11)
    expectEqual(heap.removeMin(), 12)
    expectEqual(heap.removeMin(), 13)
    expectEqual(heap.removeMin(), 14)
    expectEqual(heap.removeMin(), 15)
    expectEqual(heap.removeMin(), 16)
    expectEqual(heap.removeMin(), 17)
    expectEqual(heap.removeMin(), 18)
    expectEqual(heap.removeMin(), 19)
    expectEqual(heap.removeMin(), 20)
  }

  func test_removeMax() {
    var heap = Heap<Int>((1...20).shuffled())

    expectEqual(heap.removeMax(), 20)
    expectEqual(heap.removeMax(), 19)
    expectEqual(heap.removeMax(), 18)
    expectEqual(heap.removeMax(), 17)
    expectEqual(heap.removeMax(), 16)
    expectEqual(heap.removeMax(), 15)
    expectEqual(heap.removeMax(), 14)
    expectEqual(heap.removeMax(), 13)
    expectEqual(heap.removeMax(), 12)
    expectEqual(heap.removeMax(), 11)
    expectEqual(heap.removeMax(), 10)
    expectEqual(heap.removeMax(), 9)
    expectEqual(heap.removeMax(), 8)
    expectEqual(heap.removeMax(), 7)
    expectEqual(heap.removeMax(), 6)
    expectEqual(heap.removeMax(), 5)
    expectEqual(heap.removeMax(), 4)
    expectEqual(heap.removeMax(), 3)
    expectEqual(heap.removeMax(), 2)
    expectEqual(heap.removeMax(), 1)
  }

  func test_minimumReplacement() {
    var heap = Heap(stride(from: 0, through: 27, by: 3).shuffled())
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.min, 0)

    // No change
    heap.replaceMin(with: 0)
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.min, 0)

    // Even smaller
    heap.replaceMin(with: -1)
    expectEqual(
      heap.itemsInAscendingOrder(), [-1, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.min, -1)

    // Larger, but not enough to usurp
    heap.replaceMin(with: 2)
    expectEqual(
      heap.itemsInAscendingOrder(), [2, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.min, 2)

    // Larger, moving another element to be the smallest
    heap.replaceMin(with: 5)
    expectEqual(
      heap.itemsInAscendingOrder(), [3, 5, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.min, 3)
  }

  func test_maximumReplacement() {
    var heap = Heap(stride(from: 0, through: 27, by: 3).shuffled())
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.max, 27)

    // No change
    heap.replaceMax(with: 27)
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    expectEqual(heap.max, 27)

    // Even larger
    heap.replaceMax(with: 28)
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 28])
    expectEqual(heap.max, 28)

    // Smaller, but not enough to usurp
    heap.replaceMax(with: 26)
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 26])
    expectEqual(heap.max, 26)

    // Smaller, moving another element to be the largest
    heap.replaceMax(with: 23)
    expectEqual(
      heap.itemsInAscendingOrder(), [0, 3, 6, 9, 12, 15, 18, 21, 23, 24])
    expectEqual(heap.max, 24)

    // Check the finer details.  As these peek into the stored structure, they
    // may need to be updated whenever the internal format changes.
    var heap2 = Heap(raw: [1])
    expectEqual(heap2.max, 1)
    expectEqual(Array(heap2.unordered), [1])
    expectEqual(heap2.replaceMax(with: 2), 1)
    expectEqual(heap2.max, 2)
    expectEqual(Array(heap2.unordered), [2])

    heap2 = Heap(raw: [1, 2])
    expectEqual(heap2.max, 2)
    expectEqual(Array(heap2.unordered), [1, 2])
    expectEqual(heap2.replaceMax(with: 3), 2)
    expectEqual(heap2.max, 3)
    expectEqual(Array(heap2.unordered), [1, 3])
    expectEqual(heap2.replaceMax(with: 0), 3)
    expectEqual(heap2.max, 1)
    expectEqual(Array(heap2.unordered), [0, 1])

    heap2 = Heap(raw: [5, 20, 31, 16, 8, 7, 18])
    expectEqual(heap2.max, 31)
    expectEqual(Array(heap2.unordered), [5, 20, 31, 16, 8, 7, 18])
    expectEqual(heap2.replaceMax(with: 29), 31)
    expectEqual(Array(heap2.unordered), [5, 20, 29, 16, 8, 7, 18])
    expectEqual(heap2.max, 29)
    expectEqual(heap2.replaceMax(with: 19), 29)
    expectEqual(Array(heap2.unordered), [5, 20, 19, 16, 8, 7, 18])
    expectEqual(heap2.max, 20)
    expectEqual(heap2.replaceMax(with: 15), 20)
    expectEqual(Array(heap2.unordered), [5, 16, 19, 15, 8, 7, 18])
    expectEqual(heap2.max, 19)
    expectEqual(heap2.replaceMax(with: 4), 19)
    expectEqual(Array(heap2.unordered), [4, 16, 18, 15, 8, 7, 5])
    expectEqual(heap2.max, 18)
  }

  // MARK: -

  func test_min_struct() {
    var heap = Heap<Task>()
    expectNil(heap.min)

    let firstTask = Task(name: "Do something", priority: 10)
    heap.insert(firstTask)
    expectEqual(heap.min, firstTask)

    let higherPriorityTask = Task(name: "Urgent", priority: 100)
    heap.insert(higherPriorityTask)
    expectEqual(heap.min, firstTask)

    let lowerPriorityTask = Task(name: "Get this done today", priority: 1)
    heap.insert(lowerPriorityTask)
    expectEqual(heap.min, lowerPriorityTask)
  }

  func test_max_struct() {
    var heap = Heap<Task>()
    expectNil(heap.max)

    let firstTask = Task(name: "Do something", priority: 10)
    heap.insert(firstTask)
    expectEqual(heap.max, firstTask)

    let lowerPriorityTask = Task(name: "Get this done today", priority: 1)
    heap.insert(lowerPriorityTask)
    expectEqual(heap.max, firstTask)

    let higherPriorityTask = Task(name: "Urgent", priority: 100)
    heap.insert(higherPriorityTask)
    expectEqual(heap.max, higherPriorityTask)
  }

  func test_popMin_struct() {
    var heap = Heap<Task>()
    expectNil(heap.popMin())

    let lowPriorityTask = Task(name: "Do something when you have time", priority: 1)
    heap.insert(lowPriorityTask)

    let highPriorityTask = Task(name: "Get this done today", priority: 50)
    heap.insert(highPriorityTask)

    let urgentTask = Task(name: "Urgent", priority: 100)
    heap.insert(urgentTask)

    expectEqual(heap.popMin(), lowPriorityTask)
    expectEqual(heap.popMin(), highPriorityTask)
    expectEqual(heap.popMin(), urgentTask)
    expectNil(heap.popMin())
  }

  func test_popMax_struct() {
    var heap = Heap<Task>()
    expectNil(heap.popMax())

    let lowPriorityTask = Task(name: "Do something when you have time", priority: 1)
    heap.insert(lowPriorityTask)

    let highPriorityTask = Task(name: "Get this done today", priority: 50)
    heap.insert(highPriorityTask)

    let urgentTask = Task(name: "Urgent", priority: 100)
    heap.insert(urgentTask)

    expectEqual(heap.popMax(), urgentTask)
    expectEqual(heap.popMax(), highPriorityTask)
    expectEqual(heap.popMax(), lowPriorityTask)
    expectNil(heap.popMax())
  }

  // MARK: -

  func test_initializer_fromCollection() {
    var heap = Heap((1...20).shuffled())
    expectEqual(heap.max, 20)

    expectEqual(heap.popMin(), 1)
    expectEqual(heap.popMax(), 20)
    expectEqual(heap.popMin(), 2)
    expectEqual(heap.popMax(), 19)
    expectEqual(heap.popMin(), 3)
    expectEqual(heap.popMax(), 18)
    expectEqual(heap.popMin(), 4)
    expectEqual(heap.popMax(), 17)
    expectEqual(heap.popMin(), 5)
    expectEqual(heap.popMax(), 16)
    expectEqual(heap.popMin(), 6)
    expectEqual(heap.popMax(), 15)
    expectEqual(heap.popMin(), 7)
    expectEqual(heap.popMax(), 14)
    expectEqual(heap.popMin(), 8)
    expectEqual(heap.popMax(), 13)
    expectEqual(heap.popMin(), 9)
    expectEqual(heap.popMax(), 12)
    expectEqual(heap.popMin(), 10)
    expectEqual(heap.popMax(), 11)
  }

  func test_initializer_fromSequence() {
    let heap = Heap((1...).prefix(20))
    expectEqual(heap.count, 20)
  }

  func test_initializer_fromArrayLiteral() {
    var heap: Heap = [1, 3, 5, 7, 9]
    expectEqual(heap.count, 5)

    expectEqual(heap.popMax(), 9)
    expectEqual(heap.popMax(), 7)
    expectEqual(heap.popMax(), 5)
    expectEqual(heap.popMax(), 3)
    expectEqual(heap.popMax(), 1)
  }

  func test_initializer_fromSequence_random() {
    withEvery("c", in: 0 ... 128) { c in
      withEvery(
        "seed", in: 0 ..< Swift.min((c + 2) * (c + 1), 100)
      ) { seed in
        var rng = RepeatableRandomNumberGenerator(seed: seed)
        let input = (0 ..< c).shuffled(using: &rng)
        let heap = Heap(input)
        if c > 0 {
          expectEqual(heap.min, 0)
          expectEqual(heap.max, c - 1)
          expectEqualElements(heap.itemsInAscendingOrder(), 0 ..< c)
        } else {
          expectNil(heap.min)
          expectNil(heap.max)
          expectEqualElements(heap.itemsInAscendingOrder(), [])
        }
      }
    }
  }

  struct Distinguishable: Comparable, CustomStringConvertible {
    var value: Int
    var id: Int

    static func ==(left: Self, right: Self) -> Bool {
      left.value == right.value
    }
    static func <(left: Self, right: Self) -> Bool {
      left.value < right.value
    }
    var description: String { "\(value)/\(id)" }
  }

  func test_tieBreaks_min() {
    var heap: Heap = [
      Distinguishable(value: 1, id: 1),
      Distinguishable(value: 1, id: 2),
      Distinguishable(value: 1, id: 3),
      Distinguishable(value: 1, id: 4),
      Distinguishable(value: 1, id: 5),
    ]
    while !heap.isEmpty {
      let oldID = heap.min!.id
      let newID = 10 * oldID
      let old = heap.replaceMin(with: Distinguishable(value: 1, id: newID))
      expectEqual(old.id, oldID)
      expectEqual(heap.min?.id, 10 * oldID)
      expectNotNil(heap.removeMin()) { min in
        expectEqual(min.id, newID)
      }
    }
  }

  func test_tieBreaks_max() {
    var heap: Heap = [
      Distinguishable(value: 1, id: 1),
      Distinguishable(value: 1, id: 2),
      Distinguishable(value: 1, id: 3),
      Distinguishable(value: 1, id: 4),
      Distinguishable(value: 1, id: 5),
    ]
    while !heap.isEmpty {
      let oldID = heap.max!.id
      let newID = 10 * oldID
      print(heap.unordered)
      let old = heap.replaceMax(with: Distinguishable(value: 1, id: newID))
      expectEqual(old.id, oldID)
      expectEqual(heap.max?.id, 10 * oldID)
      expectNotNil(heap.removeMax()) { max in
        expectEqual(max.id, newID)
      }
    }
  }

  func test_removeAll_noneRemoved() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("seed", in: 0 ..< 10) { seed in
        var rng = RepeatableRandomNumberGenerator(seed: seed)
        let input = (0 ..< count).shuffled(using: &rng)
        var heap = Heap(input)
        heap.removeAll { _ in false }
        let expected = Array(0 ..< count)
        expectEqualElements(heap.itemsInAscendingOrder(), expected)
      }
    }
  }
    
  func test_removeAll_allRemoved() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("seed", in: 0 ..< 10) { seed in
        var rng = RepeatableRandomNumberGenerator(seed: seed)
        let input = (0 ..< count).shuffled(using: &rng)
        var heap = Heap(input)
        heap.removeAll { _ in true }
        expectTrue(heap.isEmpty)
      }
    }
  }
    
  func test_removeAll_removeEvenNumbers() {
    withEvery("count", in: 0 ..< 20) { count in
      withEvery("seed", in: 0 ..< 10) { seed in
        var rng = RepeatableRandomNumberGenerator(seed: seed)
        let input = (0 ..< count).shuffled(using: &rng)
        var heap = Heap(input)
        heap.removeAll { $0 % 2 == 0 }
        let expected = Array(stride(from: 1, to: count, by: 2))
        expectEqualElements(heap.itemsInAscendingOrder(), expected)
      }
    }
  }

  func test_removeAll_throw() throws {
    struct DummyError: Error {}

    try withEvery("count", in: 1 ..< 20) { count in
      try withEvery("seed", in: 0 ..< 10) { seed in
        var rng = RepeatableRandomNumberGenerator(seed: seed)
        let input = (0 ..< count).shuffled(using: &rng)
        var heap = Heap(input)
        expectThrows(
          try heap.removeAll { v in
            if v == count / 2 {
              throw DummyError()
            }
            return v % 2 == 0
          }
        ) { error in
          expectTrue(error is DummyError)
        }
        // Throwing halfway through `removeAll` is expected to reorder items,
        // but not remove any.
        expectEqualElements(heap.itemsInAscendingOrder(), 0 ..< count)
      }
    }
  }
}
