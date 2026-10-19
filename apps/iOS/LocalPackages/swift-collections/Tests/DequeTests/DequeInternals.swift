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

#if COLLECTIONS_SINGLE_MODULE
@_spi(Testing) import Collections
#else
import _CollectionsTestSupport
@_spi(Testing) import DequeModule
#endif

internal struct DequeLayout: CustomStringConvertible {
  let capacity: Int
  let startSlot: Int
  let count: Int
  let startValue: Int

  init(capacity: Int, startSlot: Int, count: Int, startValue: Int = 0) {
    self.capacity = capacity
    self.startSlot = startSlot
    self.count = count
    self.startValue = startValue
  }

  var valueRange: Range<Int> { startValue ..< startValue + count }

  var description: String {
    var result = "DequeLayout(capacity: \(capacity), startSlot: \(startSlot), count: \(count)"
    if count > 0 {
      result += ", startValue: \(startValue)"
    }
    result += ")"
    return result
  }

  var isWrapped: Bool {
    startSlot + count > capacity
  }
}

extension Deque {
  init<C: Collection>(layout: DequeLayout, contents: C) where C.Element == Element {
    precondition(contents.count == layout.count)
    self.init(_capacity: layout.capacity, startSlot: layout.startSlot, contents: Array(contents))
  }
}

extension LifetimeTracker {
  func deque(
    with layout: DequeLayout
  ) -> (deque: Deque<LifetimeTracked<Int>>, contents: [LifetimeTracked<Int>]) {
    let contents = self.instances(for: layout.valueRange)
    let deque = Deque(layout: layout, contents: contents)
    return (deque, contents)
  }
}

func withEveryDeque<C: Collection>(
  _ label: String,
  ofCapacities capacities: C,
  startValue: Int = 0,
  file: StaticString = #file, line: UInt = #line,
  _ body: (DequeLayout) throws -> Void
) rethrows -> Void where C.Element == Int {
  // Exhaustive tests for all deque layouts of various capacities
  for capacity in capacities {
    for startSlot in 0 ..< capacity {
      for count in 0 ... capacity {
        let layout = DequeLayout(
          capacity: capacity,
          startSlot: startSlot,
          count: count,
          startValue: startValue)
        let entry = TestContext.current.push("\(label): \(layout)", file: file, line: line)
        defer { TestContext.current.pop(entry) }
        try body(layout)
      }
    }
  }
}
