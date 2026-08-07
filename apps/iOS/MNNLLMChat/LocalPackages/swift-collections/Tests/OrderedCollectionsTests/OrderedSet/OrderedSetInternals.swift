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
@_spi(Testing) import OrderedCollections
#endif

struct OrderedSetLayout: Hashable, CustomStringConvertible {
  let scale: Int
  let bias: Int
  let count: Int

  var description: String {
    "OrderedSetLayout(scale: \(scale), bias: \(bias), count: \(count))"
  }
}

func withOrderedSetLayouts(
  scales: [Int],
  file: StaticString = #file,
  line: UInt = #line,
  run body: (OrderedSetLayout) throws -> Void
) rethrows {
  for scale in scales {
    for count in _interestingCounts(forScale: scale) {
      for bias in _interestingBiases(forScale: scale) {
        let layout = OrderedSetLayout(scale: scale, bias: bias, count: count)
        let entry = TestContext.current.push("layout: \(layout)")
        defer { TestContext.current.pop(entry) }
        try body(layout)
      }
    }
  }
}

func _interestingCounts(forScale scale: Int) -> [Int] {
  precondition(scale == 0 || scale >= OrderedSet<Int>._minimumScale)
  let min = OrderedSet<Int>._minimumCapacity(forScale: scale)
  let max = OrderedSet<Int>._maximumCapacity(forScale: scale)
  return [min, min + (max - min) / 2, max]
}

func _interestingBiases(forScale scale: Int) -> [Int] {
  // If we have no hash table, we can only use a bias of 0.
  if scale < OrderedSet<Int>._minimumScale { return [0] }
  let range = OrderedSet<Int>._biasRange(scale: scale)
  // For the minimum scale, try all biases.
  if scale == OrderedSet<Int>._minimumScale { return Array(range) }
  // Otherwise try 3 biases each from the start, middle, and end of range.
  var result: [Int] = []
  result += range.prefix(3)
  result += range.prefix(range.count / 2 + 1).suffix(3)
  result += range.suffix(3)
  return result
}

extension OrderedSet {
  init(layout: OrderedSetLayout, contents: [Element]) {
    self.init(_scale: layout.scale, bias: layout.bias, contents: contents)
  }
}

extension OrderedSet {
  init<C: Collection>(layout: OrderedSetLayout, contents: C)
  where C.Element == Element {
    self.init(_scale: layout.scale, bias: layout.bias, contents: contents)
  }
}

extension OrderedSet where Element == Int {
  init(layout: OrderedSetLayout) {
    self.init(layout: layout, contents: 0 ..< layout.count)
  }
}
