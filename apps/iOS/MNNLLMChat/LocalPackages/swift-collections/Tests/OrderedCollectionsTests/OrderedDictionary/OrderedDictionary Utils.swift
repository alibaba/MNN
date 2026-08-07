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
import Collections
#else
import _CollectionsTestSupport
import OrderedCollections
#endif

extension LifetimeTracker {
  func orderedDictionary<Keys: Sequence>(
    keys: Keys
  ) -> (
    dictionary: OrderedDictionary<LifetimeTracked<Int>, LifetimeTracked<Int>>,
    expected: [(key: LifetimeTracked<Int>, value: LifetimeTracked<Int>)]
  )
  where Keys.Element == Int
  {
    let k = Array(keys)
    let keys = self.instances(for: k)
    let values = self.instances(for: k.map { $0 + 100 })
    let dictionary = OrderedDictionary(uniqueKeys: keys, values: values)
    return (dictionary, (0 ..< k.count).map { (key: keys[$0], value: values[$0]) })
  }
}
