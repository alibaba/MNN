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

import CollectionsBenchmark
import OrderedCollections

extension Benchmark {
  public mutating func addOrderedDictionaryBenchmarks() {
    self.add(
      title: "OrderedDictionary<Int, Int> init(uniqueKeysWithValues:)",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      return { timer in
        blackHole(OrderedDictionary(uniqueKeysWithValues: keysAndValues))
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> init(uncheckedUniqueKeysWithValues:)",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      return { timer in
        blackHole(
          OrderedDictionary(uncheckedUniqueKeysWithValues: keysAndValues))
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> init(uncheckedUniqueKeys:values:)",
      input: [Int].self
    ) { input in
      let values = input.map { 2 * $0 }
      return { timer in
        blackHole(
          OrderedDictionary(uncheckedUniqueKeys: input, values: values))
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> sequential iteration",
      input: [Int].self
    ) { input in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input, values: input.map { 2 * $0 })
      return { timer in
        for item in d {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int>.Keys sequential iteration",
      input: [Int].self
    ) { input in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input, values: input.map { 2 * $0 })
      return { timer in
        for item in d.keys {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int>.Values sequential iteration",
      input: [Int].self
    ) { input in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input, values: input.map { 2 * $0 })
      return { timer in
        for item in d.values {
          blackHole(item)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> sequential iteration, indices",
      input: [Int].self
    ) { input in
      let d = OrderedDictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      return { timer in
        for i in d.elements.indices {
          blackHole(d.elements[i])
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> indexing subscript",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
      let indices = lookups.map { d.index(forKey: $0)! }
      return { timer in
        for i in indices {
          blackHole(d.elements[indices[i]]) // uses `elements` random-access collection view
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, successful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input, values: input.map { 2 * $0 })
      return { timer in
        for i in lookups {
          precondition(d[i] == 2 * i)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, unsuccessful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input, values: input.map { 2 * $0 })
      let c = input.count
      return { timer in
        for i in lookups {
          precondition(d[i + c] == nil)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, noop setter",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[i + c] = nil
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, set existing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i] = 0
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, _modify",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i]! *= 2
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.addSimple(
      title: "OrderedDictionary<Int, Int> subscript, append, unique",
      input: [Int].self
    ) { input in
      var d: OrderedDictionary<Int, Int> = [:]
      for i in input {
        d[i] = 2 * i
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.addSimple(
      title: "OrderedDictionary<Int, Int> subscript, append, shared",
      input: [Int].self
    ) { input in
      var d: OrderedDictionary<Int, Int> = [:]
      for i in input {
        let copy = d
        d[i] = 2 * i
        blackHole((copy, d))
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.addSimple(
      title: "OrderedDictionary<Int, Int> subscript, append, reserving capacity",
      input: [Int].self
    ) { input in
      var d: OrderedDictionary<Int, Int> = [:]
      d.reserveCapacity(input.count)
      for i in input {
        d[i] = 2 * i
      }
      precondition(d.count == input.count)
      blackHole(d)
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, remove existing, unique",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i] = nil
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, remove existing, shared",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(uniqueKeysWithValues: input.lazy.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            let copy = d
            d[i] = nil
            blackHole((copy, d))
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> subscript, remove missing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[i + c] = nil
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> defaulted subscript, successful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
      return { timer in
        for i in lookups {
          precondition(d[i, default: -1] != -1)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> defaulted subscript, unsuccessful lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
      return { timer in
        let c = d.count
        for i in lookups {
          precondition(d[i + c, default: -1] == -1)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> defaulted subscript, _modify existing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d[i, default: -1] *= 2
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> defaulted subscript, _modify missing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        let c = input.count
        timer.measure {
          for i in lookups {
            d[c + i, default: -1] *= 2
          }
        }
        precondition(d.count == 2 * input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> successful index(forKey:)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input,
        values: input.map { 2 * $0 })
      return { timer in
        for i in lookups {
          precondition(d.index(forKey: i) != nil)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> unsuccessful index(forKey:)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let d = OrderedDictionary(
        uncheckedUniqueKeys: input,
        values: input.map { 2 * $0 })
      return { timer in
        for i in lookups {
          precondition(d.index(forKey: lookups.count + i) == nil)
        }
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> updateValue(_:forKey:), existing",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d.updateValue(0, forKey: i)
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> updateValue(_:forKey:), append",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            d.updateValue(0, forKey: input.count + i)
          }
        }
        precondition(d.count == 2 * input.count)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> random swaps",
      input: [Int].self
    ) { input in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          var v = 0
          for i in input {
            d.swapAt(i, v)
            v += 1
          }
        }
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> partitioning around middle",
      input: [Int].self
    ) { input in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        let pivot = input.count / 2
        timer.measure {
          let r = d.partition(by: { $0.key >= pivot })
          precondition(r == pivot)
        }
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> sort",
      input: [Int].self
    ) { input in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          d.sort()
        }
        precondition(d.keys.elementsEqual(0 ..< input.count))
        precondition(d.values.elementsEqual((0 ..< input.count).lazy.map { 2 * $0 }))
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> removeLast",
      input: Int.self
    ) { size in
      return { timer in
        var d = OrderedDictionary(uncheckedUniqueKeys: 0 ..< size,
                                  values: size ..< 2 * size)
        timer.measure {
          for _ in 0 ..< size {
            d.removeLast()
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> removeFirst",
      input: Int.self
    ) { size in
      return { timer in
        var d = OrderedDictionary(uncheckedUniqueKeys: 0 ..< size,
                                  values: size ..< 2 * size)
        timer.measure {
          for _ in 0 ..< size {
            d.removeFirst()
          }
        }
        precondition(d.isEmpty)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> random removals (offset-based)",
      input: Insertions.self
    ) { insertions in
      return { timer in
        let insertions = insertions.values
        var d = OrderedDictionary(uncheckedUniqueKeys: 0 ..< insertions.count,
                                  values: insertions.count ..< 2 * insertions.count)
        timer.measure {
          for i in stride(from: insertions.count, to: 0, by: -1) {
            d.remove(at: insertions[i - 1])
          }
        }
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> random removals (existing keys)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { ($0, 2 * $0) })
        timer.measure {
          for i in lookups {
            precondition(d.removeValue(forKey: i) != nil)
          }
        }
        precondition(d.count == 0)
        blackHole(d)
      }
    }

    self.add(
      title: "OrderedDictionary<Int, Int> random removals (missing keys)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        let c = input.count
        var d = OrderedDictionary(
          uncheckedUniqueKeysWithValues: input.map { (c + $0, 2 * $0) })
        timer.measure {
          for i in lookups {
            precondition(d.removeValue(forKey: i) == nil)
          }
        }
        precondition(d.count == input.count)
        blackHole(d)
      }
    }
    
    self.add(
      title: "OrderedDictionary<Int, Int> equality, unique",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      let right = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "OrderedDictionary<Int, Int> equality, shared",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      let right = left
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "OrderedDictionary<Int, Int>.Values equality, unique",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      let right = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      return { timer in
        timer.measure {
          precondition(left.values == right.values)
        }
      }
    }
    
    self.add(
      title: "OrderedDictionary<Int, Int>.Values equality, shared",
      input: [Int].self
    ) { input in
      let keysAndValues = input.map { ($0, 2 * $0) }
      let left = OrderedDictionary(uniqueKeysWithValues: keysAndValues)
      let right = left
      return { timer in
        timer.measure {
          precondition(left.values == right.values)
        }
      }
    }
    
  }
}
