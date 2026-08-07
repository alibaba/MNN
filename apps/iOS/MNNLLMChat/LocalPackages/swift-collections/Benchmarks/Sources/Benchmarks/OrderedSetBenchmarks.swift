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
  public mutating func addOrderedSetBenchmarks() {
    self.addSimple(
      title: "OrderedSet<Int> init from range",
      input: Int.self
    ) { size in
      blackHole(OrderedSet(0 ..< size))
    }

    self.addSimple(
      title: "OrderedSet<Int> init from unsafe buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        blackHole(OrderedSet(buffer))
      }
    }

    self.addSimple(
      title: "OrderedSet<Int> init(uncheckedUniqueElements:) from range",
      input: Int.self
    ) { size in
      blackHole(OrderedSet(0 ..< size))
    }

    self.add(
      title: "OrderedSet<Int> random-access offset lookups",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = OrderedSet(input)
      return { timer in
        for i in lookups {
          blackHole(set[i])
        }
      }
    }

    self.add(
      title: "OrderedSet<Int> sequential iteration",
      input: [Int].self
    ) { input in
      let set = OrderedSet(input)
      return { timer in
        for i in set {
          blackHole(i)
        }
      }
    }

    self.add(
      title: "OrderedSet<Int> successful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = OrderedSet(input)
      return { timer in
        for i in lookups {
          precondition(set.contains(i))
        }
      }
    }

    self.add(
      title: "OrderedSet<Int> unsuccessful contains",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let set = OrderedSet(input)
      let lookups = lookups.map { $0 + input.count }
      return { timer in
        for i in lookups {
          precondition(!set.contains(i))
        }
      }
    }

    self.add(
      title: "OrderedSet<Int> random swaps",
      input: [Int].self
    ) { input in
      return { timer in
        var set = OrderedSet(0 ..< input.count)
        timer.measure {
          for i in input.indices {
            set.swapAt(i, input[i])
          }
        }
        blackHole(set)
      }
    }

    self.add(
      title: "OrderedSet<Int> partitioning around middle",
      input: [Int].self
    ) { input in
      return { timer in
        let pivot = input.count / 2
        var set = OrderedSet(input)
        timer.measure {
          let r = set.partition(by: { $0 >= pivot })
          precondition(r == pivot)
        }
        blackHole(set)
      }
    }

    self.add(
      title: "OrderedSet<Int> sort",
      input: [Int].self
    ) { input in
      return { timer in
        var set = OrderedSet(input)
        timer.measure {
          set.sort()
        }
        precondition(set.elementsEqual(0 ..< input.count))
      }
    }

    self.addSimple(
      title: "OrderedSet<Int> append",
      input: [Int].self
    ) { input in
      var set: OrderedSet<Int> = []
      for i in input {
        set.append(i)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "OrderedSet<Int> append, reserving capacity",
      input: [Int].self
    ) { input in
      var set: OrderedSet<Int> = []
      set.reserveCapacity(input.count)
      for i in input {
        set.append(i)
      }
      precondition(set.count == input.count)
      blackHole(set)
    }

    self.addSimple(
      title: "OrderedSet<Int> prepend",
      input: [Int].self
    ) { input in
      var set: OrderedSet<Int> = []
      for i in input {
        _ = set.insert(i, at: 0)
      }
      blackHole(set)
    }

    self.addSimple(
      title: "OrderedSet<Int> prepend, reserving capacity",
      input: [Int].self
    ) { input in
      var set: OrderedSet<Int> = []
      set.reserveCapacity(input.count)
      for i in input {
        _ = set.insert(i, at: 0)
      }
      blackHole(set)
    }

    self.add(
      title: "OrderedSet<Int> random insertions, reserving capacity",
      input: Insertions.self
    ) { insertions in
      return { timer in
        let insertions = insertions.values
        var set: OrderedSet<Int> = []
        set.reserveCapacity(insertions.count)
        timer.measure {
          for i in insertions.indices {
            _ = set.insert(i, at: insertions[i])
          }
        }
        blackHole(set)
      }
    }

    self.add(
      title: "OrderedSet<Int> remove",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        var set = OrderedSet(input)
        timer.measure {
          for i in removals {
            set.remove(i)
          }
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    self.add(
      title: "OrderedSet<Int> removeLast",
      input: Int.self
    ) { size in
      return { timer in
        var set = OrderedSet(0 ..< size)
        timer.measure {
          for _ in 0 ..< size {
            set.removeLast()
          }
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    self.add(
      title: "OrderedSet<Int> removeFirst",
      input: Int.self
    ) { size in
      return { timer in
        var set = OrderedSet(0 ..< size)
        timer.measure {
          for _ in 0 ..< size {
            set.removeFirst()
          }
        }
        precondition(set.isEmpty)
        blackHole(set)
      }
    }

    if #available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *) {
      self.add(
        title: "OrderedSet<Int> diff computation",
        input: ([Int], [Int]).self
      ) { pa, pb in
        let a = OrderedSet(pa)
        let b = OrderedSet(pb)
        return { timer in
          timer.measure {
            blackHole(b.difference(from: a))
          }
        }
      }

      self.add(
        title: "OrderedSet<Int> diff application",
        input: ([Int], [Int]).self
      ) { a, b in
        let d = OrderedSet(b).difference(from: OrderedSet(a))
        return { timer in
          timer.measure {
            blackHole(a.applying(d))
          }
        }
      }
    }

    for percent in [0, 25, 50, 75, 100] {
      self.add(
        title: "OrderedSet<Int> filter keeping \(percent)%",
        input: [Int].self
      ) { input in
        let set = OrderedSet(input)
        return { timer in
          var r: OrderedSet<Int>!
          timer.measure {
            r = set.filter { $0 % 100 < percent }
          }
          let div = input.count / 100
          let rem = input.count % 100
          precondition(r.count == percent * div + Swift.min(rem, percent))
          blackHole(r)
        }
      }
    }

    let overlaps: [(String, (Int) -> Int)] = [
      ("0%",   { c in c }),
      ("25%",  { c in 3 * c / 4 }),
      ("50%",  { c in c / 2 }),
      ("75%",  { c in c / 4 }),
      ("100%", { c in 0 }),
    ]

    // SetAlgebra operations with Self
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> union with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            blackHole(a.union(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> intersection with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            blackHole(a.intersection(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> symmetricDifference with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            blackHole(a.symmetricDifference(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> subtracting Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            blackHole(a.subtracting(b))
          }
        }
      }
    }

    // SetAlgebra operations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> union with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.union(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> intersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.intersection(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> symmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.symmetricDifference(b))
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> subtracting Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let a = OrderedSet(input)
          let b = Array(start ..< start + input.count)
          return { timer in
            blackHole(a.subtracting(b))
          }
        }
      }
    }

    // SetAlgebra mutations with Self
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formUnion with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formUnion(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formIntersection with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formIntersection(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formSymmetricDifference with Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formSymmetricDifference(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> subtract Self (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = OrderedSet(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.subtract(b)
            }
            blackHole(a)
          }
        }
      }
    }

    // SetAlgebra mutations with Array
    do {
      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formUnion with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formUnion(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formIntersection with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formIntersection(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> formSymmetricDifference with Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.formSymmetricDifference(b)
            }
            blackHole(a)
          }
        }
      }

      for (percentage, start) in overlaps {
        self.add(
          title: "OrderedSet<Int> subtract Array (\(percentage) overlap)",
          input: [Int].self
        ) { input in
          let start = start(input.count)
          let b = Array(start ..< start + input.count)
          return { timer in
            var a = OrderedSet(input)
            timer.measure {
              a.subtract(b)
            }
            blackHole(a)
          }
        }
      }
    }
    
    self.add(
      title: "OrderedSet<Int> equality, unique",
      input: Int.self
    ) { size in
      return { timer in
        let left = OrderedSet(0 ..< size)
        let right = OrderedSet(0 ..< size)
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "OrderedSet<Int> equality, shared",
      input: Int.self
    ) { size in
      return { timer in
        let left = OrderedSet(0 ..< size)
        let right = left
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "OrderedSet<Int>.SubSequence equality, unique",
      input: Int.self
    ) { size in
      return { timer in
        let left = OrderedSet(0 ..< size)[0 ..< size]
        let right = OrderedSet(0 ..< size)[0 ..< size]
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "OrderedSet<Int>.SubSequence equality, shared",
      input: Int.self
    ) { size in
      return { timer in
        let left = OrderedSet(0 ..< size)[0 ..< size]
        let right = left
        timer.measure {
          precondition(left == right)
        }
      }
    }

  }
}
