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
@_spi(Testing) import DequeModule

extension Deque {
  @inline(__always)
  internal init(discontiguous contents: [Element]) {
    self.init(_capacity: contents.count,
              startSlot: contents.count / 4,
              contents: contents)
  }
}

extension Benchmark {
  public mutating func addDequeBenchmarks() {
    self.addSimple(
      title: "Deque<Int> init from range",
      input: Int.self
    ) { size in
      blackHole(Deque(0 ..< size))
    }

    self.addSimple(
      title: "Deque<Int> init from unsafe buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        blackHole(Deque(buffer))
      }
    }

    self.add(
      title: "Deque<Int> sequential iteration (contiguous, iterator)",
      input: [Int].self
    ) { input in
      let deque = Deque(input)
      return { timer in
        for i in deque {
          blackHole(i)
        }
      }
    }

    self.add(
      title: "Deque<Int> sequential iteration (discontiguous, iterator)",
      input: [Int].self
    ) { input in
      let deque = Deque(discontiguous: input)
      return { timer in
        for i in deque {
          blackHole(i)
        }
      }
    }

    self.add(
      title: "Deque<Int> sequential iteration (contiguous, indices)",
      input: [Int].self
    ) { input in
      let deque = Deque(input)
      return { timer in
        for i in deque.indices {
          blackHole(deque[i])
        }
      }
    }

    self.add(
      title: "Deque<Int> sequential iteration (discontiguous, indices)",
      input: [Int].self
    ) { input in
      let deque = Deque(discontiguous: input)
      return { timer in
        for i in deque.indices {
          blackHole(deque[i])
        }
      }
    }

    self.add(
      title: "Deque<Int> subscript get, random offsets (contiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(input)
      return { timer in
        for i in lookups {
          blackHole(deque[i])
        }
      }
    }

    self.add(
      title: "Deque<Int> subscript get, random offsets (discontiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(discontiguous: input)
      return { timer in
        for i in lookups {
          blackHole(deque[i])
        }
      }
    }

    self.add(
      title: "Deque<Int> successful contains (contiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(input)
      return { timer in
        for i in lookups {
          precondition(deque.contains(i))
        }
      }
    }

    self.add(
      title: "Deque<Int> successful contains (discontiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(discontiguous: input)
      return { timer in
        for i in lookups {
          precondition(deque.contains(i))
        }
      }
    }

    self.add(
      title: "Deque<Int> unsuccessful contains (contiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(input)
      return { timer in
        let c = input.count
        for i in lookups {
          precondition(!deque.contains(i + c))
        }
      }
    }

    self.add(
      title: "Deque<Int> unsuccessful contains (discontiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = Deque(discontiguous: input)
      return { timer in
        let c = input.count
        for i in lookups {
          precondition(!deque.contains(i + c))
        }
      }
    }

    self.add(
      title: "Deque<Int> mutate through subscript (contiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var deque = Deque(input)
        timer.measure {
          var v = 0
          for i in lookups {
            deque[i] = v
            v += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> mutate through subscript (discontiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var deque = Deque(discontiguous: input)
        timer.measure {
          var v = 0
          for i in lookups {
            deque[i] = v
            v += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> modify through subscript (contiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var deque = Deque(input)
        timer.measure {
          for i in lookups {
            deque[i] += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> modify through subscript (discontiguous)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      return { timer in
        var deque = Deque(discontiguous: input)
        timer.measure {
          for i in lookups {
            deque[i] += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> random swaps (contiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        var deque = Deque(0 ..< input.count)
        timer.measure {
          var v = 0
          for i in input {
            deque.swapAt(i, v)
            v += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> random swaps (discontiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        var deque = Deque(discontiguous: input)
        timer.measure {
          var v = 0
          for i in input {
            deque.swapAt(i, v)
            v += 1
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> partitioning around middle (contiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        let pivot = input.count / 2
        var deque = Deque(input)
        timer.measure {
          let r = deque.partition(by: { $0 >= pivot })
          precondition(r == pivot)
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> partitioning around middle (discontiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        let pivot = input.count / 2
        var deque = Deque(discontiguous: input)
        timer.measure {
          let r = deque.partition(by: { $0 >= pivot })
          precondition(r == pivot)
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> sort (contiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        var deque = Deque(input)
        timer.measure {
          deque.sort()
        }
        precondition(deque.elementsEqual(0 ..< input.count))
      }
    }

    self.add(
      title: "Deque<Int> sort (discontiguous)",
      input: [Int].self
    ) { input in
      return { timer in
        var deque = Deque(discontiguous: input)
        timer.measure {
          deque.sort()
        }
        precondition(deque.elementsEqual(0 ..< input.count))
      }
    }

    self.addSimple(
      title: "Deque<Int> append from range",
      input: Int.self
    ) { count in
      var deque: Deque<Int> = []
      for i in 0 ..< count {
        deque.append(i)
      }
      precondition(deque.count == count)
      blackHole(deque)
    }

    self.addSimple(
      title: "Deque<Int> append",
      input: [Int].self
    ) { input in
      var deque: Deque<Int> = []
      for i in input {
        deque.append(i)
      }
      precondition(deque.count == input.count)
      blackHole(deque)
    }

    self.addSimple(
      title: "Deque<Int> append, reserving capacity",
      input: [Int].self
    ) { input in
      var deque: Deque<Int> = []
      deque.reserveCapacity(input.count)
      for i in input {
        deque.append(i)
      }
      blackHole(deque)
    }

    self.addSimple(
      title: "Deque<Int> prepend",
      input: [Int].self
    ) { input in
      var deque: Deque<Int> = []
      for i in input {
        deque.prepend(i)
      }
      blackHole(deque)
    }

    self.addSimple(
      title: "Deque<Int> prepend, reserving capacity",
      input: [Int].self
    ) { input in
      var deque: Deque<Int> = []
      deque.reserveCapacity(input.count)
      for i in input {
        deque.prepend(i)
      }
      blackHole(deque)
    }

    self.addSimple(
      title: "Deque<Int> kalimba",
      input: [Int].self
    ) { input in
      blackHole(input.kalimbaOrdered2())
    }

    self.add(
      title: "Deque<Int> random insertions",
      input: Insertions.self
    ) { insertions in
      return { timer in
        let insertions = insertions.values
        var deque: Deque<Int> = []
        timer.measure {
          for i in insertions.indices {
            deque.insert(i, at: insertions[i])
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> random insertions, reserving capacity",
      input: Insertions.self
    ) { insertions in
      return { timer in
        let insertions = insertions.values
        var deque: Deque<Int> = []
        deque.reserveCapacity(insertions.count)
        timer.measure {
          for i in insertions.indices {
            deque.insert(i, at: insertions[i])
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> removeLast (contiguous)",
      input: Int.self
    ) { size in
      return { timer in
        var deque = Deque(0 ..< size)
        timer.measure {
          for _ in 0 ..< size {
            deque.removeLast()
          }
        }
        precondition(deque.isEmpty)
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> removeLast (discontiguous)",
      input: Int.self
    ) { size in
      return { timer in
        var deque = Deque(discontiguous: Array(0 ..< size))
        timer.measure {
          for _ in 0 ..< size {
            deque.removeLast()
          }
        }
        precondition(deque.isEmpty)
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> removeFirst (contiguous)",
      input: Int.self
    ) { size in
      return { timer in
        var deque = Deque(0 ..< size)
        timer.measure {
          for _ in 0 ..< size {
            deque.removeFirst()
          }
        }
        precondition(deque.isEmpty)
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> removeFirst (discontiguous)",
      input: Int.self
    ) { size in
      return { timer in
        var deque = Deque(discontiguous: Array(0 ..< size))
        timer.measure {
          for _ in 0 ..< size {
            deque.removeFirst()
          }
        }
        precondition(deque.isEmpty)
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> random removals (contiguous)",
      input: Insertions.self
    ) { insertions in
      let removals = Array(insertions.values.reversed())
      return { timer in
        var deque = Deque(0 ..< removals.count)
        timer.measure {
          for i in removals {
            deque.remove(at: i)
          }
        }
        blackHole(deque)
      }
    }

    self.add(
      title: "Deque<Int> random removals (discontiguous)",
      input: Insertions.self
    ) { insertions in
      let removals = Array(insertions.values.reversed())
      return { timer in
        let size = removals.count
        var deque = Deque(discontiguous: Array(0 ..< size))
        timer.measure {
          for i in removals {
            deque.remove(at: i)
          }
        }
        blackHole(deque)
      }
    }
    
    self.add(
      title: "Deque<Int> equality, unique",
      input: Int.self
    ) { size in
      let left = Deque(0 ..< size)
      let right = Deque(0 ..< size)
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
    self.add(
      title: "Deque<Int> equality, shared",
      input: Int.self
    ) { size in
      let left = Deque(0 ..< size)
      let right = Deque(0 ..< size)
      return { timer in
        timer.measure {
          precondition(left == right)
        }
      }
    }
    
  }
}
