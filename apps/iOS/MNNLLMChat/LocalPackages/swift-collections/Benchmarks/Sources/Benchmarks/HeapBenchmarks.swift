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
import HeapModule
import CppBenchmarks

extension Benchmark {
  public mutating func addHeapBenchmarks() {
    self.addSimple(
      title: "Heap<Int> init from range",
      input: Int.self
    ) { size in
      blackHole(Heap(0..<size))
    }

    self.addSimple(
      title: "Heap<Int> init from buffer",
      input: [Int].self
    ) { input in
      blackHole(Heap(input))
    }

    self.addSimple(
      title: "Heap<Int> insert",
      input: [Int].self
    ) { input in
      var queue = Heap<Int>()
      for i in input {
        queue.insert(i)
      }
      precondition(queue.count == input.count)
      blackHole(queue)
    }

    self.add(
      title: "Heap<Int> insert(contentsOf:)",
      input: ([Int], [Int]).self
    ) { (existing, new) in
      return { timer in
        var queue = Heap(existing)
        queue.insert(contentsOf: new)
        precondition(queue.count == existing.count + new.count)
        blackHole(queue)
      }
    }

    self.add(
      title: "Heap<Int> popMax",
      input: [Int].self
    ) { input in
      return { timer in
        var queue = Heap(input)
        timer.measure {
          while let max = queue.popMax() {
            blackHole(max)
          }
        }
        precondition(queue.isEmpty)
        blackHole(queue)
      }
    }

    self.add(
      title: "Heap<Int> popMin",
      input: [Int].self
    ) { input in
      return { timer in
        var queue = Heap(input)
        timer.measure {
          while let min = queue.popMin() {
            blackHole(min)
          }
        }
        precondition(queue.isEmpty)
        blackHole(queue)
      }
    }
  }
}
