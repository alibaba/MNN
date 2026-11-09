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
import CppBenchmarks

internal class CppPriorityQueue {
  var ptr: UnsafeMutableRawPointer?

  init(_ input: [Int]) {
    self.ptr = input.withUnsafeBufferPointer { buffer in
      cpp_priority_queue_create(buffer.baseAddress, buffer.count)
    }
  }

  convenience init() {
    self.init([])
  }

  deinit {
    destroy()
  }

  func destroy() {
    if let ptr = ptr {
      cpp_priority_queue_destroy(ptr)
    }
    ptr = nil
  }

  func push(_ value: Int) {
    cpp_priority_queue_push(ptr, value)
  }

  func push(_ values: [Int]) {
    values.withUnsafeBufferPointer { buffer in
      cpp_priority_queue_push_loop(ptr, buffer.baseAddress, buffer.count)
    }
  }

  func pop() -> Int {
    cpp_priority_queue_pop(ptr)
  }

  func popAll() {
    cpp_priority_queue_pop_all(ptr)
  }
}

extension Benchmark {
  internal mutating func _addCppPriorityQueueBenchmarks() {
    self.addSimple(
      title: "std::priority_queue<intptr_t> construct from buffer",
      input: [Int].self
    ) { input in
      let pq = CppPriorityQueue(input)
      blackHole(pq)
    }

    self.add(
      title: "std::priority_queue<intptr_t> push",
      input: [Int].self
    ) { input in
      return { timer in
        let pq = CppPriorityQueue()
        timer.measure {
          pq.push(input)
        }
        blackHole(pq)
        pq.destroy()
      }
    }

    self.add(
      title: "std::priority_queue<intptr_t> pop",
      input: [Int].self
    ) { input in
      return { timer in
        let pq = CppPriorityQueue(input)
        timer.measure {
          pq.popAll()
        }
        blackHole(pq)
        pq.destroy()
      }
    }
  }
}
