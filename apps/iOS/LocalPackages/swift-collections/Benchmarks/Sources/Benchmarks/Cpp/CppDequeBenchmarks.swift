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

internal class CppDeque {
  var ptr: UnsafeMutableRawPointer?

  init(_ input: [Int]) {
    self.ptr = input.withUnsafeBufferPointer { buffer in
      cpp_deque_create(buffer.baseAddress, buffer.count)
    }
  }

  deinit {
    destroy()
  }

  func destroy() {
    if let ptr = ptr {
      cpp_deque_destroy(ptr)
    }
    ptr = nil
  }
}

extension Benchmark {
  internal mutating func _addCppDequeBenchmarks() {
    self.addSimple(
      title: "std::deque<intptr_t> push_back from integer range",
      input: Int.self
    ) { count in
      cpp_deque_from_int_range(count)
    }

    self.addSimple(
      title: "std::deque<intptr_t> constructor from buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_deque_from_int_buffer(buffer.baseAddress, buffer.count)
      }
    }

    self.add(
      title: "std::deque<intptr_t> sequential iteration",
      input: [Int].self
    ) { input in
      let deque = CppDeque(input)
      return { timer in
        cpp_deque_iterate(deque.ptr)
      }
    }

    self.add(
      title: "std::deque<intptr_t> random-access offset lookups (operator [])",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let vector = CppDeque(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_deque_lookups_subscript(vector.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::deque<intptr_t> at, random offsets",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = CppDeque(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_deque_lookups_at(deque.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.addSimple(
      title: "std::deque<intptr_t> push_back",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_deque_append_integers(buffer.baseAddress, buffer.count)
      }
    }

    self.addSimple(
      title: "std::deque<intptr_t> push_front",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_deque_prepend_integers(buffer.baseAddress, buffer.count)
      }
    }

    self.addSimple(
      title: "std::deque<intptr_t> random insertions",
      input: Insertions.self
    ) { insertions in
      insertions.values.withUnsafeBufferPointer { buffer in
        cpp_deque_random_insertions(buffer.baseAddress, buffer.count)
      }
    }

    self.add(
      title: "std::deque<intptr_t> pop_back",
      input: Int.self
    ) { size in
      return { timer in
        let deque = CppDeque(Array(0 ..< size))
        timer.measure {
          cpp_deque_pop_back(deque.ptr)
        }
        deque.destroy()
      }
    }

    self.add(
      title: "std::deque<intptr_t> pop_front",
      input: Int.self
    ) { size in
      return { timer in
        let deque = CppDeque(Array(0 ..< size))
        timer.measure {
          cpp_deque_pop_front(deque.ptr)
        }
        deque.destroy()
      }
    }

    self.add(
      title: "std::deque<intptr_t> random removals",
      input: Insertions.self
    ) { insertions in
      let removals = Array(insertions.values.reversed())
      return { timer in
        let deque = CppDeque(Array(0 ..< removals.count))
        timer.measure {
          removals.withUnsafeBufferPointer { buffer in
            cpp_deque_random_removals(deque.ptr, buffer.baseAddress, buffer.count)
          }
        }
        deque.destroy()
      }
    }

    self.add(
      title: "std::deque<intptr_t> sort",
      input: [Int].self
    ) { input in
      return { timer in
        let deque = CppDeque(input)
        timer.measure {
          cpp_deque_sort(deque.ptr)
        }
        deque.destroy()
      }
    }
  }
}
