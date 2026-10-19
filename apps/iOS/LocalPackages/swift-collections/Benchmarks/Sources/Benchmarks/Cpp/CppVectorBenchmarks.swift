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

internal class CppVector {
  var ptr: UnsafeMutableRawPointer?

  init(_ input: [Int]) {
    self.ptr = input.withUnsafeBufferPointer { buffer in
      cpp_vector_create(buffer.baseAddress, buffer.count)
    }
  }

  deinit {
    destroy()
  }

  func destroy() {
    if let ptr = ptr {
      cpp_vector_destroy(ptr)
    }
    ptr = nil
  }
}

extension Benchmark {
  internal mutating func _addCppVectorBenchmarks() {
    self.addSimple(
      title: "std::vector<intptr_t> push_back from integer range",
      input: Int.self
    ) { count in
      cpp_vector_from_int_range(count)
    }

    self.addSimple(
      title: "std::vector<intptr_t> constructor from buffer",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_from_int_buffer(buffer.baseAddress, buffer.count)
      }
    }

    self.add(
      title: "std::vector<intptr_t> sequential iteration",
      input: [Int].self
    ) { input in
      let vector = CppVector(input)
      return { timer in
        cpp_vector_iterate(vector.ptr)
      }
    }

    self.add(
      title: "std::vector<intptr_t> random-access offset lookups (operator [])",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let vector = CppVector(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_vector_lookups_subscript(vector.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<intptr_t> random-access offset lookups (at)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let deque = CppVector(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_vector_lookups_at(deque.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.addSimple(
      title: "std::vector<intptr_t> push_back",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_append_integers(buffer.baseAddress, buffer.count, false)
      }
    }

    self.addSimple(
      title: "std::vector<intptr_t> push_back, reserving capacity",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_append_integers(buffer.baseAddress, buffer.count, true)
      }
    }

    self.addSimple(
      title: "std::vector<intptr_t> insert at front",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_prepend_integers(buffer.baseAddress, buffer.count, false)
      }
    }

    self.addSimple(
      title: "std::vector<intptr_t> insert at front, reserving capacity",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_prepend_integers(buffer.baseAddress, buffer.count, true)
      }
    }

    self.addSimple(
      title: "std::vector<intptr_t> random insertions",
      input: Insertions.self
    ) { insertions in
      insertions.values.withUnsafeBufferPointer { buffer in
        cpp_vector_random_insertions(buffer.baseAddress, buffer.count, false)
      }
    }

    self.add(
      title: "std::vector<intptr_t> pop_back",
      input: Int.self
    ) { size in
      return { timer in
        let vector = CppVector(Array(0 ..< size))
        timer.measure {
          cpp_vector_pop_back(vector.ptr)
        }
        vector.destroy()
      }
    }

    self.add(
      title: "std::vector<intptr_t> erase first",
      input: Int.self
    ) { size in
      return { timer in
        let vector = CppVector(Array(0 ..< size))
        timer.measure {
          cpp_vector_pop_front(vector.ptr)
        }
        vector.destroy()
      }
    }

    self.add(
      title: "std::vector<intptr_t> random removals",
      input: Insertions.self
    ) { insertions in
      let removals = Array(insertions.values.reversed())
      return { timer in
        let vector = CppVector(Array(0 ..< removals.count))
        timer.measure {
          removals.withUnsafeBufferPointer { buffer in
            cpp_vector_random_removals(vector.ptr, buffer.baseAddress, buffer.count)
          }
        }
        vector.destroy()
      }
    }

    self.add(
      title: "std::vector<intptr_t> sort",
      input: [Int].self
    ) { input in
      return { timer in
        let vector = CppVector(input)
        timer.measure {
          cpp_vector_sort(vector.ptr)
        }
        vector.destroy()
      }
    }
  }
}
