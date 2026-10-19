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

internal class CppUnorderedMap {
  var ptr: UnsafeMutableRawPointer?

  init(_ input: [Int]) {
    self.ptr = input.withUnsafeBufferPointer { buffer in
      cpp_unordered_map_create(buffer.baseAddress, buffer.count)
    }
  }

  deinit {
    destroy()
  }

  func destroy() {
    if let ptr = ptr {
      cpp_unordered_map_destroy(ptr)
    }
    ptr = nil
  }
}

extension Benchmark {
  internal mutating func _addCppUnorderedMapBenchmarks() {
    self.addSimple(
      title: "std::unordered_map<intptr_t, intptr_t> insert from integer range",
      input: Int.self
    ) { count in
      cpp_unordered_map_from_int_range(count)
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> sequential iteration",
      input: [Int].self
    ) { input in
      let map = CppUnorderedMap(input)
      return { timer in
        cpp_unordered_map_iterate(map.ptr)
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> successful find",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let map = CppUnorderedMap(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_unordered_map_lookups(map.ptr, buffer.baseAddress, buffer.count, true)
        }
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> unsuccessful find",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let map = CppUnorderedMap(input)
      let lookups = lookups.map { $0 + input.count }
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_unordered_map_lookups(map.ptr, buffer.baseAddress, buffer.count, false)
        }
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> subscript, existing key",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let map = CppUnorderedMap(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_unordered_map_subscript(map.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> subscript, new key",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let map = CppUnorderedMap(input)
      let lookups = lookups.map { $0 + input.count }
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_unordered_map_subscript(map.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.addSimple(
      title: "std::unordered_map<intptr_t, intptr_t> insert",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_unordered_map_insert_integers(buffer.baseAddress, buffer.count, false)
      }
    }

    self.addSimple(
      title: "std::unordered_map<intptr_t, intptr_t> insert, reserving capacity",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_unordered_map_insert_integers(buffer.baseAddress, buffer.count, true)
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> erase existing",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        let map = CppUnorderedMap(input)
        timer.measure {
          removals.withUnsafeBufferPointer { buffer in
            cpp_unordered_map_removals(map.ptr, buffer.baseAddress, buffer.count)
          }
        }
        map.destroy()
      }
    }

    self.add(
      title: "std::unordered_map<intptr_t, intptr_t> erase missing",
      input: ([Int], [Int]).self
    ) { input, removals in
      return { timer in
        let map = CppUnorderedMap(input.map { input.count + $0 })
        timer.measure {
          removals.withUnsafeBufferPointer { buffer in
            cpp_unordered_map_removals(map.ptr, buffer.baseAddress, buffer.count)
          }
        }
        map.destroy()
      }
    }
  }
}
