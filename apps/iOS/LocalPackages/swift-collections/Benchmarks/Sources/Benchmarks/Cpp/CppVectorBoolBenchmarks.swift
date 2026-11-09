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

internal class CppVectorBool {
  var ptr: UnsafeMutableRawPointer?

  init(repeating value: Bool, count: Int) {
    self.ptr = cpp_vector_bool_create_repeating(count, value)
  }

  init(count: Int, trueBits: ArraySlice<Int>) {
    self.ptr = cpp_vector_bool_create_repeating(count, false)
    trueBits.withUnsafeBufferPointer { buffer in
      cpp_vector_bool_set_indices_subscript(
        self.ptr,
        buffer.baseAddress,
        buffer.count)
    }
    withExtendedLifetime(self) {}
  }

  deinit {
    destroy()
  }

  func destroy() {
    if let ptr = ptr {
      cpp_vector_bool_destroy(ptr)
    }
    ptr = nil
  }
}

extension Array where Element == Bool {
  init<S: Sequence>(count: Int, trueBits: S) where S.Element == Int {
    self.init(repeating: false, count: count)
    for index in trueBits {
      self[index] = true
    }
  }
}

extension Benchmark {
  internal mutating func _addCppVectorBoolBenchmarks() {
    self.addSimple(
      title: "std::vector<bool> create from integer buffer (subscript)",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let v = CppVectorBool(repeating: false, count: input.count)
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_bool_set_indices_subscript(
          v.ptr,
          buffer.baseAddress,
          trueCount)
      }
      v.destroy()
    }

    self.addSimple(
      title: "std::vector<bool> create from integer buffer (at)",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let v = CppVectorBool(repeating: false, count: input.count)
      input.withUnsafeBufferPointer { buffer in
        cpp_vector_bool_set_indices_at(
          v.ptr,
          buffer.baseAddress,
          trueCount)
      }
      v.destroy()
    }

    self.add(
      title: "std::vector<bool> const_iterator",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
      return { timer in
        cpp_vector_bool_iterate(identity(v.ptr))
      }
    }

    self.add(
      title: "std::vector<bool> find true bits",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
      return { timer in
        precondition(
          cpp_vector_bool_find_true_bits(identity(v.ptr)) == trueCount)
      }
    }

    self.add(
      title: "std::vector<bool> count true bits",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
      return { timer in
        precondition(
          cpp_vector_bool_count_true_bits(identity(v.ptr)) == trueCount)
      }
    }

    self.add(
      title: "std::vector<bool> random-access offset lookups (subscript)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let trueCount = input.count / 2
      let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_lookups_subscript(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> random-access offset lookups (at)",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let trueCount = input.count / 2
      let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_lookups_at(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> set bits to true (subscript)",
      input: [Int].self
    ) { input in
      let v = CppVectorBool(repeating: false, count: input.count)
      return { timer in
        input.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_set_indices_subscript(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> set bits to true (at)",
      input: [Int].self
    ) { input in
      let v = CppVectorBool(repeating: false, count: input.count)
      return { timer in
        input.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_set_indices_at(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> set bits to false (subscript)",
      input: [Int].self
    ) { input in
      let v = CppVectorBool(repeating: true, count: input.count)
      return { timer in
        input.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_reset_indices_subscript(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> set bits to false (at)",
      input: [Int].self
    ) { input in
      let v = CppVectorBool(repeating: true, count: input.count)
      return { timer in
        input.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_reset_indices_at(v.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }

    self.add(
      title: "std::vector<bool> push_back",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let bools = Array<Bool>(count: input.count, trueBits: input[..<trueCount])

      return { timer in
        bools.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_push_back(buffer.baseAddress, buffer.count, /*reserve*/false)
        }
      }
    }

    self.add(
      title: "std::vector<bool> push_back, reserving capacity",
      input: [Int].self
    ) { input in
      let trueCount = input.count / 2
      let bools = Array<Bool>(count: input.count, trueBits: input[..<trueCount])

      return { timer in
        bools.withUnsafeBufferPointer { buffer in
          cpp_vector_bool_push_back(buffer.baseAddress, buffer.count, /*reserve*/true)
        }
      }
    }

    self.add(
      title: "std::vector<bool> pop_back",
      input: [Int].self
    ) { input in
      return { timer in
        let trueCount = input.count / 2
        let v = CppVectorBool(count: input.count, trueBits: input[..<trueCount])
        timer.measure {
          cpp_vector_bool_pop_back(v.ptr, input.count)
        }
        v.destroy()
      }
    }
  }
}
