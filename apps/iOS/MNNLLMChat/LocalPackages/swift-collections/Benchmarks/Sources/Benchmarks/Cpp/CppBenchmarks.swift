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

extension Benchmark {
  public mutating func addCppBenchmarks() {
    cpp_set_hash_fn { value in value._rawHashValue(seed: 0) }

    self.addSimple(
      title: "std::hash<intptr_t>",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_hash(buffer.baseAddress, buffer.count)
      }
    }

    self.addSimple(
      title: "custom_intptr_hash (using Swift.Hasher)",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_custom_hash(buffer.baseAddress, buffer.count)
      }
    }

    _addCppVectorBenchmarks()
    _addCppDequeBenchmarks()
    _addCppUnorderedSetBenchmarks()
    _addCppUnorderedMapBenchmarks()
    _addCppPriorityQueueBenchmarks()
    _addCppVectorBoolBenchmarks()
    _addCppMapBenchmarks()
  }
}
