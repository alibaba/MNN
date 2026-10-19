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

internal class CppMap {
  var ptr: UnsafeMutableRawPointer?
  
  init(_ input: [Int]) {
    self.ptr = input.withUnsafeBufferPointer { buffer in
      cpp_map_create(buffer.baseAddress, buffer.count)
    }
  }
  
  deinit {
    destroy()
  }
  
  func destroy() {
    if let ptr = ptr {
      cpp_map_destroy(ptr)
    }
    ptr = nil
  }
}

extension Benchmark {
  internal mutating func _addCppMapBenchmarks() {
    self.addSimple(
      title: "std::map<intptr_t, intptr_t> insert",
      input: [Int].self
    ) { input in
      input.withUnsafeBufferPointer { buffer in
        cpp_map_insert_integers(buffer.baseAddress, buffer.count)
      }
    }
    
    self.add(
      title: "std::map<intptr_t, intptr_t> successful find",
      input: ([Int], [Int]).self
    ) { input, lookups in
      let map = CppMap(input)
      return { timer in
        lookups.withUnsafeBufferPointer { buffer in
          cpp_map_lookups(map.ptr, buffer.baseAddress, buffer.count)
        }
      }
    }
  }
}
