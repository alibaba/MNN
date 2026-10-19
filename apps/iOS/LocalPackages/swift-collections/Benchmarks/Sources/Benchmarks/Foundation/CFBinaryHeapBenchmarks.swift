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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Foundation

extension CFBinaryHeap {
  internal static func _create(capacity: Int) -> CFBinaryHeap {
    var callbacks = CFBinaryHeapCallBacks(
      version: 0,
      retain: nil,
      release: nil,
      copyDescription: { value in
        let result = "\(Int(bitPattern: value))" as NSString
        return Unmanaged.passRetained(result)
      },
      compare: { left, right, context in
        let left = Int(bitPattern: left)
        let right = Int(bitPattern: right)
        if left == right { return .compareEqualTo }
        if left < right { return .compareLessThan }
        return .compareGreaterThan
      })
    return CFBinaryHeapCreate(kCFAllocatorDefault, capacity, &callbacks, nil)
  }
}
#endif

extension Benchmark {
  internal mutating func _addCFBinaryHeapBenchmarks() {
    self.add(
      title: "CFBinaryHeapAddValue",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      return { timer in
        let heap = CFBinaryHeap._create(capacity: 0)
        timer.measure {
          for value in input {
            CFBinaryHeapAddValue(heap, UnsafeRawPointer(bitPattern: value))
          }
        }
        blackHole(heap)
      }
#else
      // CFBinaryHeap isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBinaryHeapAddValue, reserving capacity",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      return { timer in
        let heap = CFBinaryHeap._create(capacity: input.count)
        timer.measure {
          for value in input {
            CFBinaryHeapAddValue(heap, UnsafeRawPointer(bitPattern: value))
          }
        }
        blackHole(heap)
      }
#else
      return nil
#endif
    }

    self.add(
      title: "CFBinaryHeapRemoveMinimumValue",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      return { timer in
        let heap = CFBinaryHeap._create(capacity: input.count)
        for value in input {
          CFBinaryHeapAddValue(heap, UnsafeRawPointer(bitPattern: value))
        }
        timer.measure {
          for _ in 0 ..< input.count {
            blackHole(CFBinaryHeapGetMinimum(heap))
            CFBinaryHeapRemoveMinimumValue(heap)
          }
        }
        blackHole(heap)
      }
#else
      return nil
#endif
    }
  }
}
