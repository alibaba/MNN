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

extension CFMutableBitVector {
  static func _create<S: Sequence>(
    count: Int,
    trueBits: S
  ) -> CFMutableBitVector where S.Element == Int {
    let bits = CFBitVectorCreateMutable(nil, count)!
    CFBitVectorSetCount(bits, count)
    for index in trueBits {
      CFBitVectorSetBitAtIndex(bits, index, 1)
    }
    return bits
  }
}
#endif

extension Benchmark {
  internal mutating func _addCFBitVectorBenchmarks() {
    self.add(
      title: "CFBitVector create from integer buffer",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      return { timer in
        let trueCount = input.count / 2
        let bits = CFMutableBitVector._create(
          count: input.count,
          trueBits: input[..<trueCount])
        blackHole(bits)
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBitVectorGetBitAtIndex (sequential iteration)",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      let trueCount = input.count / 2
      let bits = CFMutableBitVector._create(
        count: input.count,
        trueBits: input[..<trueCount])
      return { timer in
        for index in 0 ..< input.count {
          blackHole(CFBitVectorGetBitAtIndex(bits, index))
        }
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBitVectorGetBitAtIndex (random-access lookups)",
      input: ([Int], [Int]).self
    ) { input, lookups in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      let trueCount = input.count / 2
      let bits = CFMutableBitVector._create(
        count: input.count,
        trueBits: input[..<trueCount])
      return { timer in
        for index in lookups {
          blackHole(CFBitVectorGetBitAtIndex(bits, index))
        }
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBitVectorGetFirstIndexOfBit",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      let trueCount = input.count / 2
      let bits = CFMutableBitVector._create(
        count: input.count,
        trueBits: input[..<trueCount])
      return { timer in
        var c = 0
        var index = 0
        while true {
          let range = CFRangeMake(index, input.count - index)
          index = CFBitVectorGetFirstIndexOfBit(bits, range, 1)
          guard index != kCFNotFound else { break }
          c += 1
          index += 1
        }
        precondition(c == trueCount, "count: \(input.count), found: \(c), expected: \(trueCount)")
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBitVectorGetCountOfBit",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      let trueCount = input.count / 2
      let bits = CFMutableBitVector._create(
        count: input.count,
        trueBits: input[..<trueCount])
      return { timer in
        let range = CFRangeMake(0, input.count)
        let c = CFBitVectorGetCountOfBit(bits, range, 1)
        precondition(c == trueCount)
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

    self.add(
      title: "CFBitVectorSetBitAtIndex (random-access set)",
      input: [Int].self
    ) { input in
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
      return { timer in
        let bits = CFBitVectorCreateMutable(nil, input.count)
        CFBitVectorSetCount(bits, input.count)
        timer.measure {
          for index in input {
            CFBitVectorSetBitAtIndex(bits, index, 1)
          }
        }
        blackHole(bits)
      }
#else
      // CFBitVector isn't available
      return nil
#endif
    }

  }
}
