//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import ArgumentParser
import CollectionsBenchmark
import Collections

@main
struct MemoryBenchmarks: ParsableCommand {
  static var configuration: CommandConfiguration {
    CommandConfiguration(
      commandName: "memory-statistics",
      abstract: "A utility for running memory benchmarks for collection types.")
  }

  @OptionGroup
  var sizes: Benchmark.Options.SizeSelection

  mutating func run() throws {
    let sizes = try self.sizes.resolveSizes()

    var i = 0

    var d: Dictionary<String, String> = [:]
    var pd: TreeDictionary<String, String> = [:]

    print("""
      Size,"Dictionary<String, String>",\
      "TreeDictionary<String, String>",\
      "average node size",\
      "average item depth"
      """)

    var sumd: Double = 0
    var sump: Double = 0
    for size in sizes {
      while i < size.rawValue {
        let key = "key \(i)"
        let value = "value \(i)"
        d[key] = value
        pd[key] = value
        i += 1
      }

      let dstats = d.statistics
      let pstats = pd._statistics
      print("""
        \(size.rawValue),\
        \(dstats.memoryEfficiency),\
        \(pstats.memoryEfficiency),\
        \(pstats.averageNodeSize),\
        \(pstats.averageItemDepth)
        """)
      sumd += dstats.memoryEfficiency
      sump += pstats.memoryEfficiency
    }

    let pstats = pd._statistics
    complain("""
      Averages:
        Dictionary: \(sumd / Double(sizes.count))
        TreeDictionary: \(sump / Double(sizes.count))

      TreeDictionary at 1M items:
        average node size: \(pstats.averageNodeSize)
        average item depth: \(pstats.averageItemDepth)
        average lookup chain length: \(pstats.averageLookupChainLength)
      """)
  }
}


