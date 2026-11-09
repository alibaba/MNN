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
import Benchmarks
import DequeModule

if Deque<Int>._isConsistencyCheckingEnabled {
  complain("""
    *** INTERNAL CONSISTENCY CHECKING IS ENABLED ***

    Performance guarantees aren't valid in this configuration,
    and benchmarking data will be largely useless. Proceed at
    your own risk.

    """)
}

var benchmark = Benchmark(title: "Collection Benchmarks")
benchmark.registerCustomGenerators()
benchmark.addArrayBenchmarks()
benchmark.addSetBenchmarks()
benchmark.addDictionaryBenchmarks()
benchmark.addTreeDictionaryBenchmarks()
benchmark.addDequeBenchmarks()
benchmark.addOrderedSetBenchmarks()
benchmark.addOrderedDictionaryBenchmarks()
benchmark.addHeapBenchmarks()
benchmark.addBitSetBenchmarks()
benchmark.addTreeSetBenchmarks()
benchmark.addCppBenchmarks()
benchmark.addBigStringBenchmarks()
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
benchmark.addFoundationBenchmarks()
#endif

benchmark.main()
