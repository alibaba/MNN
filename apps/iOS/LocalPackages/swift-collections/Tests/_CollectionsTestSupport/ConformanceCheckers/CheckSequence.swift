//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

// Loosely adapted from https://github.com/apple/swift/tree/main/stdlib/private/StdlibCollectionUnittest

// FIXME: Port all of the collection validation tests from the Swift compiler codebase.

import XCTest

// FIXME: Port the collection validation tests from the Swift compiler codebase.

public func checkSequence<S: Sequence, Expected: Sequence>(
  _ sequenceGenerator: () -> S,
  expectedContents: Expected,
  file: StaticString = #file,
  line: UInt = #line
) where S.Element == Expected.Element, S.Element: Equatable {
  checkSequence(
    sequenceGenerator,
    expectedContents: expectedContents,
    by: ==,
    file: file,
    line: line)
}

public func checkSequence<S: Sequence, Expected: Sequence>(
  _ sequenceGenerator: () -> S,
  expectedContents: Expected,
  by areEquivalent: (S.Element, S.Element) -> Bool,
  file: StaticString = #file,
  line: UInt = #line
) where S.Element == Expected.Element {
  let entry = TestContext.current.push("checkSequence", file: file, line: line)
  defer { TestContext.current.pop(entry) }

  let expectedContents = Array(expectedContents)

  do {
    let seq = sequenceGenerator()
    let underestimatedCount = seq.underestimatedCount
    expectLessThanOrEqual(underestimatedCount, expectedContents.count)
    expectGreaterThanOrEqual(underestimatedCount, 0)
  }

  // Check contiguous storage.
  do {
    let seq = sequenceGenerator()
    let r: Int? = seq.withContiguousStorageIfAvailable { buffer in
      expectEquivalentElements(buffer, expectedContents, by: areEquivalent)
      return 42
    }
    expectTrue(r == 42 || r == nil)
  }

  do {
    let seq = sequenceGenerator()
    var it = seq.makeIterator()
    var i = 0
    while i < expectedContents.count {
      expectEquivalent(it.next(), expectedContents[i], by: areEquivalent)
      i += 1
    }
    expectNil(it.next())
    expectNil(it.next())
  }

  do {
    let seq = sequenceGenerator()
    let underestimatedCount = seq.underestimatedCount
    var state: (it: S.Iterator, count: Int)!
    var array = Array<S.Element>(
      unsafeUninitializedCapacity: underestimatedCount
    ) { buffer, count in
      state = seq._copyContents(initializing: buffer)
      count = state.count
    }
    expectEqual(state.count, underestimatedCount)
    expectEquivalentElements(
      array,
      expectedContents[..<state.count],
      by: areEquivalent)
    while let item = state.it.next() {
      array.append(item)
    }
    expectEquivalentElements(array, expectedContents, by: areEquivalent)
  }
}
