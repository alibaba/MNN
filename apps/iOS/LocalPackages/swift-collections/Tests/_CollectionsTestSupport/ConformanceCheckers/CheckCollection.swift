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

extension Sequence {
  func _contentsByIterator() -> [Element] {
    var result: [Element] = []
    var it = makeIterator()
    while let item = it.next() {
      result.append(item)
    }
    return result
  }

  func _contentsByCopyContents(_ count: Int? = nil) -> [Element] {
    var it: Iterator?
    var result = Array(
      unsafeUninitializedCapacity: count ?? self.underestimatedCount
    ) { buffer, count in
      (it, count) = self._copyContents(initializing: buffer)
    }
    while let next = it!.next() {
      result.append(next)
    }
    return result
  }
}

extension Collection {
  func _indicesByIndexAfter() -> [Index] {
    var result: [Index] = []
    var i = startIndex
    while i != endIndex {
      result.append(i)
      i = index(after: i)
    }
    return result
  }

  func _indicesByFormIndexAfter() -> [Index] {
    var result: [Index] = []
    var i = startIndex
    while i != endIndex {
      result.append(i)
      formIndex(after: &i)
    }
    return result
  }
}

public func checkCollection<C: Collection, Expected: Sequence>(
  _ collection: C,
  expectedContents: Expected,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) where C.Element: Equatable, Expected.Element == C.Element {
  let expectedContents = Array(expectedContents)

  checkCollection(
    collection,
    expectedContents: expectedContents,
    by: ==,
    maxSamples: maxSamples,
    file: file, line: line)

  let indicesByIndexAfter = collection._indicesByIndexAfter()
  for item in expectedContents {
    if let first = collection._customIndexOfEquatableElement(item) {
      expectNotNil(first) { first in
        expectEqual(collection[first], item)
        let expected = expectedContents.firstIndex(of: item)!
        expectEqual(first, indicesByIndexAfter[expected])
      }
    }
    if let last = collection._customLastIndexOfEquatableElement(item) {
      expectNotNil(last) { last in
        expectEqual(collection[last], item)
        let expected = expectedContents.lastIndex(of: item)!
        expectEqual(last, indicesByIndexAfter[expected])
      }
    }
  }
}

public func checkCollection<C: Collection, Expected: Sequence>(
  _ collection: C,
  expectedContents: Expected,
  by areEquivalent: (C.Element, C.Element) -> Bool,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) where Expected.Element == C.Element {
  checkSequence(
    { collection },
    expectedContents: expectedContents,
    by: areEquivalent,
    file: file, line: line)
  _checkCollection(
    collection,
    expectedContents: expectedContents,
    by: areEquivalent,
    maxSamples: maxSamples,
    file: file, line: line)
}

public func _checkCollection<C: Collection, Expected: Sequence>(
  _ collection: C,
  expectedContents: Expected,
  by areEquivalent: (C.Element, C.Element) -> Bool,
  maxSamples: Int? = nil,
  file: StaticString = #file,
  line: UInt = #line
) where Expected.Element == C.Element {
  let entry = TestContext.current.push("checkCollection", file: file, line: line)
  defer { TestContext.current.pop(entry) }

  let expectedContents = Array(expectedContents)
  expectEqual(collection.isEmpty, expectedContents.isEmpty)
  expectEqual(collection.count, expectedContents.count)

  // Check that `index(after:)` produces the same results as `formIndex(after:)`
  let indicesByIndexAfter = collection._indicesByIndexAfter()
  let indicesByFormIndexAfter = collection._indicesByFormIndexAfter()
  expectEqual(indicesByIndexAfter, indicesByFormIndexAfter)

  // Check contents using the iterator.
  expectEquivalentElements(
    collection._contentsByIterator(),
    expectedContents,
    by: areEquivalent)

  // Check _copyContents.
  expectEquivalentElements(
    collection._contentsByCopyContents(),
    expectedContents,
    by: areEquivalent)

  // Check contents using indexing.
  let indexContents1 = indicesByIndexAfter.map { collection[$0] }
  expectEquivalentElements(indexContents1, expectedContents, by: areEquivalent)
  let indexContents2 = indicesByFormIndexAfter.map { collection[$0] }
  expectEquivalentElements(indexContents2, expectedContents, by: areEquivalent)

  // Check the endIndex.
  expectEqual(collection.endIndex, collection.indices.endIndex)

  // Quickly check the Indices associated type
  expectEqual(collection.indices.count, collection.count)
  expectEqualElements(collection.indices, indicesByIndexAfter)
  expectEqual(collection.indices.endIndex, collection.endIndex)

  // The sequence of indices must be monotonically increasing.
  var allIndices = indicesByIndexAfter
  allIndices.append(collection.endIndex)

  if C.Index.self != Int.self {
    checkComparable(
      allIndices,
      oracle: { .comparing($0, $1) },
      maxSamples: maxSamples)
  }

  // Check `index(_,offsetBy:limitedBy:)`
  let limits =
    Set([0, allIndices.count - 1, allIndices.count / 2])
    .sorted()
  withEvery("limit", in: limits) { limit in
    withEvery("i", in: 0 ..< allIndices.count) { i in
      let max = allIndices.count - i + (limit >= i ? 2 : 0)
      withEvery("delta", in: 0 ..< max) { delta in
        let actual = collection.index(
          allIndices[i],
          offsetBy: delta,
          limitedBy: allIndices[limit])
        let j = i + delta
        let expected = i > limit || j <= limit ? allIndices[j] : nil
        expectEqual(actual, expected)
      }
    }
  }
  
  withSomeRanges(
    "range", in: 0 ..< allIndices.count - 1, maxSamples: maxSamples
  ) { range in
    let i = range.lowerBound
    let j = range.upperBound

    // Check `index(_,offsetBy:)`
    let e = collection.index(allIndices[i], offsetBy: j - i)
    expectEqual(e, allIndices[j])
    if j < expectedContents.count {
      expectEquivalent(collection[e], expectedContents[j], by: areEquivalent)
    }

    // Check `distance(from:to:)`
    let d = collection.distance(from: allIndices[i], to: allIndices[j])
    expectEqual(d, j - i)
    
    // Check slicing.
    let range = allIndices[i] ..< allIndices[j]
    let slice = collection[range]
    expectEqual(slice.count, j - i)
    expectEqual(slice.isEmpty, i == j)
    expectEqual(slice.startIndex, allIndices[i])
    expectEqual(slice.endIndex, allIndices[j])
    expectEqual(slice.distance(from: allIndices[i], to: allIndices[j]), j - i)
    expectEqual(slice.index(allIndices[i], offsetBy: j - i), allIndices[j])
    
    expectEqual(slice.index(allIndices[i], offsetBy: j - i, limitedBy: allIndices[j]),
                allIndices[j])
    expectEqual(slice.index(allIndices[i], offsetBy: j - i, limitedBy: allIndices[i]),
                j - i > 0 ? nil : allIndices[i])
    expectEqual(slice.index(allIndices[i], offsetBy: j - i, limitedBy: allIndices[0]),
                i > 0 || j == 0 ? allIndices[j] : nil)
    
    expectEquivalentElements(slice, expectedContents[i ..< j],
                             by: areEquivalent)
    
    expectEquivalentElements(
      slice._contentsByIterator(), expectedContents[i ..< j],
      by: areEquivalent)
    expectEquivalentElements(
      slice._contentsByCopyContents(), expectedContents[i ..< j],
      by: areEquivalent)
    
    // Check _copyContents.
    let copyContents = collection._contentsByCopyContents()
    expectEquivalentElements(
      copyContents, expectedContents,
      by: areEquivalent)
    
    expectEqualElements(slice._indicesByIndexAfter(), allIndices[i ..< j])
    expectEqualElements(slice._indicesByFormIndexAfter(), allIndices[i ..< j])
    expectEqualElements(slice.indices, allIndices[i ..< j])
    expectEqualElements(slice.indices._indicesByIndexAfter(), allIndices[i ..< j])
    expectEqualElements(slice.indices._indicesByFormIndexAfter(), allIndices[i ..< j])
    // Check the subsequence iterator.
    expectEquivalentElements(
      slice, expectedContents[i ..< j],
      by: areEquivalent)
    // Check the subsequence subscript.
    withSome("k", in: i ..< j, maxSamples: 25) { k in
      expectEquivalent(
        slice[allIndices[k]],
        expectedContents[k],
        by: areEquivalent)
    }
    // Check _copyToContiguousArray.
    expectEquivalentElements(
      Array(slice), expectedContents[i ..< j],
      by: areEquivalent)
    // Check slicing of slices.
    expectEquivalentElements(
      slice[range], expectedContents[i ..< j],
      by: areEquivalent)
  }

  if C.Indices.self != C.self && C.Indices.self != DefaultIndices<C>.self {
    // Do a more exhaustive check on Indices.
    TestContext.current.withTrace("Indices") {
      checkCollection(
        collection.indices,
        expectedContents: indicesByIndexAfter,
        maxSamples: maxSamples)
    }
  }
}
