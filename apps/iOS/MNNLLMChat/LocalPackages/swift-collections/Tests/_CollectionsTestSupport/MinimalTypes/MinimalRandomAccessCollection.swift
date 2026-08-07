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

/// A minimal implementation of `RandomAccessCollection` with extra checks.
public struct MinimalRandomAccessCollection<Element> {
  internal var _core: _MinimalCollectionCore<Element>

  public let timesMakeIteratorCalled = ResettableValue(0)
  public let timesUnderestimatedCountCalled = ResettableValue(0)
  public let timesRangeChecksCalled = ResettableValue(0)
  public let timesIndexNavigationCalled = ResettableValue(0)
  public let timesSubscriptCalled = ResettableValue(0)
  public let timesRangeSubscriptCalled = ResettableValue(0)

  public init<S: Sequence>(
    _ elements: S,
    context: TestContext = TestContext.current,
    underestimatedCount: UnderestimatedCountBehavior = .value(0)
  ) where S.Element == Element {
    self._core = _MinimalCollectionCore(context: context, elements: elements, underestimatedCount: underestimatedCount)
  }

  var _context: TestContext {
    _core.context
  }
}

extension MinimalRandomAccessCollection: Sequence {
  public typealias Iterator = MinimalIterator<Element>

  public func makeIterator() -> MinimalIterator<Element> {
    timesMakeIteratorCalled.increment()
    return MinimalIterator(_core.elements)
  }

  public var underestimatedCount: Int {
    timesUnderestimatedCountCalled.increment()
    return _core.underestimatedCount
  }
}

extension MinimalRandomAccessCollection: RandomAccessCollection {
  public typealias Index = MinimalIndex
  public typealias SubSequence = Slice<Self>
  public typealias Indices = DefaultIndices<Self>

  public var startIndex: MinimalIndex {
    timesIndexNavigationCalled.increment()
    return _core.startIndex
  }

  public var endIndex: MinimalIndex {
    timesIndexNavigationCalled.increment()
    return _core.endIndex
  }

  public var isEmpty: Bool {
    timesIndexNavigationCalled.increment()
    return _core.isEmpty
  }

  public var count: Int {
    timesIndexNavigationCalled.increment()
    return _core.count
  }

  public func _failEarlyRangeCheck(
    _ index: MinimalIndex,
    bounds: Range<MinimalIndex>
  ) {
    timesRangeChecksCalled.increment()
    _core._failEarlyRangeCheck(index, bounds: bounds)
  }

  public func _failEarlyRangeCheck(
    _ range: Range<MinimalIndex>,
    bounds: Range<MinimalIndex>
  ) {
    timesRangeChecksCalled.increment()
    _core._failEarlyRangeCheck(range, bounds: bounds)
  }

  public func index(after i: MinimalIndex) -> MinimalIndex {
    timesIndexNavigationCalled.increment()
    return _core.index(after: i)
  }

  public func index(before i: MinimalIndex) -> MinimalIndex {
    timesIndexNavigationCalled.increment()
    return _core.index(before: i)
  }

  public func distance(from start: MinimalIndex, to end: MinimalIndex)
    -> Int {
    timesIndexNavigationCalled.increment()
    return _core.distance(from: start, to: end)
  }

  public func index(_ i: Index, offsetBy n: Int) -> Index {
    timesIndexNavigationCalled.increment()
    return _core.index(i, offsetBy: n)
  }

  public subscript(i: MinimalIndex) -> Element {
    timesSubscriptCalled.increment()
    return _core[i]
  }

  public subscript(bounds: Range<MinimalIndex>) -> SubSequence {
    timesRangeSubscriptCalled.increment()
    _core.assertValid(bounds)
    return Slice(base: self, bounds: bounds)
  }
}
