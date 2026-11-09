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

/// A minimal implementation of `RandomAccessCollection & MutableCollection & RangeReplaceableCollection` with extra checks.
public struct MinimalMutableRangeReplaceableRandomAccessCollection<Element> {
  internal var _core: _MinimalCollectionCore<Element>

  public let timesMakeIteratorCalled = ResettableValue(0)
  public let timesUnderestimatedCountCalled = ResettableValue(0)
  public let timesRangeChecksCalled = ResettableValue(0)
  public let timesIndexNavigationCalled = ResettableValue(0)
  public let timesSubscriptGetterCalled = ResettableValue(0)
  public let timesSubscriptSetterCalled = ResettableValue(0)
  public let timesRangeSubscriptGetterCalled = ResettableValue(0)
  public let timesRangeSubscriptSetterCalled = ResettableValue(0)
  public let timesSwapCalled = ResettableValue(0)
  public let timesPartitionCalled = ResettableValue(0)

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

extension MinimalMutableRangeReplaceableRandomAccessCollection: Sequence {
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

extension MinimalMutableRangeReplaceableRandomAccessCollection: RandomAccessCollection {
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
    get {
      timesSubscriptGetterCalled.increment()
      return _core[i]
    }
    set {
      timesSubscriptSetterCalled.increment()
      _core[i] = newValue
    }
  }

  public subscript(bounds: Range<MinimalIndex>) -> SubSequence {
    get {
      timesRangeSubscriptGetterCalled.increment()
      _core.assertValid(bounds)
      return Slice(base: self, bounds: bounds)
    }
    set {
      timesRangeSubscriptSetterCalled.increment()
      _core.assertValid(bounds)
      _core.elements[bounds.lowerBound._offset ..< bounds.upperBound._offset] =
        Self._coreSlice(from: newValue)
      // Don't invalidate indices.
    }
  }

  static func _coreSlice(from slice: SubSequence) -> ArraySlice<Element> {
    slice.base._core.assertValid(slice.startIndex ..< slice.endIndex)
    return slice.base._core.elements[slice.startIndex._offset ..< slice.endIndex._offset]
  }
}

extension MinimalMutableRangeReplaceableRandomAccessCollection: MutableCollection {
  public mutating func partition(
    by belongsInSecondPartition: (Element) throws -> Bool
  ) rethrows -> Index {
    timesPartitionCalled.increment()
    let pivot = try _core.elements.partition(by: belongsInSecondPartition)
    return _core.index(at: pivot)
  }

  public mutating func swapAt(_ i: Index, _ j: Index) {
    _core.assertValidIndexBeforeEnd(i)
    _core.assertValidIndexBeforeEnd(j)
    timesSwapCalled.increment()
    _core.elements.swapAt(i._offset, j._offset)
    // Don't invalidate indices.
  }

  public mutating func _withUnsafeMutableBufferPointerIfSupported<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    return nil
  }

  public mutating func withContiguousMutableStorageIfAvailable<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    return nil
  }
}

extension MinimalMutableRangeReplaceableRandomAccessCollection: RangeReplaceableCollection {
  public init() {
    self.init([])
  }

  public mutating func replaceSubrange<C: Collection>(
    _ subrange: Range<Index>,
    with newElements: C
  ) where C.Element == Element {
    _core.replaceSubrange(subrange, with: newElements)
  }

  public mutating func reserveCapacity(_ n: Int) {
    _core.reserveCapacity(minimumCapacity: n)
  }

  public init<S: Sequence>(_ elements: S) where S.Element == Element {
    self.init(elements, context: TestContext.current)
  }

  public mutating func append(_ newElement: Element) {
    _core.append(newElement)
  }

  public mutating func append<S: Sequence>(
    contentsOf newElements: S
  ) where S.Element == Element {
    _core.append(contentsOf: newElements)
  }

  public mutating func insert(_ newElement: Element, at i: Index) {
    _core.insert(newElement, at: i)
  }

  @discardableResult
  public mutating func remove(at i: Index) -> Element {
    return _core.remove(at: i)
  }

  public mutating func removeSubrange(_ bounds: Range<Index>) {
    _core.removeSubrange(bounds)
  }

  public mutating func _customRemoveLast() -> Element? {
    return _core._customRemoveLast()
  }

  public mutating func _customRemoveLast(_ n: Int) -> Bool {
    return _core._customRemoveLast(n)
  }

  @discardableResult
  public mutating func removeFirst() -> Element {
    return _core.removeFirst()
  }

  public mutating func removeFirst(_ n: Int) {
    _core.removeFirst(n)
  }

  public mutating func removeAll(keepingCapacity keepCapacity: Bool) {
    _core.removeAll(keepingCapacity: keepCapacity)
  }

  public mutating func removeAll(
    where shouldBeRemoved: (Element) throws -> Bool) rethrows {
    try _core.removeAll(where: shouldBeRemoved)
  }
}
