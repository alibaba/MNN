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

struct _MinimalCollectionCore<Element> {
  var state: _CollectionState
  var elements: [Element]
  var underestimatedCount: Int

  public init<S: Sequence>(
    context: TestContext,
    elements: S,
    underestimatedCount: UnderestimatedCountBehavior? = nil
  ) where S.Element == Element {
    self.init(context: context, elements: Array(elements), underestimatedCount: underestimatedCount)
  }

  public init(
    context: TestContext,
    elements: [Element],
    underestimatedCount: UnderestimatedCountBehavior? = nil
  ) {
    self.state = _CollectionState(context: context, parent: nil, count: elements.count)
    self.elements = elements
    self.underestimatedCount = underestimatedCount?.value(forCount: elements.count) ?? elements.count
  }

  var context: TestContext { state.context }
}

extension _MinimalCollectionCore {
  var startIndex: MinimalIndex {
    MinimalIndex(state: state, offset: 0)
  }

  var endIndex: MinimalIndex {
    MinimalIndex(state: state, offset: count)
  }

  var count: Int {
    elements.count
  }

  var isEmpty: Bool {
    elements.isEmpty
  }
}

extension _MinimalCollectionCore {
  mutating func ensureUniqueState() {
    if !isKnownUniquelyReferenced(&state) {
      state = state.spawnChild()
    }
  }
}

// MARK: Index validation

extension _MinimalCollectionCore {
  internal func index(at offset: Int) -> MinimalIndex {
    return MinimalIndex(state: state, offset: offset)
  }

  func isValidIndex(_ index: MinimalIndex) -> Bool {
    state.isValidIndex(index)
  }

  func assertValidIndex(
    _ index: MinimalIndex,
    _ message: @autoclosure () -> String = "Invalid index",
    file: StaticString = #file,
    line: UInt = #line
  ) {
    expectTrue(isValidIndex(index),
               message(), trapping: true,
               file: file, line: line)
  }

  func assertValidIndexBeforeEnd(
    _ index: MinimalIndex,
    _ message: @autoclosure () -> String = "Invalid index",
    file: StaticString = #file,
    line: UInt = #line
  ) {
    expectTrue(isValidIndex(index),
               message(), trapping: true,
               file: file, line: line)

    expectTrue(index._offset != elements.count,
               message(), trapping: true,
               file: file, line: line)
  }

  func assertValid(
    _ range: Range<MinimalIndex>,
    message: @autoclosure () -> String = "Invalid index range",
    file: StaticString = #file,
    line: UInt = #line
  ) {
    assertValidIndex(range.lowerBound, message())
    assertValidIndex(range.upperBound, message())
  }
}

// MARK: Index navigation

extension _MinimalCollectionCore {
  func index(after index: MinimalIndex) -> MinimalIndex {
    assertValidIndex(index)
    expectTrue(index._offset < count,
               "Can't advance beyond endIndex",
               trapping: true)
    return self.index(at: index._offset + 1)
  }

  func index(before index: MinimalIndex) -> MinimalIndex {
    assertValidIndex(index)
    expectTrue(index._offset > 0,
               "Can't advance before startIndex",
               trapping: true)
    return self.index(at: index._offset - 1)
  }

  func formIndex(after index: inout MinimalIndex) {
    assertValidIndex(index)
    expectTrue(index._offset < count,
               "Can't advance beyond endIndex",
               trapping: true)
    index._offset += 1
  }

  func formIndex(before index: inout MinimalIndex) {
    assertValidIndex(index)
    expectTrue(index._offset > 0,
               "Can't advance before startIndex",
               trapping: true)
    index._offset -= 1
  }

  public func distance(from start: MinimalIndex, to end: MinimalIndex) -> Int {
    assertValidIndex(start)
    assertValidIndex(end)
    return end._offset - start._offset
  }

  public func index(
    _ index: MinimalIndex,
    offsetBy n: Int
  ) -> MinimalIndex {
    assertValidIndex(index)
    expectTrue(index._offset + n >= 0,
               "Can't advance before startIndex",
               trapping: true)
    expectTrue(index._offset + n <= count,
               "Can't advance after endIndex",
               trapping: true)
    return self.index(at: index._offset + n)
  }

  func _failEarlyRangeCheck(
    _ index: MinimalIndex,
    bounds: Range<MinimalIndex>
  ) {
    assertValidIndex(index)
    assertValid(bounds)
    expectLessThanOrEqual(
      bounds.lowerBound._offset, index._offset,
      "Index out of bounds", trapping: true)
    expectLessThan(
      index._offset, bounds.upperBound._offset,
      "Index out of bounds", trapping: true)
  }

  public func _failEarlyRangeCheck(
    _ range: Range<MinimalIndex>,
    bounds: Range<MinimalIndex>
  ) {
    assertValid(range)
    assertValid(bounds)
    expectLessThanOrEqual(
      bounds.lowerBound._offset, range.lowerBound._offset,
      "Index range out of bounds", trapping: true)
    expectLessThanOrEqual(
      range.upperBound._offset, bounds.upperBound._offset,
      "Index range out of bounds", trapping: true)
  }

  func _count(of range: Range<MinimalIndex>) -> Int {
    range.upperBound._offset - range.lowerBound._offset
  }
}

// MARK: Mutations

extension _MinimalCollectionCore {
  subscript(index: MinimalIndex) -> Element {
    get {
      assertValidIndexBeforeEnd(index)
      return elements[index._offset]
    }
    set {
      assertValidIndexBeforeEnd(index)
      elements[index._offset] = newValue
      // MutableCollection requires that the subscript setter
      // not invalidate any indices.
    }
  }

  mutating func replaceSubrange<C: Collection>(
    _ subrange: Range<MinimalIndex>,
    with newElements: C
  ) where C.Element == Element {
    assertValid(subrange)
    elements.replaceSubrange(
      subrange.lowerBound._offset ..< subrange.upperBound._offset,
      with: newElements)

    ensureUniqueState()
    state.replace(subrange, with: newElements.count)
  }

  mutating func reserveCapacity(minimumCapacity: Int) {
    elements.reserveCapacity(minimumCapacity)
    state.replaceAll()
  }

  mutating func append(_ item: Element) {
    elements.append(item)
    ensureUniqueState()
    state.insert(count: 1, at: state.count)
  }

  mutating func append<S: Sequence>(
    contentsOf newElements: S
  ) where S.Element == Element {
    elements.append(contentsOf: newElements)
    ensureUniqueState()
    state.insert(count: elements.count - state.count, at: state.count)
  }

  mutating func insert(
    _ newElement: Element,
    at index: MinimalIndex
  ) {
    assertValidIndex(index)
    elements.insert(newElement, at: index._offset)
    ensureUniqueState()
    state.insert(count: 1, at: index._offset)
  }

  mutating func insert<C: Collection>(
    contentsOf newElements: C,
    at index: MinimalIndex
  ) where C.Element == Element {
    assertValidIndex(index)
    elements.insert(contentsOf: newElements, at: index._offset)
    ensureUniqueState()
    state.insert(count: newElements.count, at: index._offset)
  }

  mutating func remove(at index: MinimalIndex) -> Element {
    assertValidIndexBeforeEnd(index)
    ensureUniqueState()
    state.remove(count: 1, at: index._offset)
    return elements.remove(at: index._offset)
  }

  mutating func removeSubrange(_ bounds: Range<MinimalIndex>) {
    assertValid(bounds)
    elements.removeSubrange(bounds.lowerBound._offset ..< bounds.upperBound._offset)
    ensureUniqueState()
    state.remove(count: _count(of: bounds), at: bounds.lowerBound._offset)
  }

  mutating func removeFirst() -> Element {
    expectTrue(
      count > 0,
      "Can't remove first element of an empty collection",
      trapping: true)
    ensureUniqueState()
    state.remove(count: 1, at: 0)
    return elements.removeFirst()
  }

  mutating func removeFirst(_ n: Int) {
    expectTrue(
      n >= 0,
      "Can't remove a negative number of elements",
      trapping: true)
    expectTrue(
      n <= count,
      "Can't remove more elements than there are in the collection",
      trapping: true)
    ensureUniqueState()
    state.remove(count: n, at: 0)
    elements.removeFirst(n)
  }

  public mutating func _customRemoveLast() -> Element? {
    expectTrue(
      !isEmpty,
      "Can't remove last element of an empty collection",
      trapping: true)
    ensureUniqueState()
    state.remove(count: 1, at: state.count)
    return elements.removeLast()
  }

  public mutating func _customRemoveLast(_ n: Int) -> Bool {
    expectTrue(
      n >= 0,
      "Can't remove a negative number of elements",
      trapping: true)
    expectTrue(
      count >= n,
      "Can't remove more elements than there are in the collection",
      trapping: true)
    elements.removeLast(n)
    ensureUniqueState()
    state.remove(count: n, at: state.count - n)
    return true
  }


  mutating func removeAll(keepingCapacity keepCapacity: Bool) {
    elements.removeAll(keepingCapacity: keepCapacity)
    ensureUniqueState()
    state.remove(count: state.count, at: 0)
  }

  mutating func removeAll(
    where shouldBeRemoved: (Element) throws -> Bool
  ) rethrows {
    defer {
      ensureUniqueState()
      state.reset(count: elements.count)
    }
    try elements.removeAll(where: shouldBeRemoved)
  }
}

