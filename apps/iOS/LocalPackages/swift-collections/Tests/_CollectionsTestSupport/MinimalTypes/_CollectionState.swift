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

final class _CollectionState {
  /// The collection context in which this state belongs.
  let context: TestContext
  /// The unique identifier of this state within its context.
  let id: Int
  /// The state from which this instance spawned.
  private(set) var parent: _CollectionState?
  private(set) var count: Int
  private(set) var strategy: IndexInvalidationStrategy
  /// The identifier of the state in which each currently valid index first became valid.
  /// (Note: the end index is tracked separately, in `endIndexValidity`.
  /// The identifier tracked here is guaranteed to match one of the states on the `parent` chain.
  private var indexValidity: [Int]
  /// The identitifer of the state in which the current endIndex first became valid.
  private var endIndexValidity: Int

  enum IndexInvalidationStrategy: Hashable {
    case allIndices
    case afterChange
  }


  init(
    context: TestContext,
    parent: _CollectionState?,
    count: Int,
    strategy: IndexInvalidationStrategy = .allIndices,
    indexValidity: [Int],
    endIndexValidity: Int
  ) {
    self.context = context
    self.id = context.nextStateId()
    self.parent = parent
    self.count = count
    self.strategy = strategy
    self.indexValidity = indexValidity
    self.endIndexValidity = endIndexValidity
  }

  init(
    context: TestContext,
    parent: _CollectionState?,
    count: Int,
    strategy: IndexInvalidationStrategy = .allIndices
  ) {
    self.context = context
    self.parent = parent
    self.id = context.nextStateId()
    self.count = count
    self.strategy = strategy
    switch strategy {
    case .allIndices:
      self.indexValidity = []
      self.endIndexValidity = 0
    case .afterChange:
      self.indexValidity = Array(repeatElement(id, count: count))
      self.endIndexValidity = id
    }
  }

  deinit {
    // Prevent stack overflow for long parent chains.
    var node = self
    while isKnownUniquelyReferenced(&node.parent) {
      let p = node.parent!
      node.parent = nil
      node = p
    }
  }

  func spawnChild() -> _CollectionState {
    _CollectionState(
      context: context,
      parent: self,
      count: count,
      strategy: strategy,
      indexValidity: indexValidity,
      endIndexValidity: endIndexValidity)
  }
}

extension _CollectionState: Equatable {
  public static func == (left: _CollectionState, right: _CollectionState) -> Bool {
    return left.id == right.id
  }
}

extension _CollectionState: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(id)
  }
}

extension _CollectionState {
  func isValidIndex(_ index: MinimalIndex) -> Bool {
    if index._state === self { return true }
    guard index._state.context == context else { return false }
    guard index._offset >= 0 && index._offset <= count else { return false }
    switch strategy {
    case .allIndices:
      guard index._state.id == self.id else { return false }
    case .afterChange:
      if index._offset < count {
        guard index._state.id == indexValidity[index._offset] else { return false }
      } else {
        guard index._state.id == endIndexValidity else { return false }
      }
    }
    return true
  }
}

extension _CollectionState {
  internal subscript(validityAt offset: Int) -> Int {
    get {
      switch strategy {
      case .allIndices:
        return id
      case .afterChange:
        precondition(offset >= 0)
        if offset > count {
          return id
        }
        if offset == count {
          return endIndexValidity
        }
        return indexValidity[offset]
      }
    }
    set {
      precondition(strategy == .allIndices, "Cannot invalidate individual indices")
      precondition(offset >= 0 && offset <= count)
      if offset == count {
        endIndexValidity = newValue
      } else {
        indexValidity[offset] = newValue
      }
    }
  }
}

extension _CollectionState {
  func insert(count: Int, at offset: Int) {
    precondition(count >= 0 && offset >= 0 && offset <= self.count)
    guard count > 0 else { return }
    self.count += count
    switch strategy {
    case .allIndices:
      // Do nothing
      break
    case .afterChange:
      let rest = offset ..< indexValidity.count
      self.indexValidity.replaceSubrange(rest, with: repeatElement(id, count: rest.count + count))
      self.endIndexValidity = self.id
      assert(indexValidity.count == self.count)
    }
  }

  func remove(_ range: Range<MinimalIndex>) {
    remove(count: range.upperBound._offset - range.lowerBound._offset,
           at: range.lowerBound._offset)
  }

  func remove(count: Int, at offset: Int) {
    precondition(count >= 0 && offset >= 0 && offset + count <= self.count)
    guard count > 0 else { return }
    switch strategy {
    case .allIndices:
      self.count -= count
    case .afterChange:
      let rest = offset ..< self.count
      self.indexValidity.replaceSubrange(rest, with: repeatElement(id, count: rest.count - count))
      self.endIndexValidity = self.id
      self.count -= count
      assert(indexValidity.count == self.count)
    }
  }

  func replace(_ range: Range<MinimalIndex>, with newCount: Int) {
    replace(oldCount: range.upperBound._offset - range.lowerBound._offset,
           at: range.lowerBound._offset,
           with: newCount)
  }

  func replace(oldCount: Int, at offset: Int, with newCount: Int) {
    precondition(oldCount >= 0 && offset >= 0 && newCount >= 0)
    precondition(offset + oldCount <= self.count)
    guard oldCount > 0 || newCount > 0 else { return }
    switch strategy {
    case .allIndices:
      self.count += newCount - oldCount
    case .afterChange:
      let rest = offset ..< self.count
      self.indexValidity.replaceSubrange(
        rest,
        with: repeatElement(id, count: rest.count + newCount - oldCount))
      self.endIndexValidity = self.id
      self.count += newCount - oldCount
      assert(indexValidity.count == self.count)
    }
  }

  func replaceAll() {
    switch strategy {
    case .allIndices:
      // Do nothing
      break
    case .afterChange:
      self.indexValidity = Array(repeating: id, count: count)
      self.endIndexValidity = id
      assert(indexValidity.count == self.count)
    }
  }

  func reset(count: Int) {
    self.count = count
    switch strategy {
    case .allIndices:
      // Do nothing
      break
    case .afterChange:
      self.indexValidity = Array(repeating: id, count: count)
      self.endIndexValidity = id
      assert(indexValidity.count == self.count)
    }
  }
}
