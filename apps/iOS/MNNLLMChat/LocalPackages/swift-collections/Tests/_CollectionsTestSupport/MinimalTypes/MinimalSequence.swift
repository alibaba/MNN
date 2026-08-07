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

public enum UnderestimatedCountBehavior {
  /// Return the actual number of elements.
  case precise

  /// Return the actual number of elements divided by 2.
  case half

  /// Return an overestimated count.  Useful to test how algorithms reserve
  /// memory.
  case overestimate

  /// Return the provided value.
  case value(Int)

  func value(forCount count: Int) -> Int {
    switch self {
    case .precise:
      return count
    case .half:
      return count / 2
    case .overestimate:
      return count * 3 + 5
    case .value(let count):
      return count
    }
  }
}

/// A Sequence that implements the protocol contract in the most
/// narrow way possible.
///
/// This sequence is consumed when its iterator is advanced.
public struct MinimalSequence<T>: Sequence, CustomDebugStringConvertible {
  public let timesMakeIteratorCalled = ResettableValue(0)

  internal let _sharedState: _MinimalIteratorSharedState<T>
  internal let _isContiguous: Bool

  public init<S: Sequence>(
    elements: S,
    underestimatedCount: UnderestimatedCountBehavior = .value(0),
    isContiguous: Bool = false
  ) where S.Element == T {
    let data = Array(elements)
    self._sharedState = _MinimalIteratorSharedState(data)
    self._isContiguous = isContiguous
    switch underestimatedCount {
    case .precise:
      self._sharedState.underestimatedCount = data.count

    case .half:
      self._sharedState.underestimatedCount = data.count / 2

    case .overestimate:
      self._sharedState.underestimatedCount = data.count * 3 + 5

    case .value(let count):
      self._sharedState.underestimatedCount = count
    }
  }

  public func makeIterator() -> MinimalIterator<T> {
    timesMakeIteratorCalled.value += 1
    return MinimalIterator(_sharedState)
  }

  public var underestimatedCount: Int {
    return Swift.max(0, self._sharedState.underestimatedCount - self._sharedState.i)
  }

  public var debugDescription: String {
    return "MinimalSequence(\(_sharedState.data[_sharedState.i...]))"
  }

  public func withContiguousStorageIfAvailable<R>(
    _ body: (UnsafeBufferPointer<T>) throws -> R
  ) rethrows -> R? {
    guard _isContiguous else { return nil }
    return try _sharedState.data[_sharedState.i...]
      .withContiguousStorageIfAvailable(body)
  }
}
