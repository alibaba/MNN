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

/// State shared by all iterators of a MinimalSequence instance.
internal class _MinimalIteratorSharedState<Element> {
  internal init(_ data: [Element]) {
    self.data = data
  }

  internal let data: [Element]
  internal var i: Int = 0
  internal var underestimatedCount: Int = 0

  public func next() -> Element? {
    if i == data.count {
      return nil
    }
    defer { i += 1 }
    return data[i]
  }
}

/// An iterator that implements the IteratorProtocol contract in the most
/// narrow way possible.
public struct MinimalIterator<Element>: IteratorProtocol {
  internal let _sharedState: _MinimalIteratorSharedState<Element>

  public init<S: Sequence>(_ s: S) where S.Element == Element {
    self._sharedState = _MinimalIteratorSharedState(Array(s))
  }

  public init(_ data: [Element]) {
    self._sharedState = _MinimalIteratorSharedState(data)
  }

  internal init(_ _sharedState: _MinimalIteratorSharedState<Element>) {
    self._sharedState = _sharedState
  }

  public func next() -> Element? {
    _sharedState.next()
  }
}
