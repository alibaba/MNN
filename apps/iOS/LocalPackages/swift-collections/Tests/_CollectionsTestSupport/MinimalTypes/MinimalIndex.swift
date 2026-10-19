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

// Loosely adapted from https://github.com/apple/swift/tree/main/stdlib/private/StdlibUnittest

import XCTest

public struct MinimalIndex {
  internal let _state: _CollectionState
  internal let _id: Int
  internal var _offset: Int

  internal init(state: _CollectionState, offset: Int) {
    _state = state
    _id = state.context.nextIndexId()
    _offset = offset
    precondition(_offset >= 0 && _offset <= state.count)
  }

  public var offset: Int { _offset }
  public var context: TestContext { _state.context }
}

extension MinimalIndex: Equatable {
  public static func == (left: Self, right: Self) -> Bool {
    left._assertCompatible(with: right)
    return left._offset == right._offset
  }
}

extension MinimalIndex: Comparable {
  public static func < (left: Self, right: Self) -> Bool {
    left._assertCompatible(with: right)
    return left._offset < right._offset
  }
}

extension MinimalIndex: CustomStringConvertible {
  public var description: String {
    return "MinimalIndex(offset: \(_offset), state: \(_state.id))"
  }
}

extension MinimalIndex {
  func _assertCompatible(with other: Self) {
    if self._state === other._state { return }
    expectTrue(self._state.isValidIndex(other), "Invalid index", trapping: true)
    expectTrue(other._state.isValidIndex(self), "Invalid index", trapping: true)
  }
}
