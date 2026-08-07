//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  mutating func split(
    at index: Index
  ) -> Builder {
    let b = _ropeBuilder(at: index)
    let state = b._breakState()
    let builder = Builder(base: b, prefixEndState: state, suffixStartState: state)
    return builder
  }

  mutating func split(
    at index: Index,
    state: _CharacterRecognizer
  ) -> Builder {
    let b = _ropeBuilder(at: index)
    let builder = Builder(base: b, prefixEndState: state, suffixStartState: state)
    return builder
  }

  mutating func _ropeBuilder(at index: Index) -> _Rope.Builder {
    if let ri = index._rope, _rope.isValid(ri) {
      return _rope.split(at: ri, index._chunkIndex)
    }
    return _rope.builder(splittingAt: index.utf8Offset, in: _UTF8Metric())
  }
}
