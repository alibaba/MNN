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
  mutating func _insert(
    contentsOf other: __owned Substring,
    at index: Index
  ) {
    precondition(index <= endIndex, "Index out of bounds")
    if other.isEmpty { return }
    if index == endIndex {
      self.append(contentsOf: other)
      return
    }
    
    // Note: we must disable the forward peeking optimization as we'll need to know the actual
    // full grapheme breaking state of the ingester at the start of the insertion.
    // (We'll use it to resync grapheme breaks in the parts that follow the insertion.)
    let index = self.resolve(index, preferEnd: true)
    var ingester = _ingester(forInserting: other, at: index, allowForwardPeek: false)
    var ri = index._rope!
    var ci = index._chunkIndex
    let r = _rope.update(at: &ri) { $0.insert(from: &ingester, at: ci) }
    switch r {
    case let .inline(r):
      assert(ingester.isAtEnd)
      guard var r = r else { break }
      ci = String.Index(_utf8Offset: ci._utf8Offset + r.increment)
      let i = Index(baseUTF8Offset: index._utf8BaseOffset, _rope: ri, chunk: ci)
      resyncBreaks(startingAt: i, old: &r.old, new: &r.new)
    case let .split(spawn: spawn, endStates: r):
      assert(ingester.isAtEnd)
      _rope.formIndex(after: &ri)
      _rope.insert(spawn, at: ri)
      guard var r = r else { break }
      let i = Index(_utf8Offset: index.utf8Offset + r.increment, _rope: ri, chunkOffset: spawn.utf8Count)
      resyncBreaks(startingAt: i, old: &r.old, new: &r.new)
    case .large:
      var builder = self.split(at: index, state: ingester.state)
      builder.append(from: &ingester)
      self = builder.finalize()
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  mutating func _insert(contentsOf other: __owned Self, at index: Index) {
    guard index < endIndex else {
      precondition(index == endIndex, "Index out of bounds")
      self.append(contentsOf: other)
      return
    }
    guard index > startIndex else {
      self.prepend(contentsOf: other)
      return
    }
    if other._rope.isSingleton {
      // Fast path when `other` is tiny.
      let chunk = other._rope.first!
      insert(contentsOf: chunk.string, at: index)
      return
    }
    var builder = self.split(at: index)
    builder.append(other)
    self = builder.finalize()
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  mutating func _insert(contentsOf other: __owned Self, in range: Range<Index>, at index: Index) {
    guard index < endIndex else {
      precondition(index == endIndex, "Index out of bounds")
      self._append(contentsOf: other, in: range)
      return
    }
    guard index > startIndex else {
      self.prepend(contentsOf: other, in: range)
      return
    }
    guard !range._isEmptyUTF8 else { return }
    // FIXME: Fast path when `other` is tiny.
    var builder = self.split(at: index)
    builder.append(other, in: range)
    self = builder.finalize()
  }
}
