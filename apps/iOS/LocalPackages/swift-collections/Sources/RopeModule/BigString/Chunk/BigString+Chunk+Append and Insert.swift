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
extension BigString._Chunk {
  mutating func append(_ other: __owned Self) {
    self._append(other.string[...], other.counts)
  }

  mutating func append(from ingester: inout BigString._Ingester) -> Self? {
    let desired = BigString._Ingester.desiredNextChunkSize(
      remaining: self.utf8Count + ingester.remainingUTF8)
    if desired == self.utf8Count {
      return nil
    }
    if desired > self.utf8Count {
      if let slice = ingester.nextSlice(maxUTF8Count: desired - self.utf8Count) {
        self._append(slice)
      }
      return nil
    }

    // Split current chunk.
    let cut = string.unicodeScalars._index(roundingDown: string._utf8Index(at: desired))
    var new = self.split(at: cut)
    precondition(!self.isUndersized)
    let slice = ingester.nextSlice()!
    new._append(slice)
    precondition(ingester.isAtEnd)
    precondition(!new.isUndersized)
    return new
  }

  mutating func _append(_ other: Slice) {
    let c = Counts(other)
    _append(other.string, c)
  }

  mutating func _append(_ str: __owned Substring, _ other: Counts) {
    self.counts.append(other)
    self.string += str
    invariantCheck()
  }

  mutating func _prepend(_ other: Slice) {
    let c = Counts(other)
    _prepend(other.string, c)
  }

  mutating func _prepend(_ str: __owned Substring, _ other: Counts) {
    let c = self.counts
    self.counts = other
    self.counts.append(c)
    self.string = str + self.string
    invariantCheck()
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  mutating func _insert(
    _ slice: Slice,
    at index: String.Index,
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> String.Index? {
    let offset = string._utf8Offset(of: index)
    let count = slice.string.utf8.count
    precondition(utf8Count + count <= Self.maxUTF8Count)

    let parts = self.splitCounts(at: index)
    self.counts = parts.left
    self.counts.append(Counts(slice))
    self.counts.append(parts.right)

    string.insert(contentsOf: slice.string, at: index)

    let end = string._utf8Index(at: offset + count)
    return resyncBreaks(startingAt: end, old: &old, new: &new)
  }

  typealias States = (increment: Int, old: _CharacterRecognizer, new: _CharacterRecognizer)

  mutating func insertAll(
    from ingester: inout BigString._Ingester,
    at index: String.Index
  ) -> States? {
    let remaining = ingester.remainingUTF8
    precondition(self.utf8Count + remaining <= Self.maxUTF8Count)
    var startState = ingester.state
    guard let slice = ingester.nextSlice(maxUTF8Count: remaining) else { return nil }
    var endState = ingester.state
    assert(ingester.isAtEnd)
    let offset = string._utf8Offset(of: index)
    if let _ = self._insert(slice, at: index, old: &startState, new: &endState) {
      return nil
    }
    return (self.utf8Count - offset, startState, endState)
  }

  enum InsertResult {
    case inline(States?)
    case split(spawn: BigString._Chunk, endStates: States?)
    case large
  }

  mutating func insert(
    from ingester: inout BigString._Ingester,
    at index: String.Index
  ) -> InsertResult {
    let origCount = self.utf8Count
    let rem = ingester.remainingUTF8
    guard rem > 0 else { return .inline(nil) }
    let sum = origCount + rem

    let offset = string._utf8Offset(of: index)
    if sum <= Self.maxUTF8Count {
      let r = insertAll(from: &ingester, at: index)
      return .inline(r)
    }

    let desired = BigString._Ingester.desiredNextChunkSize(remaining: sum)
    guard sum - desired + Self.maxSlicingError <= Self.maxUTF8Count else { return .large }

    if desired <= offset {
      // Inserted text lies entirely within `spawn`.
      let cut = string.unicodeScalars._index(roundingDown: string._utf8Index(at: desired))
      var spawn = split(at: cut)
      let i = spawn.string._utf8Index(at: offset - self.utf8Count)
      let r = spawn.insertAll(from: &ingester, at: i)
      assert(r == nil || r?.increment == sum - offset)
      return .split(spawn: spawn, endStates: r)
    }
    if desired >= offset + rem {
      // Inserted text lies entirely within `self`.
      let cut = string.unicodeScalars._index(roundingDown: string._utf8Index(at: desired - rem))
      assert(cut >= index)
      var spawn = split(at: cut)
      guard
        var r = self.insertAll(from: &ingester, at: string._utf8Index(at: offset)),
        nil == spawn.resyncBreaks(startingAt: spawn.string.startIndex, old: &r.old, new: &r.new)
      else {
        return .split(spawn: spawn, endStates: nil)
      }
      return .split(spawn: spawn, endStates: (sum - offset, r.old, r.new))
    }
    // Inserted text is split across `self` and `spawn`.
    var spawn = split(at: index)
    var old = ingester.state
    if let slice = ingester.nextSlice(maxUTF8Count: desired - offset) {
      self._append(slice)
    }
    let slice = ingester.nextSlice()!
    assert(ingester.isAtEnd)
    var new = ingester.state
    let stop = spawn._insert(slice, at: spawn.string.startIndex, old: &old, new: &new)
    if stop != nil {
      return .split(spawn: spawn, endStates: nil)
    }
    return .split(spawn: spawn, endStates: (sum - offset, old, new))
  }
}
