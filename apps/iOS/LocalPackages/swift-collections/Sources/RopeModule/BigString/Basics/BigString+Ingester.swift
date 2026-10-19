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
  func _ingester(
    forInserting input: __owned Substring,
    at index: Index,
    allowForwardPeek: Bool
  ) -> _Ingester {
    let hint = allowForwardPeek ? input.unicodeScalars.first : nil
    let state = self._breakState(upTo: index, nextScalarHint: hint)
    return _Ingester(input, startState: state)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  internal struct _Ingester {
    typealias _Chunk = BigString._Chunk
    typealias Counts = BigString._Chunk.Counts
    
    var input: Substring
    
    /// The index of the beginning of the next chunk.
    var start: String.Index
    
    /// Grapheme breaking state at the start of the next chunk.
    var state: _CharacterRecognizer
    
    init(_ input: Substring) {
      self.init(input, startState: _CharacterRecognizer())
    }
    
    init(_ input: Substring, startState: __owned _CharacterRecognizer) {
      self.input = input
      // Prevent accidentally quadratic operation by ensuring that we have
      // a native UTF-8 string.
      // FIXME: This is wasteful: if `input` happens to be a bridged
      // FIXME: NSString instance, then it temporarily allocates a full
      // FIXME: copy of the (transcoded) input string, only to then copy
      // FIXME: its pieces into the tree later.
      // FIXME: We should have a direct ingester path for native UTF-16 data.
      self.input.makeContiguousUTF8()
      self.start = input.startIndex
      self.state = startState
    }
    
    init(_ input: String) {
      self.init(input[...])
    }
    
    init<S: StringProtocol>(_ input: S) {
      self.init(Substring(input))
    }
    
    var isAtEnd: Bool {
      start == input.endIndex
    }
    
    var remainingUTF8: Int {
      input.utf8.distance(from: start, to: input.endIndex)
    }
    
    mutating func nextSlice(
      maxUTF8Count: Int = _Chunk.maxUTF8Count
    ) -> _Chunk.Slice? {
      guard let range = input.base._nextSlice(
        after: start, limit: input.endIndex, maxUTF8Count: maxUTF8Count)
      else {
        assert(start == input.endIndex)
        return nil
      }
      if range.isEmpty {
        return nil // Not enough room.
      }
      assert(range.lowerBound == start && range.upperBound <= input.endIndex)
      start = range.upperBound
      
      var s = input[range]
      let c8 = s.utf8.count
      guard let r = state.firstBreak(in: s) else {
        // Anomalous case -- chunk is entirely a continuation of a single character.
        return (
          string: s,
          characters: 0,
          prefix: c8,
          suffix: c8)
      }
      let first = r.lowerBound
      s = s.suffix(from: r.upperBound)
      
      var characterCount = 1
      var last = first
      while let r = state.firstBreak(in: s) {
        last = r.lowerBound
        s = s.suffix(from: r.upperBound)
        characterCount += 1
      }
      let prefixCount = input.utf8.distance(from: range.lowerBound, to: first)
      let suffixCount = input.utf8.distance(from: last, to: range.upperBound)
      return (
        string: input[range],
        characters: characterCount,
        prefix: prefixCount,
        suffix: suffixCount)
    }
    
    mutating func nextChunk(maxUTF8Count: Int = _Chunk.maxUTF8Count) -> _Chunk? {
      guard let slice = nextSlice(maxUTF8Count: maxUTF8Count) else { return nil }
      return _Chunk(slice)
    }
    
    static func desiredNextChunkSize(remaining: Int) -> Int {
      if remaining <= _Chunk.maxUTF8Count {
        return remaining
      }
      if remaining >= _Chunk.maxUTF8Count + _Chunk.minUTF8Count {
        return _Chunk.maxUTF8Count
      }
      return remaining - _Chunk.minUTF8Count
    }
    
    mutating func nextWellSizedSlice(suffix: Int = 0) -> _Chunk.Slice? {
      let desired = Self.desiredNextChunkSize(remaining: remainingUTF8 + suffix)
      return nextSlice(maxUTF8Count: desired)
    }
    
    mutating func nextWellSizedChunk(suffix: Int = 0) -> _Chunk? {
      guard let slice = nextWellSizedSlice(suffix: suffix) else { return nil }
      return _Chunk(slice)
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension String {
  func _nextSlice(
    after i: Index,
    limit: Index,
    maxUTF8Count: Int
  ) -> Range<Index>? {
    assert(maxUTF8Count >= 0)
    assert(i._isKnownScalarAligned)
    guard i < limit else { return nil }
    let end = self.utf8.index(i, offsetBy: maxUTF8Count, limitedBy: limit) ?? limit
    let j = self.unicodeScalars._index(roundingDown: end)
    return Range(uncheckedBounds: (i, j))
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  init(_ string: String) {
    guard !string.isEmpty else { self.init(); return }
    assert(string.utf8.count <= Self.maxUTF8Count)
    var ingester = BigString._Ingester(string)
    self = ingester.nextChunk()!
    assert(ingester.isAtEnd)
  }
}
