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
  func _breakState(
    upTo index: Index,
    nextScalarHint: Unicode.Scalar? = nil
  ) -> _CharacterRecognizer {
    assert(index == _unicodeScalarIndex(roundingDown: index))
    guard index > startIndex else {
      return _CharacterRecognizer()
    }
    let index = resolve(index, preferEnd: true)
    let ropeIndex = index._rope!
    let chunkIndex = index._chunkIndex
    let chunk = _rope[ropeIndex]

    guard ropeIndex > _rope.startIndex || chunkIndex > chunk.string.startIndex else {
      return _CharacterRecognizer()
    }

    if let next = nextScalarHint, chunkIndex > chunk.string.startIndex {
      let i = chunk.string.unicodeScalars.index(before: chunkIndex)
      let prev = chunk.string.unicodeScalars[i]
      if _CharacterRecognizer.quickBreak(between: prev, and: next) == true {
        return _CharacterRecognizer()
      }
    }

    if let r = chunk.immediateBreakState(upTo: chunkIndex) {
      return r.state
    }
    // Find chunk that includes the start of the character.
    var ri = ropeIndex
    while ri > _rope.startIndex {
      _rope.formIndex(before: &ri)
      if _rope[ri].hasBreaks { break }
    }
    precondition(ri < ropeIndex)

    // Collect grapheme breaking state.
    var state = _CharacterRecognizer(partialCharacter: _rope[ri].suffix)
    _rope.formIndex(after: &ri)
    while ri < ropeIndex {
      state.consumePartialCharacter(_rope[ri].string[...])
      _rope.formIndex(after: &ri)
    }
    state.consumePartialCharacter(chunk.string[..<chunkIndex])
    return state
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  /// - Returns: the position at which the grapheme breaks finally sync up with the originals.
  ///  (or nil if they never did).
  @discardableResult
  mutating func resyncBreaks(
    startingAt index: Index,
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> (ropeIndex: _Rope.Index, chunkIndex: String.Index)? {
    guard index < endIndex else { return nil }
    let i = resolve(index, preferEnd: false)
    var ropeIndex = i._rope!
    var chunkIndex = i._chunkIndex
    let end = _rope.mutatingForEach(from: &ropeIndex) { chunk in
      let start = chunkIndex
      chunkIndex = String.Index(_utf8Offset: 0)
      if let i = chunk.resyncBreaks(startingAt: start, old: &old, new: &new) {
        return i
      }
      return nil
    }
    guard let end else { return nil }
    return (ropeIndex, end)
  }

  mutating func resyncBreaksToEnd(
    startingAt index: Index,
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) {
    guard let _ = resyncBreaks(startingAt: index, old: &old, new: &new) else { return }
    let state = _breakState(upTo: endIndex)
    old = state
    new = state
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Rope {
  mutating func resyncBreaks(
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> Bool {
    _resyncBreaks(old: &old, new: &new) != nil
  }

  mutating func _resyncBreaks(
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> (ropeIndex: Index, chunkIndex: String.Index)? {
    var ropeIndex = startIndex
    let chunkIndex = self.mutatingForEach(from: &ropeIndex) {
      $0.resyncBreaksFromStart(old: &old, new: &new)
    }
    guard let chunkIndex else { return nil }
    return (ropeIndex, chunkIndex)
  }

  mutating func resyncBreaksToEnd(
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) {
    guard let (ropeIndex, chunkIndex) = _resyncBreaks(old: &old, new: &new) else { return }

    let chars = self.summary.characters
    if chars > 0 {
      var i = endIndex
      while i > ropeIndex {
        formIndex(before: &i)
        if self[i].hasBreaks { break }
      }
      if i > ropeIndex || self[i].lastBreak > chunkIndex {
        new = self[i].immediateLastBreakState!
        formIndex(after: &i)
        while i < endIndex {
          new.consumePartialCharacter(self[i].string[...])
          formIndex(after: &i)
        }
        old = new
        return
      }
    }

    var ri = ropeIndex
    let suffix = self[ri].string.unicodeScalars[chunkIndex...].dropFirst()
    new.consumePartialCharacter(suffix)
    formIndex(after: &ri)
    while ri < endIndex {
      new.consumePartialCharacter(self[ri].string[...])
      formIndex(after: &ri)
    }
    old = new
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  /// Resyncronize chunk metadata with the (possibly) reshuffled grapheme
  /// breaks after an insertion that ended at `index`.
  ///
  /// This assumes that the chunk's prefix and suffix counts have already
  /// been adjusted to remain on Unicode scalar boundaries, and that they
  /// are also in sync with the grapheme breaks up to `index`. If the
  /// prefix ends after `index`, then this function may update it to address
  /// the correct scalar. Similarly, the suffix count may be updated to
  /// reflect the new position of the last grapheme break, if necessary.
  mutating func resyncBreaks(
    startingAt index: String.Index,
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> String.Index? {
    var i = index
    let u = string.unicodeScalars
    assert(u._index(roundingDown: i) == i)

    // FIXME: Rewrite in terms of `firstBreak(in:)`.
    var first: String.Index? = nil
    var last: String.Index? = nil
  loop:
    while i < u.endIndex {
      let scalar = u[i]
      let a = old.hasBreak(before: scalar)
      let b = new.hasBreak(before: scalar)
      if b {
        first = first ?? i
        last = i
      }
      switch (a, b) {
      case (true, true):
        break loop // Resync complete ✨
      case (false, false):
        if old._isKnownEqual(to: new) { break loop }
      case (false, true):
        counts._characters += 1
      case (true, false):
        counts._characters -= 1
      }
      u.formIndex(after: &i)
    }

    // Grapheme breaks in `index...i` may have changed. Update `firstBreak` and `lastBreak`
    // accordingly.

    assert((first == nil) == (last == nil))
    if index <= firstBreak {
      if let first {
        // We have seen the new first break.
        firstBreak = first
      } else if i >= firstBreak {
        // The old firstBreak is no longer a break.
        if i < lastBreak {
          // The old lastBreak is still valid. Find the first break in i+1...endIndex.
          let j = u.index(after: i)
          var tmp = new
          firstBreak = tmp.firstBreak(in: string[j...])!.lowerBound
        } else {
          // No breaks anywhere in the string.
          firstBreak = u.endIndex
        }
      }
    }

    if i >= lastBreak {
      if let last {
        // We have seen the new last break.
        lastBreak = last
      } else if firstBreak == u.endIndex {
        // We already know there are no breaks anywhere in the string.
        lastBreak = u.startIndex
      } else {
        // We have a `firstBreak`, but no break in `index...i`.
        // Find last break in `firstBreak...<index`.
        let wholeChars = self.string[firstBreak..<index]
        lastBreak = wholeChars.index(before: wholeChars.endIndex)
      }
    }

    return i < u.endIndex ? i : nil
  }

  mutating func resyncBreaksFromStart(
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) -> String.Index? {
    resyncBreaks(startingAt: string.startIndex, old: &old, new: &new)
  }

  mutating func resyncBreaksFromStartToEnd(
    old: inout _CharacterRecognizer,
    new: inout _CharacterRecognizer
  ) {
    guard let i = resyncBreaks(startingAt: string.startIndex, old: &old, new: &new) else {
      return
    }
    if i < lastBreak {
      new = _CharacterRecognizer(partialCharacter: string[lastBreak...])
    } else {
      //assert(old == new)
      let j = string.unicodeScalars.index(after: i)
      new.consumePartialCharacter(string[j...])
    }
    old = new
    return
  }
}
