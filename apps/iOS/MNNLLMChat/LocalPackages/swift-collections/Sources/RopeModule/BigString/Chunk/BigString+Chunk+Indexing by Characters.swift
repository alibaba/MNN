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
extension UInt8 {
  /// Returns true if this is a leading code unit in the UTF-8 encoding of a Unicode scalar that
  /// is outside the BMP.
  var _isUTF8NonBMPLeadingCodeUnit: Bool { self >= 0b11110000 }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  func characterDistance(
    from start: String.Index,
    to end: String.Index
  ) -> Int {
    // Note: both indices must already have been rounded to a character
    // boundary.
    if start == end { return 0 }
    // Character rounding never ends up in a chunk with no breaks, unless both
    // indices are at the very end of an overlong character. (Handled above.)
    assert(hasBreaks)
    let firstBreak = self.firstBreak
    let lastBreak = self.lastBreak

    assert(start <= end)

    assert(start >= firstBreak && (start <= lastBreak || start._utf8Offset == self.utf8Count),
           "start is not Character aligned")
    assert(end >= firstBreak && (end <= lastBreak || end._utf8Offset == self.utf8Count),
           "end is not Character aligned")

    let wholeCharacters = self.wholeCharacters
    let s = Swift.max(start, firstBreak)
    let e = Swift.min(end, lastBreak)
    var result: Int
    // Run the actual grapheme breaking algorithm, making sure we measure
    // as little data as possible, by making best use of the known character
    // count for the whole chunk.
    if e._utf8Offset - s._utf8Offset <= (lastBreak._utf8Offset - firstBreak._utf8Offset) / 2 {
      result = wholeCharacters.distance(from: s, to: e)
    } else {
      result = characterCount - 1 // The last break is handled below
      result -= wholeCharacters.distance(from: firstBreak, to: s)
      result -= wholeCharacters.distance(from: e, to: lastBreak)
    }

    if end > lastBreak {
      /// We know `end` is rounded, so it can only ever be after the last break
      /// if the final partial character actually ends with this chunk.
      /// In that case, we need to include that extra character here.
      assert(end._utf8Offset == self.utf8Count)
      result += 1
    }
    return result
  }

  /// If this returns false, the next position is on the first grapheme break following this
  /// chunk.
  func formCharacterIndex(after i: inout String.Index) -> Bool {
    if i >= lastBreak {
      i = string.endIndex
      return false
    }
    let first = firstBreak
    if i < first {
      i = first
      return true
    }
    wholeCharacters.formIndex(after: &i)
    return true
  }

  /// If this returns false, the right position is `distance` steps from the first grapheme break
  /// following this chunk if `distance` was originally positive. Otherwise the right position is
  /// `-distance` steps from the first grapheme break preceding this chunk.
  func formCharacterIndex(
    _ i: inout String.Index, offsetBy distance: inout Int
  ) -> (found: Bool, forward: Bool) {
    // FIXME: Make use of the chunk's known character count to reduce work
    if distance == 0 {
      if i < firstBreak {
        i = string.startIndex
        return (false, false)
      }
      if i >= lastBreak {
        i = lastBreak
        return (true, false)
      }
      i = wholeCharacters._index(roundingDown: i)
      return (true, false)
    }
    if distance > 0 {
      if i >= lastBreak {
        i = string.endIndex
        distance -= 1
        return (false, true)
      }
      if i < firstBreak {
        i = firstBreak
        distance -= 1
        if distance == 0 { return (true, true) }
      }
      if
        distance <= characterCount,
        let r = wholeCharacters.index(i, offsetBy: distance, limitedBy: string.endIndex)
      {
        i = r
        distance = 0
        return (i < string.endIndex, true)
      }
      distance -= wholeCharacters.distance(from: i, to: lastBreak) + 1
      i = string.endIndex
      return (false, true)
    }
    if i <= firstBreak {
      i = string.startIndex
      if i == firstBreak { distance += 1 }
      return (false, false)
    }
    if i > lastBreak {
      i = lastBreak
      distance += 1
      if distance == 0 { return (true, false) }
    }
    if
      distance.magnitude <= characterCount,
      let r = self.wholeCharacters.index(i, offsetBy: distance, limitedBy: firstBreak)
    {
      i = r
      distance = 0
      return (true, false)
    }
    distance += self.wholeCharacters.distance(from: firstBreak, to: i)
    i = string.startIndex
    return (false, false)
  }
}
