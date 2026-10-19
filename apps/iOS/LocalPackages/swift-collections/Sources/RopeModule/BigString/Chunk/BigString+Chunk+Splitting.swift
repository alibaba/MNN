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
  func splitCounts(at i: String.Index) -> (left: Counts, right: Counts) {
    precondition(i <= string.endIndex)
    guard i < string.endIndex else {
      return (self.counts, Counts())
    }
    guard i > string.startIndex else {
      return (Counts(), self.counts)
    }
    let i = string.unicodeScalars._index(roundingDown: i)

    let leftUTF16: Int
    let leftScalars: Int
    let rightUTF16: Int
    let rightScalars: Int
    if string._utf8Offset(of: i) <= utf8Count / 2 {
      leftUTF16 = string.utf16.distance(from: string.startIndex, to: i)
      rightUTF16 = self.utf16Count - leftUTF16
      leftScalars = string.unicodeScalars.distance(from: string.startIndex, to: i)
      rightScalars = self.unicodeScalarCount - leftScalars
    } else {
      rightUTF16 = string.utf16.distance(from: i, to: string.endIndex)
      leftUTF16 = self.utf16Count - rightUTF16
      rightScalars = string.unicodeScalars.distance(from: i, to: string.endIndex)
      leftScalars = self.unicodeScalarCount - rightScalars
    }

    let left = _counts(upTo: i, utf16: leftUTF16, scalars: leftScalars)
    let right = _counts(from: i, utf16: rightUTF16, scalars: rightScalars)

    assert(left.utf8 + right.utf8 == self.counts.utf8)
    assert(left.utf16 + right.utf16 == self.counts.utf16)
    assert(left.unicodeScalars + right.unicodeScalars == self.counts.unicodeScalars)
    assert(left.characters + right.characters == self.counts.characters)
    return (left, right)
  }

  func counts(upTo i: String.Index) -> Counts {
    precondition(i <= string.endIndex)
    let i = string.unicodeScalars._index(roundingDown: i)
    guard i > string.startIndex else { return Counts() }
    guard i < string.endIndex else { return self.counts }

    let utf16 = string.utf16.distance(from: string.startIndex, to: i)
    let scalars = string.unicodeScalars.distance(from: string.startIndex, to: i)
    return _counts(from: i, utf16: utf16, scalars: scalars)
  }

  func _counts(upTo i: String.Index, utf16: Int, scalars: Int) -> Counts {
    assert(i > string.startIndex && i < string.endIndex)
    let s = string[..<i]

    var result = Counts(
      utf8: s.utf8.count,
      utf16: utf16,
      unicodeScalars: scalars,
      characters: 0,
      prefix: 0,
      suffix: 0)

    let firstBreak = self.firstBreak
    let lastBreak = self.lastBreak

    if i <= firstBreak {
      result.characters = 0
      result._prefix = result.utf8
      result._suffix = result.utf8
    } else if i > lastBreak {
      result._characters = self.counts._characters
      result._prefix = self.counts._prefix
      result._suffix = self.counts._suffix - (self.counts.utf8 - result.utf8)
    } else { // i > firstBreak, i <= lastBreak
      result._prefix = self.counts._prefix
      let wholeChars = string[firstBreak ..< i]
      assert(!wholeChars.isEmpty)
      var state = _CharacterRecognizer()
      let (characters, _, last) = state.consume(wholeChars)!
      result.characters = characters
      result.suffix = string.utf8.distance(from: last, to: i)
    }
    return result
  }

  func counts(from i: String.Index) -> Counts {
    precondition(i <= string.endIndex)
    let i = string.unicodeScalars._index(roundingDown: i)
    guard i > string.startIndex else { return self.counts }
    guard i < string.endIndex else { return Counts() }

    let utf16 = string.utf16.distance(from: i, to: string.endIndex)
    let scalars = string.unicodeScalars.distance(from: i, to: string.endIndex)
    return _counts(from: i, utf16: utf16, scalars: scalars)
  }

  func _counts(from i: String.Index, utf16: Int, scalars: Int) -> Counts {
    assert(i > string.startIndex && i < string.endIndex)
    let s = string[i...]

    var result = Counts(
      utf8: s.utf8.count,
      utf16: utf16,
      unicodeScalars: scalars,
      characters: 0,
      prefix: 0,
      suffix: 0)

    let firstBreak = self.firstBreak
    let lastBreak = self.lastBreak

    if i > lastBreak {
      result._characters = 0
      result._prefix = result.utf8
      result._suffix = result.utf8
    } else if i <= firstBreak {
      result._characters = self.counts._characters
      result.prefix = self.counts.prefix - self.string._utf8Offset(of: i)
      result._suffix = self.counts._suffix
    } else { // i > firstBreak, i <= lastBreak
      result._suffix = self.counts._suffix
      let prevBreak = string[firstBreak..<lastBreak]._index(roundingDown: i)
      var state = _CharacterRecognizer(partialCharacter: string[prevBreak ..< i])
      if let (characters, first, _) = state.consume(string[i ..< lastBreak]) {
        assert(first >= i)
        result.characters = characters + 1
        result.prefix = string.utf8.distance(from: i, to: first)
      } else {
        result.characters = 1
        result.prefix = string.utf8.distance(from: i, to: lastBreak)
      }
    }
    return result
  }
}
