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
  @inline(__always)
  var hasBreaks: Bool { counts.hasBreaks }

  var firstBreak: String.Index {
    get {
      string._utf8Index(at: prefixCount)
    }
    set {
      counts.prefix = string._utf8Offset(of: newValue)
    }
  }

  var lastBreak: String.Index {
    get {
      string._utf8Index(at: utf8Count - suffixCount)
    }
    set {
      counts.suffix = utf8Count - string._utf8Offset(of: newValue)
    }
  }

  var prefix: Substring { string[..<firstBreak] }
  var suffix: Substring { string[lastBreak...] }

  var wholeCharacters: Substring { string[firstBreak...] }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  var immediateLastBreakState: _CharacterRecognizer? {
    guard hasBreaks else { return nil }
    return _CharacterRecognizer(partialCharacter: string[lastBreak...])
  }

  func nearestBreak(before index: String.Index) -> String.Index? {
    let index = string.unicodeScalars._index(roundingDown: index)
    let first = firstBreak
    guard index > first else { return nil }
    let last = lastBreak
    guard index <= last else { return last }
    let w = string[first...]
    let rounded = w._index(roundingDown: index)
    guard rounded == index else { return rounded }
    return w.index(before: rounded)
  }

  func immediateBreakState(
    upTo index: String.Index
  ) -> (prevBreak: String.Index, state: _CharacterRecognizer)? {
    guard let prev = nearestBreak(before: index) else { return nil }
    let state = _CharacterRecognizer(partialCharacter: string[prev..<index])
    return (prev, state)
  }
}
