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
extension StringProtocol {
  @inline(__always)
  var _indexOfLastCharacter: Index {
    guard !isEmpty else { return endIndex }
    return index(before: endIndex)
  }
  
  @inline(__always)
  func _index(at offset: Int) -> Index {
    self.index(self.startIndex, offsetBy: offset)
  }
  
  @inline(__always)
  func _utf8Index(at offset: Int) -> Index {
    self.utf8.index(startIndex, offsetBy: offset)
  }
  
  @inline(__always)
  func _utf8ClampedIndex(at offset: Int) -> Index {
    self.utf8.index(startIndex, offsetBy: offset, limitedBy: endIndex) ?? endIndex
  }
  
  @inline(__always)
  func _utf8Offset(of index: Index) -> Int {
    self.utf8.distance(from: startIndex, to: index)
  }
  
  @inline(__always)
  var _lastCharacter: (index: Index, utf8Length: Int) {
    let i = _indexOfLastCharacter
    let length = utf8.distance(from: i, to: endIndex)
    return (i, length)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension String {
  internal func _lpad(to width: Int, with pad: Character = " ") -> String {
    let c = self.count
    if c >= width { return self }
    return String(repeating: pad, count: width - c) + self
  }

  internal func _rpad(to width: Int, with pad: Character = " ") -> String {
    let c = self.count
    if c >= width { return self }
    return self + String(repeating: pad, count: width - c)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension String {
  @discardableResult
  mutating func _appendQuotedProtectingLeft(
    _ str: String,
    with state: inout _CharacterRecognizer,
    maxLength: Int = Int.max
  ) -> String.Index {
    guard !str.isEmpty else { return str.endIndex }
    let startUTF8 = self.utf8.count
    var i = str.unicodeScalars.startIndex
    var needsBreak = true
    while i < str.endIndex {
      let us = str.unicodeScalars[i]
      var scalar = us.escaped(asASCII: false)
      if needsBreak {
        var t = state
        if let r = t.firstBreak(in: scalar[...]), r.lowerBound == scalar.startIndex {
        } else {
          scalar = us.escaped(asASCII: true)
        }
      }
      needsBreak = (scalar != String(us))
      self.append(scalar)
      _ = state.consume(scalar[...])
      
      str.unicodeScalars.formIndex(after: &i)
      
      let start = self._utf8Index(at: startUTF8)
      if self.distance(from: start, to: self.endIndex) >= maxLength {
        break
      }
    }
    return i
  }
  
  mutating func _appendProtectingRight(_ str: String, with state: inout _CharacterRecognizer) {
    var suffix = str
    while !self.isEmpty {
      guard let first = suffix.unicodeScalars.first else { return }
      self.unicodeScalars.append(first)
      let i = self.index(before: self.endIndex)
      if self.unicodeScalars.distance(from: i, to: self.endIndex) == 1 {
        self.unicodeScalars.append(contentsOf: suffix.unicodeScalars.dropFirst())
        break
      }
      self.unicodeScalars.removeLast()
      let last = self.unicodeScalars.removeLast()
      suffix.insert(contentsOf: last.escaped(asASCII: true), at: suffix.startIndex)
    }
  }
  
  /// A representation of the string that is suitable for debugging.
  /// This implementation differs from `String.debugDescription` by properly quoting
  /// continuation characters after the opening quotation mark and similar meta-characters.
  var _properDebugDescription: String {
    var result = "\""
    var state = _CharacterRecognizer(consuming: result)
    result._appendQuotedProtectingLeft(self, with: &state)
    result._appendProtectingRight("\"", with: &state)
    return result
  }
}
