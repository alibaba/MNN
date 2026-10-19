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
  mutating func _append(contentsOf other: __owned Substring) {
    if other.isEmpty { return }
    if isEmpty {
      self = Self(other)
      return
    }
    var ingester = _ingester(forInserting: other, at: endIndex, allowForwardPeek: true)
    
    let last = _rope.index(before: _rope.endIndex)
    if let final = _rope[last].append(from: &ingester) {
      precondition(!final.isUndersized)
      _rope.append(final)
      return
    }
    
    // Make a temp rope out of the rest of the chunks and then join the two trees together.
    if !ingester.isAtEnd {
      var builder = _Rope.Builder()
      while let chunk = ingester.nextWellSizedChunk() {
        precondition(!chunk.isUndersized)
        builder.insertBeforeTip(chunk)
      }
      precondition(ingester.isAtEnd)
      _rope = _Rope.join(_rope, builder.finalize())
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  var _firstUnicodeScalar: Unicode.Scalar {
    assert(!isEmpty)
    return _rope.root.firstItem.value.string.unicodeScalars.first!
  }

  mutating func _append(contentsOf other: __owned BigString) {
    guard !other.isEmpty else { return }
    guard !self.isEmpty else {
      self = other
      return
    }

    let hint = other._firstUnicodeScalar
    var other = other._rope    
    var old = _CharacterRecognizer()
    var new = self._breakState(upTo: endIndex, nextScalarHint: hint)
    _ = other.resyncBreaks(old: &old, new: &new)
    _append(other)
  }

  mutating func _append(contentsOf other: __owned BigString, in range: Range<Index>) {
    guard !range._isEmptyUTF8 else { return }
    guard !self.isEmpty else {
      self = Self(_from: other, in: range)
      return
    }

    var other = BigString(_from: other, in: range)
    let hint = other._firstUnicodeScalar
    var old = _CharacterRecognizer()
    var new = self._breakState(upTo: endIndex, nextScalarHint: hint)
    _ = other._rope.resyncBreaks(old: &old, new: &new)
    _append(other._rope)
  }

  mutating func prepend(contentsOf other: __owned BigString) {
    guard !other.isEmpty else { return }
    guard !self.isEmpty else {
      self = other
      return
    }
    let hint = self._firstUnicodeScalar
    var old = _CharacterRecognizer()
    var new = other._breakState(upTo: other.endIndex, nextScalarHint: hint)
    _ = self._rope.resyncBreaks(old: &old, new: &new)
    _prepend(other._rope)
  }

  mutating func prepend(contentsOf other: __owned BigString, in range: Range<Index>) {
    guard !range._isEmptyUTF8 else { return }
    let extract = Self(_from: other, in: range)
    guard !self.isEmpty else {
      self = extract
      return
    }
    let hint = self._firstUnicodeScalar
    var old = _CharacterRecognizer()
    var new = extract._breakState(upTo: extract.endIndex, nextScalarHint: hint)
    _ = self._rope.resyncBreaks(old: &old, new: &new)
    _prepend(extract._rope)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  var isUndersized: Bool {
    _utf8Count < _Chunk.minUTF8Count
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  /// Note: This assumes `other` already has the correct break positions.
  mutating func _append(_ other: __owned _Chunk) {
    assert(!other.isEmpty)
    guard !self.isEmpty else {
      self._rope.append(other)
      return
    }
    guard self.isUndersized || other.isUndersized else {
      self._rope.append(other)
      return
    }
    var other = other
    let last = self._rope.index(before: self._rope.endIndex)
    if !self._rope[last].rebalance(nextNeighbor: &other) {
      assert(!other.isUndersized)
      self._rope.append(other)
    }
  }
  
  /// Note: This assumes `self` and `other` already have the correct break positions.
  mutating func _prepend(_ other: __owned _Chunk) {
    assert(!other.isEmpty)
    guard !self.isEmpty else {
      self._rope.prepend(other)
      return
    }
    guard self.isUndersized || other.isUndersized else {
      self._rope.prepend(other)
      return
    }
    var other = other
    let first = self._rope.startIndex
    if !self._rope[first].rebalance(prevNeighbor: &other) {
      self._rope.prepend(other)
    }
  }
  
  /// Note: This assumes `other` already has the correct break positions.
  mutating func _append(_ other: __owned _Rope) {
    guard !other.isEmpty else { return }
    guard !self._rope.isEmpty else {
      self._rope = other
      return
    }
    if other.isSingleton {
      self._append(other.first!)
      return
    }
    if self.isUndersized {
      assert(self._rope.isSingleton)
      let chunk = self._rope.first!
      self._rope = other
      self._prepend(chunk)
      return
    }
    self._rope = _Rope.join(self._rope, other)
  }
  
  /// Note: This assumes `self` and `other` already have the correct break positions.
  mutating func _prepend(_ other: __owned _Rope) {
    guard !other.isEmpty else { return }
    guard !self.isEmpty else {
      self._rope = other
      return
    }
    if other.isSingleton {
      self._prepend(other.first!)
      return
    }
    if self.isUndersized {
      assert(self._rope.isSingleton)
      let chunk = self._rope.first!
      self._rope = other
      self._append(chunk)
      return
    }
    self._rope = _Rope.join(other, self._rope)
  }
}
