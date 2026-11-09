//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension BitSet {
  @usableFromInline
  internal typealias _UnsafeHandle = _UnsafeBitSet
}

extension _UnsafeBitSet {
  @inline(__always)
  internal func _isReachable(_ index: Index) -> Bool {
    index == endIndex || contains(index.value)
  }
}

extension _UnsafeBitSet {
  internal func _emptySuffix() -> Int {
    var i = wordCount - 1
    while i >= 0, _words[i].isEmpty {
      i -= 1
    }
    return wordCount - 1 - i
  }
}

extension _UnsafeBitSet {
  @inlinable
  public mutating func combineSharedPrefix(
    with other: Self,
    using function: (inout _Word, _Word) -> Void
  ) {
    ensureMutable()
    let c = Swift.min(self.wordCount, other.wordCount)
    for w in 0 ..< c {
      function(&self._mutableWords[w], other._words[w])
    }
  }
}

extension _UnsafeBitSet {
  @_effects(releasenone)
  public mutating func formUnion(_ range: Range<UInt>) {
    ensureMutable()
    let l = Index(range.lowerBound)
    let u = Index(range.upperBound)
    assert(u.value <= capacity)

    if l.word == u.word {
      guard l.word < wordCount else { return }
      let w = _Word(from: l.bit, to: u.bit)
      _mutableWords[l.word].formUnion(w)
      return
    }
    _mutableWords[l.word].formUnion(_Word(upTo: l.bit).complement())
    for w in l.word + 1 ..< u.word {
      _mutableWords[w] = .allBits
    }
    if u.word < wordCount {
      _mutableWords[u.word].formUnion(_Word(upTo: u.bit))
    }
  }

  @_effects(releasenone)
  public mutating func formIntersection(_ range: Range<UInt>) {
    ensureMutable()
    let l = Index(Swift.min(range.lowerBound, capacity))
    let u = Index(Swift.min(range.upperBound, capacity))

    for w in 0 ..< l.word {
      _mutableWords[w] = .empty
    }

    if l.word == u.word {
      guard l.word < wordCount else { return }

      let w = _Word(from: l.bit, to: u.bit)
      _mutableWords[l.word].formIntersection(w)
      return
    }

    _mutableWords[l.word].formIntersection(_Word(upTo: l.bit).complement())
    if u.word < wordCount {
      _mutableWords[u.word].formIntersection(_Word(upTo: u.bit))
      _mutableWords[(u.word + 1)...].update(repeating: .empty)
    }
  }

  @_effects(releasenone)
  public mutating func formSymmetricDifference(_ range: Range<UInt>) {
    ensureMutable()
    let l = Index(range.lowerBound)
    let u = Index(range.upperBound)
    assert(u.value <= capacity)

    if l.word == u.word {
      guard l.word < wordCount else { return }
      let w = _Word(from: l.bit, to: u.bit)
      _mutableWords[l.word].formSymmetricDifference(w)
      return
    }
    _mutableWords[l.word]
      .formSymmetricDifference(_Word(upTo: l.bit).complement())
    for w in l.word + 1 ..< u.word {
      _mutableWords[w].formComplement()
    }
    if u.word < wordCount {
      _mutableWords[u.word].formSymmetricDifference(_Word(upTo: u.bit))
    }
  }

  @_effects(releasenone)
  public mutating func subtract(_ range: Range<UInt>) {
    ensureMutable()
    let l = Index(Swift.min(range.lowerBound, capacity))
    let u = Index(Swift.min(range.upperBound, capacity))

    if l.word == u.word {
      guard l.word < wordCount else { return }
      let w = _Word(from: l.bit, to: u.bit)
      _mutableWords[l.word].subtract(w)
      return
    }

    _mutableWords[l.word].subtract(_Word(upTo: l.bit).complement())
    _mutableWords[(l.word + 1) ..< u.word].update(repeating: .empty)
    if u.word < wordCount {
      _mutableWords[u.word].subtract(_Word(upTo: u.bit))
    }
  }

  @_effects(releasenone)
  public func isDisjoint(with range: Range<UInt>) -> Bool {
    if self.isEmpty { return true }
    let lower = Index(Swift.min(range.lowerBound, capacity))
    let upper = Index(Swift.min(range.upperBound, capacity))
    if lower == upper { return true }

    if lower.word == upper.word {
      guard lower.word < wordCount else { return true }
      let w = _Word(from: lower.bit, to: upper.bit)
      return _words[lower.word].intersection(w).isEmpty
    }

    let lw = _Word(upTo: lower.bit).complement()
    guard _words[lower.word].intersection(lw).isEmpty else { return false }

    for i in lower.word + 1 ..< upper.word {
      guard _words[i].isEmpty else { return false }
    }
    if upper.word < wordCount {
      let uw = _Word(upTo: upper.bit)
      guard _words[upper.word].intersection(uw).isEmpty else { return false }
    }
    return true
  }

  @_effects(releasenone)
  public func isSubset(of range: Range<UInt>) -> Bool {
    guard !range.isEmpty else { return isEmpty }
    guard !_words.isEmpty else { return true }
    let r = range.clamped(to: 0 ..< UInt(capacity))

    let lower = Index(r.lowerBound)
    let upper = Index(r.upperBound)

    for w in 0 ..< lower.word {
      guard _words[w].isEmpty else { return false }
    }

    guard lower.word < wordCount else { return true }

    let lw = _Word(upTo: lower.bit)
    guard _words[lower.word].intersection(lw).isEmpty else { return false }

    guard upper.word < wordCount else { return true }

    let hw = _Word(upTo: upper.bit).complement()
    guard _words[upper.word].intersection(hw).isEmpty else { return false }

    return true
  }

  @_effects(releasenone)
  public func isSuperset(of range: Range<UInt>) -> Bool {
    guard !range.isEmpty else { return true }
    let r = range.clamped(to: 0 ..< UInt(capacity))
    guard r == range else { return false }

    let lower = Index(range.lowerBound)
    let upper = Index(range.upperBound)

    if lower.word == upper.word {
      let w = _Word(from: lower.bit, to: upper.bit)
      return _words[lower.word].intersection(w) == w
    }
    let lw = _Word(upTo: lower.bit).complement()
    guard _words[lower.word].intersection(lw) == lw else { return false }

    for w in lower.word + 1 ..< upper.word {
      guard _words[w].isFull else { return false }
    }

    guard upper.word < wordCount else { return true }
    let uw = _Word(upTo: upper.bit)
    return _words[upper.word].intersection(uw) == uw
  }

  @_effects(releasenone)
  public func isEqualSet(to range: Range<UInt>) -> Bool {
    if range.isEmpty { return self.isEmpty }
    let r = range.clamped(to: 0 ..< UInt(capacity))
    guard r == range else { return false }

    let lower = Index(range.lowerBound).split
    let upper = Index(range.upperBound).endSplit

    guard upper.word == wordCount &- 1 else { return false }

    for w in 0 ..< lower.word {
      guard _words[w].isEmpty else { return false }
    }

    if lower.word == upper.word {
      let w = _Word(from: lower.bit, to: upper.bit)
      return _words[lower.word] == w
    }
    let lw = _Word(upTo: lower.bit).complement()
    guard _words[lower.word] == lw else { return false }

    for w in lower.word + 1 ..< upper.word {
      guard _words[w].isFull else { return false }
    }

    let uw = _Word(upTo: upper.bit)
    return _words[upper.word] == uw
  }
}

