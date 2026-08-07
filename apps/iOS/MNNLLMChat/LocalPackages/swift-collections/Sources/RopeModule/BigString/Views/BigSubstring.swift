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
public struct BigSubstring: Sendable {
  var _base: BigString
  var _bounds: Range<Index>

  public init(_unchecked base: BigString, in bounds: Range<Index>) {
    assert(bounds.lowerBound == base.index(roundingDown: bounds.lowerBound))
    assert(bounds.upperBound == base.index(roundingDown: bounds.upperBound))
    self._base = base
    self._bounds = bounds
  }

  public init(_ base: BigString, in bounds: Range<Index>) {
    self._base = base
    // Sub-character slicing could change character boundaries in the tree, requiring
    // resyncing metadata. This would not be acceptable to do during slicing, so let's
    // round substring bounds down to the nearest character.
    let start = base.index(roundingDown: bounds.lowerBound)
    let end = base.index(roundingDown: bounds.upperBound)
    self._bounds = Range(uncheckedBounds: (start, end))
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring {
  public var base: BigString { _base }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring {
  func _foreachChunk(
    _ body: (Substring) -> Void
  ) {
    self._base._foreachChunk(from: _bounds.lowerBound, to: _bounds.upperBound, body)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: CustomStringConvertible {
  public var description: String {
    String(_from: _base, in: _bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: CustomDebugStringConvertible {
  public var debugDescription: String {
    description.debugDescription
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self.init(value)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: LosslessStringConvertible {
  // init?(_: String) is implemented by RangeReplaceableCollection.init(_:)
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    // FIXME: Implement properly normalized comparisons & hashing.
    // This is somewhat tricky as we shouldn't just normalize individual pieces of the string
    // split up on random Character boundaries -- Unicode does not promise that
    // norm(a + c) == norm(a) + norm(b) in this case.
    // To do this properly, we'll probably need to expose new stdlib entry points. :-/
    if left.isIdentical(to: right) { return true }

    guard left.count == right.count else { return false }

    // FIXME: Even if we keep doing characterwise comparisons, we should skip over shared subtrees.
    var it1 = left.makeIterator()
    var it2 = right.makeIterator()
    var a: Character? = nil
    var b: Character? = nil
    repeat {
      a = it1.next()
      b = it2.next()
      guard a == b else { return false }
    } while a != nil
    return true
  }

  public func isIdentical(to other: Self) -> Bool {
    guard self._base.isIdentical(to: other._base) else { return false }
    return self._bounds == other._bounds
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: Hashable {
  public func hash(into hasher: inout Hasher) {
    var it = self.makeIterator()
    while let character = it.next() {
      let s = String(character)
      s._withNFCCodeUnits { hasher.combine($0) }
    }
    hasher.combine(0xFF as UInt8)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: Comparable {
  public static func < (left: Self, right: Self) -> Bool {
    // FIXME: Implement properly normalized comparisons & hashing.
    // This is somewhat tricky as we shouldn't just normalize individual pieces of the string
    // split up on random Character boundaries -- Unicode does not promise that
    // norm(a + c) == norm(a) + norm(b) in this case.
    // To do this properly, we'll probably need to expose new stdlib entry points. :-/
    if left.isIdentical(to: right) { return false }
    // FIXME: Even if we keep doing characterwise comparisons, we should skip over shared subtrees.
    var it1 = left.makeIterator()
    var it2 = right.makeIterator()
    while true {
      switch (it1.next(), it2.next()) {
      case (nil, nil): return false
      case (nil, .some): return true
      case (.some, nil): return false
      case let (a?, b?):
        if a == b { continue }
        return a < b
      }
    }
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: Sequence {
  public typealias Element = Character

  public struct Iterator: IteratorProtocol {
    let _end: BigString.Index
    var _it: BigString.Iterator

    init(_substring: BigSubstring) {
      self._it = BigString.Iterator(_substring._base, from: _substring.startIndex)
      self._end = _substring.endIndex
    }

    public mutating func next() -> Character? {
      guard _it.isBelow(_end) else { return nil }
      return _it.next()
    }
  }

  public func makeIterator() -> Iterator {
    Iterator(_substring: self)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: BidirectionalCollection {
  public typealias Index = BigString.Index
  public typealias SubSequence = Self

  @inline(__always)
  public var startIndex: Index { _bounds.lowerBound }

  @inline(__always)
  public var endIndex: Index { _bounds.upperBound }

  public var count: Int {
    distance(from: _bounds.lowerBound, to: _bounds.upperBound)
  }

  @inline(__always)
  public func index(after i: Index) -> Index {
    precondition(i < endIndex, "Can't advance above end index")
    return _base.index(after: i)
  }

  @inline(__always)
  public func index(before i: Index) -> Index {
    precondition(i > startIndex, "Can't advance below start index")
    return _base.index(before: i)
  }

  @inline(__always)
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    let j = _base.index(i, offsetBy: distance)
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func index(_ i: Index, offsetBy distance: Int, limitedBy limit: Index) -> Index? {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    guard let j = _base.index(i, offsetBy: distance, limitedBy: limit) else { return nil }
    precondition(j >= startIndex && j <= endIndex, "Index out of bounds")
    return j
  }

  public func distance(from start: Index, to end: Index) -> Int {
    precondition(start >= startIndex && start <= endIndex, "Index out of bounds")
    precondition(end >= startIndex && end <= endIndex, "Index out of bounds")
    return _base.distance(from: start, to: end)
  }

  public subscript(position: Index) -> Character {
    precondition(position >= startIndex && position < endIndex, "Index out of bounds")
    return _base[position]
  }

  public subscript(bounds: Range<Index>) -> Self {
    precondition(
      bounds.lowerBound >= startIndex && bounds.upperBound <= endIndex,
      "Range out of bounds")
    return Self(_base, in: bounds)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring {
  public func index(roundingDown i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base.index(roundingDown: i)
  }

  public func index(roundingUp i: Index) -> Index {
    precondition(i >= startIndex && i <= endIndex, "Index out of bounds")
    return _base.index(roundingUp: i)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring {
  /// Run the closure `body` to mutate the contents of this view within `range`, then update
  /// the bounds of this view to maintain an approximation of their logical position in the
  /// resulting string.
  ///
  /// The `range` argument is validated to be within the original bounds of the substring.
  internal mutating func _mutateBasePreservingBounds<R>(
    in range: Range<Index>,
    with body: (inout BigString) -> R
  ) -> R {
    precondition(
      range.lowerBound >= _bounds.lowerBound && range.upperBound <= _bounds.upperBound,
      "Range out of bounds")

    let startOffset = self.startIndex.utf8Offset
    let endOffset = self.endIndex.utf8Offset
    let oldCount = self._base._utf8Count

    defer {
      // Substring mutations may change grapheme boundaries across the bounds of the original
      // substring value, and we need to ensure that the substring's bounds remain well-aligned.
      // Unfortunately, there are multiple ways of doing this, none of which are obviously
      // superior to others. To keep the behavior easier to explan, we emulate substring
      // initialization and round the start and end indices down to the nearest Character boundary
      // after each mutation.
      let delta = self._base._utf8Count - oldCount
      let start = _base.index(roundingDown: Index(_utf8Offset: startOffset))
      let end = _base.index(roundingDown: Index(_utf8Offset: endOffset + delta))
      self._bounds = start ..< end
    }
    return body(&self._base)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigSubstring: RangeReplaceableCollection {
  public init() {
    let str = BigString()
    let bounds = Range(uncheckedBounds: (str.startIndex, str.endIndex))
    self.init(_unchecked: str, in: bounds)
  }

  public mutating func reserveCapacity(_ n: Int) {
    // Do nothing.
  }

  public mutating func replaceSubrange( // Note: Sequence, not Collection
    _ subrange: Range<Index>,
    with newElements: __owned some Sequence<Character>
  ) {
    _mutateBasePreservingBounds(in: subrange) {
      $0.replaceSubrange(subrange, with: newElements)
    }
  }

  public init(_ elements: some Sequence<Character>) {
    let base = BigString(elements)
    self.init(base, in: base.startIndex ..< base.endIndex)
  }

  public init(repeating repeatedValue: Character, count: Int) {
    self.init(BigString(repeating: repeatedValue, count: count))
  }

  public init(repeating repeatedValue: some StringProtocol, count: Int) {
    self.init(BigString(repeating: repeatedValue, count: count))
  }

  public init(repeating repeatedValue: BigString, count: Int) {
    self.init(BigString(repeating: repeatedValue, count: count))
  }

  public init(repeating repeatedValue: BigSubstring, count: Int) {
    self.init(BigString(repeating: repeatedValue, count: count))
  }

  public mutating func append(_ newElement: Character) {
    let i = endIndex
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(newElement, at: i)
    }
  }

  public mutating func append(contentsOf newElements: __owned some Sequence<Character>) {
    let i = endIndex
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(contentsOf: newElements, at: i)
    }
  }

  public mutating func insert(_ newElement: Character, at i: Index) {
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(newElement, at: i)
    }
  }

  public mutating func insert(
    contentsOf newElements: __owned some Sequence<Character>, // Note: Sequence, not Collection
    at i: Index
  ) {
    _mutateBasePreservingBounds(in: i ..< i) {
      $0.insert(contentsOf: newElements, at: i)
    }
  }

  @discardableResult
  public mutating func remove(at i: Index) -> Character {
    let j = self.index(after: i)
    return _mutateBasePreservingBounds(in: i ..< j) {
      $0.remove(at: i)
    }
  }

  public mutating func removeSubrange(_ bounds: Range<Index>) {
    _mutateBasePreservingBounds(in: bounds) {
      $0.removeSubrange(bounds)
    }
  }

  public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
    let bounds = self._bounds
    _mutateBasePreservingBounds(in: bounds) {
      $0.removeSubrange(bounds)
    }
    assert(_bounds.isEmpty)
  }
}
