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
extension BigString._Chunk: RopeElement {
  typealias Summary = BigString.Summary
  typealias Index = String.Index

  var summary: BigString.Summary {
    Summary(self)
  }

  var isEmpty: Bool { string.isEmpty }

  var isUndersized: Bool { utf8Count < Self.minUTF8Count }

  func invariantCheck() {
#if COLLECTIONS_INTERNAL_CHECKS
    precondition(string.endIndex._canBeUTF8)
    let c = utf8Count
    if c == 0 {
      precondition(counts == Counts(), "Non-empty counts")
      return
    }
    precondition(c <= Self.maxUTF8Count, "Oversized chunk")
    //precondition(utf8Count >= Self.minUTF8Count, "Undersized chunk")

    precondition(counts.utf8 == string.utf8.count, "UTF-8 count mismatch")
    precondition(counts.utf16 == string.utf16.count, "UTF-16 count mismatch")
    precondition(counts.unicodeScalars == string.unicodeScalars.count, "Scalar count mismatch")

    precondition(counts.prefix <= c, "Invalid prefix count")
    precondition(counts.suffix <= c && counts.suffix > 0, "Invalid suffix count")
    if Int(counts.prefix) + Int(counts.suffix) <= c {
      let i = firstBreak
      let j = lastBreak
      precondition(i <= j, "Overlapping prefix and suffix")
      let s = string[i...]
      precondition(counts.characters == s.count, "Inconsistent character count")
      precondition(j == s.index(before: s.endIndex), "Inconsistent suffix count")
    } else {
      // Anomalous case
      precondition(counts.prefix == c, "Inconsistent prefix count (continuation)")
      precondition(counts.suffix == c, "Inconsistent suffix count (continuation)")
      precondition(counts.characters == 0, "Inconsistent character count (continuation)")
    }
#endif
  }

  mutating func rebalance(nextNeighbor right: inout Self) -> Bool {
    if self.isEmpty {
      swap(&self, &right)
      return true
    }
    guard !right.isEmpty else { return true }
    guard self.isUndersized || right.isUndersized else { return false }
    let sum = self.utf8Count + right.utf8Count
    let desired = BigString._Ingester.desiredNextChunkSize(remaining: sum)

    precondition(desired != self.utf8Count)
    if desired < self.utf8Count {
      let i = self.string._utf8Index(at: desired)
      let j = self.string.unicodeScalars._index(roundingDown: i)
      Self._redistributeData(&self, &right, splittingLeftAt: j)
    } else {
      let i = right.string._utf8Index(at: desired - self.utf8Count)
      let j = right.string.unicodeScalars._index(roundingDown: i)
      Self._redistributeData(&self, &right, splittingRightAt: j)
    }
    assert(right.isEmpty || (!self.isUndersized && !right.isUndersized))
    return right.isEmpty
  }

  mutating func rebalance(prevNeighbor left: inout Self) -> Bool {
    if self.isEmpty {
      swap(&self, &left)
      return true
    }
    guard !left.isEmpty else { return true }
    guard left.isUndersized || self.isUndersized else { return false }
    let sum = left.utf8Count + self.utf8Count
    let desired = BigString._Ingester.desiredNextChunkSize(remaining: sum)

    precondition(desired != self.utf8Count)
    if desired < self.utf8Count {
      let i = self.string._utf8Index(at: self.utf8Count - desired)
      let j = self.string.unicodeScalars._index(roundingDown: i)
      let k = (i == j ? i : self.string.unicodeScalars.index(after: j))
      Self._redistributeData(&left, &self, splittingRightAt: k)
    } else {
      let i = left.string._utf8Index(at: left.utf8Count + self.utf8Count - desired)
      let j = left.string.unicodeScalars._index(roundingDown: i)
      let k = (i == j ? i : left.string.unicodeScalars.index(after: j))
      Self._redistributeData(&left, &self, splittingLeftAt: k)
    }
    assert(left.isEmpty || (!left.isUndersized && !self.isUndersized))
    return left.isEmpty
  }

  mutating func split(at i: String.Index) -> Self {
    assert(i == string.unicodeScalars._index(roundingDown: i))
    let c = splitCounts(at: i)
    let new = Self(string[i...], c.right)
    self = Self(string[..<i], c.left)
    return new
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString._Chunk {
  static func _redistributeData(
    _ left: inout Self,
    _ right: inout Self,
    splittingRightAt i: String.Index
  ) {
    assert(i == right.string.unicodeScalars._index(roundingDown: i))
    assert(i > right.string.startIndex)
    guard i < right.string.endIndex else {
      left.append(right)
      right.clear()
      return
    }
    let counts = right.splitCounts(at: i)
    left._append(right.string[..<i], counts.left)
    right = Self(right.string[i...], counts.right)
  }

  static func _redistributeData(
    _ left: inout Self,
    _ right: inout Self,
    splittingLeftAt i: String.Index
  ) {
    assert(i == left.string.unicodeScalars._index(roundingDown: i))

    assert(i < left.string.endIndex)
    guard i > left.string.startIndex else {
      left.append(right)
      right.clear()
      swap(&left, &right)
      return
    }
    let counts = left.splitCounts(at: i)
    right._prepend(left.string[i...], counts.right)
    left = Self(left.string[..<i], counts.left)
  }
}
