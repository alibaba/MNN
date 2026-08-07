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
extension BigString: Comparable {
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
extension BigString {
  /// Lexicographically compare the UTF-8 representations of `left` to `right`, returning a Boolean
  /// value indicating whether `left` is ordered before `right`.
  internal func utf8IsLess(than other: Self) -> Bool {
    if self.isIdentical(to: other) { return false }

    // FIXME: Implement a structural comparison that ignores shared subtrees when possible.
    // This is somewhat tricky as this is only possible to do when the shared subtrees start
    // at the same logical position in both trees. Additionally, the two input trees may not
    // have the same height, even if they are equal.

    var it1 = self.makeChunkIterator()
    var it2 = other.makeChunkIterator()
    var s1: Substring = ""
    var s2: Substring = ""
    while true {
      if s1.isEmpty {
        s1 = it1.next()?[...] ?? ""
      }
      if s2.isEmpty {
        s2 = it2.next()?[...] ?? ""
      }
      if s1.isEmpty {
        return !s2.isEmpty
      }
      if s2.isEmpty {
        return false
      }
      let c = Swift.min(s1.utf8.count, s2.utf8.count)
      assert(c > 0)
      let r: Bool? = s1.withUTF8 { b1 in
        s2.withUTF8 { b2 in
          for i in 0 ..< c {
            let u1 = b1[i]
            let u2 = b2[i]
            if u1 < u2 { return true }
            if u1 > u2 { return false }
          }
          return nil
        }
      }
      if let r = r { return r }
      s1 = s1.suffix(from: s1._utf8Index(at: c))
      s2 = s2.suffix(from: s2._utf8Index(at: c))
    }
  }
}
