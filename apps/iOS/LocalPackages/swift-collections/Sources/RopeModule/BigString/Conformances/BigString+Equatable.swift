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
  public func isIdentical(to other: Self) -> Bool {
    self._rope.isIdentical(to: other._rope)
  }
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString: Equatable {
  public static func ==(left: Self, right: Self) -> Bool {
    // FIXME: Implement properly normalized comparisons & hashing.
    // This is somewhat tricky as we shouldn't just normalize individual pieces of the string
    // split up on random Character boundaries -- Unicode does not promise that
    // norm(a + c) == norm(a) + norm(b) in this case.
    // To do this properly, we'll probably need to expose new stdlib entry points. :-/
    if left.isIdentical(to: right) { return true }
    guard left._characterCount == right._characterCount else { return false }
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
}

@available(macOS 13.3, iOS 16.4, watchOS 9.4, tvOS 16.4, *)
extension BigString {
  /// Lexicographically compare the UTF-8 representations of `left` to `right`, returning a Boolean
  /// value indicating whether `left` is equal to `right`.
  internal static func utf8IsEqual(_ left: Self, to right: Self) -> Bool {
    if left.isIdentical(to: right) { return true }
    guard left._rope.summary == right._rope.summary else { return false }

    // FIXME: Implement a structural comparison that ignores shared subtrees when possible.
    // This is somewhat tricky as this is only possible to do when the shared subtrees start
    // at the same logical position in both trees. Additionally, the two input trees may not
    // have the same height, even if they are equal.

    var it1 = left.utf8.makeIterator()
    var it2 = right.utf8.makeIterator()
    var remaining = left._utf8Count

    while remaining > 0 {
      let consumed = it1.next(maximumCount: remaining) { b1 in
        let consumed = it2.next(maximumCount: b1.count) { b2 in
          guard b2.elementsEqual(b1.prefix(b2.count)) else { return (0, 0) }
          return (b2.count, b2.count)
        }
        return (consumed, consumed)
      }
      guard consumed > 0 else { return false }
      remaining -= consumed
    }
    return true
  }

  internal static func utf8IsEqual(
    _ left: Self,
    in leftRange: Range<Index>,
    to right: Self,
    in rightRange: Range<Index>
  ) -> Bool {
    let leftUTF8Count = left._utf8Distance(from: leftRange.lowerBound, to: leftRange.upperBound)
    let rightUTF8Count = right._utf8Distance(from: rightRange.lowerBound, to: rightRange.upperBound)
    guard leftUTF8Count == rightUTF8Count else { return false }

    var remaining = leftUTF8Count
    var it1 = BigString.UTF8View.Iterator(_base: left, from: leftRange.lowerBound)
    var it2 = BigString.UTF8View.Iterator(_base: right, from: rightRange.lowerBound)

    while remaining > 0 {
      let consumed = it1.next(maximumCount: remaining) { b1 in
        let consumed = it2.next(maximumCount: b1.count) { b2 in
          guard b2.elementsEqual(b1.prefix(b2.count)) else { return (0, 0) }
          return (b2.count, b2.count)
        }
        return (consumed, consumed)
      }
      guard consumed > 0 else { return false }
      remaining -= consumed
    }
    return true
  }
}
