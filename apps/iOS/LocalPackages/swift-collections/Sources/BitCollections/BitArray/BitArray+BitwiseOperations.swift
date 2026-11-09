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

#if false
// FIXME: Bitwise operators disabled for now. I have two concerns:
// 1. We need to support bitwise operations over slices of bit arrays, not just
//    whole arrays.
// 2. We need to put in-place mutations as the primary operation, and they
//    need to avoid copy-on-write copies unless absolutely necessary.
//
// It seems unlikely that the operator syntax will survive these points.
// We have five (5!) separate cases:
//
//     foo |= bar
//     foo[i ..< j] |= bar
//     foo |= bar[u ..< v]
//     foo[i ..< j] |= bar[u ..< v]
//     foo[i ..< j] |= foo[k ..< l]
//
// The last one where the array is ORed with itself is particularly problematic
// -- like memcpy, these operations can easily support overlapping inputs, but
// it doesn't seem likely we can implement that with this nice slicing syntax,
// unless we are okay with forcing a CoW copy. (Which we aren't.)
//
// Even ignoring that, I would not like to end up with four overloads for each
// operator, especially not for such niche operations. So we'll entirely disable
// these for now, to prevent any shipping API from interfering with an eventual
// redesign. (This is an active area of experimentation, as it will potentially
// also affect our non-copyable container design.)

extension BitArray {
  /// Stores the result of performing a bitwise OR operation on two
  /// equal-sized bit arrays in the left-hand-side variable.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Complexity: O(left.count)
  public static func |=(left: inout Self, right: Self) {
    precondition(left.count == right.count)
    left._update { target in
      right._read { source in
        for i in 0 ..< target._words.count {
          target._mutableWords[i].formUnion(source._words[i])
        }
      }
    }
    left._checkInvariants()
  }

  /// Returns the result of performing a bitwise OR operation on two
  /// equal-sized bit arrays.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Returns: The bitwise OR of `left` and `right`.
  /// - Complexity: O(left.count)
  public static func |(left: Self, right: Self) -> Self {
    precondition(left.count == right.count)
    var result = left
    result |= right
    return result
  }

  /// Stores the result of performing a bitwise AND operation on two given
  /// equal-sized bit arrays in the left-hand-side variable.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Complexity: O(left.count)
  public static func &=(left: inout Self, right: Self) {
    precondition(left.count == right.count)
    left._update { target in
      right._read { source in
        for i in 0 ..< target._words.count {
          target._mutableWords[i].formIntersection(source._words[i])
        }
      }
    }
    left._checkInvariants()
  }

  /// Returns the result of performing a bitwise AND operation on two
  /// equal-sized bit arrays.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Returns: The bitwise AND of `left` and `right`.
  /// - Complexity: O(left.count)
  public static func &(left: Self, right: Self) -> Self {
    precondition(left.count == right.count)
    var result = left
    result &= right
    return result
  }

  /// Stores the result of performing a bitwise XOR operation on two given
  /// equal-sized bit arrays in the left-hand-side variable.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Complexity: O(left.count)
  public static func ^=(left: inout Self, right: Self) {
    precondition(left.count == right.count)
    left._update { target in
      right._read { source in
        for i in 0 ..< target._words.count {
          target._mutableWords[i].formSymmetricDifference(source._words[i])
        }
      }
    }
    left._checkInvariants()
  }

  /// Returns the result of performing a bitwise XOR operation on two
  /// equal-sized bit arrays.
  ///
  /// - Parameter left: A bit array.
  /// - Parameter right: Another bit array of the same size.
  /// - Returns: The bitwise XOR of `left` and `right`.
  /// - Complexity: O(left.count)
  public static func ^(left: Self, right: Self) -> Self {
    precondition(left.count == right.count)
    var result = left
    result ^= right
    return result
  }
}

extension BitArray {
  /// Returns the complement of the given bit array.
  ///
  /// - Parameter value: A bit array.
  /// - Returns: A bit array constructed by flipping each bit in `value`.
  ///     flipped.
  /// - Complexity: O(value.count)
  public static prefix func ~(value: Self) -> Self {
    var result = value
    result.toggleAll()
    return result
  }
}
#endif

extension BitArray {
  public mutating func toggleAll() {
    _update { handle in
      let w = handle._mutableWords
      for i in 0 ..< handle._words.count {
        w[i].formComplement()
      }
      let p = handle.end
      if p.bit > 0 {
        w[p.word].subtract(_Word(upTo: p.bit).complement())
      }
    }
    _checkInvariants()
  }

  public mutating func toggleAll(in range: Range<Int>) {
    precondition(range.upperBound <= count, "Range out of bounds")
    guard !range.isEmpty else { return }
    _update { handle in
      let words = handle._mutableWords
      let start = _BitPosition(range.lowerBound)
      let end = _BitPosition(range.upperBound)
      if start.word == end.word {
        let bits = _Word(from: start.bit, to: end.bit)
        words[start.word].formSymmetricDifference(bits)
        return
      }
      words[start.word].formSymmetricDifference(
        _Word(upTo: start.bit).complement())
      for i in stride(from: start.word + 1, to: end.word, by: 1) {
        words[i].formComplement()
      }
      if end.bit > 0 {
        words[end.word].formSymmetricDifference(_Word(upTo: end.bit))
      }
    }
  }

  @inlinable
  public mutating func toggleAll(in range: some RangeExpression<Int>) {
    toggleAll(in: range.relative(to: self))
  }
}
