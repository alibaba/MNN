//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import InternalCollectionsUtilities
#endif

extension BitSet: _UniqueCollection {}

extension BitSet {
  /// Creates a new empty bit set with enough storage capacity to store values
  /// up to the given maximum value without reallocating storage.
  ///
  /// - Parameter maximumValue: The desired maximum value.
  public init(reservingCapacity maximumValue: Int) {
    self.init()
    self.reserveCapacity(maximumValue)
  }

  /// Prepares the bit set to store the specified maximum value without
  /// reallocating storage.
  ///
  /// - Parameter maximumValue: The desired maximum value.
  public mutating func reserveCapacity(_ maximumValue: Int) {
    let wc = _Word.wordCount(forBitCount: UInt(Swift.max(0, maximumValue)) + 1)
    _storage.reserveCapacity(wc)
  }
}

extension BitSet {
  /// A subscript operation for querying or updating membership in this
  /// bit set as a boolean value.
  ///
  /// This is operation is a convenience shortcut for the `contains`, `insert`
  /// and `remove` operations, enabling a uniform syntax that resembles the
  /// corresponding `BitArray` subscript operation.
  ///
  ///     var bits: BitSet = [1, 2, 3]
  ///     bits[member: 4] = true // equivalent to `bits.insert(4)`
  ///     bits[member: 2] = false // equivalent to `bits.remove(2)`
  ///     bits[member: 5].toggle()
  ///
  ///     print(bits) // [1, 3, 4, 5]
  ///     print(bits[member: 4]) // true, equivalent to `bits.contains(4)`
  ///     print(bits[member: -4]) // false
  ///     print(bits[member: 10]) // false
  ///
  /// Note that unlike `BitArray`'s subscript, this operation may dynamically
  /// resizes the underlying bitmap storage as needed.
  ///
  /// - Parameter member: An integer value. When setting membership via this
  ///    subscript, the value must be nonnegative.
  /// - Returns: `true` if the bit set contains `member`, `false` otherwise.
  /// - Complexity: O(1)
  public subscript(member member: Int) -> Bool {
    get {
      contains(member)
    }
    set {
      guard let member = UInt(exactly: member) else {
        precondition(!newValue, "Can't insert a negative value to a BitSet")
        return
      }
      if newValue {
        _ensureCapacity(forValue: member)
      } else if member >= _capacity {
        return
      }
      _updateThenShrink { handle, shrink in
        shrink = handle.update(member, to: newValue)
      }
    }
  }

  /// Accesses the contiguous subrange of the collection’s elements that are
  /// contained within a specific integer range.
  ///
  ///     let bits: BitSet = [2, 5, 6, 8, 9]
  ///     let a = bits[members: 3..<7] // [5, 6]
  ///     let b = bits[members: 4...] // [5, 6, 8, 9]
  ///     let c = bits[members: ..<8] // [2, 5, 6]
  ///
  /// This enables you to easily find the closest set member to any integer
  /// value.
  ///
  ///     let firstMemberNotLessThanFive = bits[members: 5...].first // Optional(6)
  ///     let lastMemberBelowFive = bits[members: ..<5].last // Optional(2)
  ///
  /// - Complexity: Equivalent to two invocations of `index(after:)`.
  public subscript(members bounds: Range<Int>) -> Slice<BitSet> {
    let bounds: Range<Index> = _read { handle in
      let bounds = bounds._clampedToUInt()
      var lower = _UnsafeBitSet.Index(bounds.lowerBound)
      if lower >= handle.endIndex {
        lower = handle.endIndex
      } else if !handle.contains(lower.value) {
        lower = handle.index(after: lower)
      }
      assert(lower == handle.endIndex || handle.contains(lower.value))

      var upper = _UnsafeBitSet.Index(bounds.upperBound)
      if upper <= lower {
        upper = lower
      } else if upper >= handle.endIndex {
        upper = handle.endIndex
      } else if !handle.contains(upper.value) {
        upper = handle.index(after: upper)
      }
      assert(upper == handle.endIndex || handle.contains(upper.value))
      assert(lower <= upper)
      return Range(
        uncheckedBounds: (Index(_position: lower), Index(_position: upper)))
    }
    return Slice(base: self, bounds: bounds)
  }

  /// Accesses the contiguous subrange of the collection’s elements that are
  /// contained within a specific integer range expression.
  ///
  ///     let bits: BitSet = [2, 5, 6, 8, 9]
  ///     let a = bits[members: 3..<7] // [5, 6]
  ///     let b = bits[members: 4...] // [5, 6, 8, 9]
  ///     let c = bits[members: ..<8] // [2, 5, 6]
  ///
  /// This enables you to easily find the closest set member to any integer
  /// value.
  ///
  ///     let firstMemberNotLessThanFive = bits[members: 5...].first
  ///     // Optional(6)
  ///
  ///     let lastMemberBelowFive = bits[members: ..<5].last
  ///     // Optional(2)
  ///
  /// - Complexity: Equivalent to two invocations of `index(after:)`.
  public subscript(members bounds: some RangeExpression<Int>) -> Slice<BitSet> {
    let bounds = bounds.relative(to: Int.min ..< Int.max)
    return self[members: bounds]
  }
}

extension BitSet {
  /// Removes and returns the element at the specified position.
  ///
  /// - Parameter i: The position of the element to remove. `index` must be
  ///   a valid index of the collection that is not equal to the collection's
  ///   end index.
  ///
  /// - Returns: The removed element.
  ///
  /// - Complexity: O(`1`) if the set is a unique value (with no live copies),
  ///    and the removed value is less than the largest value currently in the
  ///    set (named *max*). Otherwise the complexity is at worst O(*max*).
  @discardableResult
  public mutating func remove(at index: Index) -> Element {
    let removed = _remove(index._value)
    precondition(removed, "Invalid index")
    return Int(bitPattern: index._value)
  }

  /// Returns a new bit set containing the elements of the set that satisfy the
  /// given predicate.
  ///
  /// In this example, `filter(_:)` is used to include only even members.
  ///
  ///     let bits = BitSet(0 ..< 20)
  ///     let evens = bits.filter { $0.isMultiple(of: 2) }
  ///
  ///     evens.isSubset(of: bits) // true
  ///     evens.contains(5) // false
  ///
  /// - Parameter isIncluded: A closure that takes an element as its argument
  ///   and returns a Boolean value indicating whether the element should be
  ///   included in the returned set.
  /// - Returns: A set of the elements that `isIncluded` allows.
  public func filter(
    _ isIncluded: (Element) throws -> Bool
  ) rethrows -> Self {
    var words = [_Word](repeating: .empty, count: _storage.count)
    try words.withUnsafeMutableBufferPointer { buffer in
      var target = _UnsafeHandle(words: buffer, mutable: true)
      for i in self {
        guard try isIncluded(i) else { continue }
        target.insert(UInt(i))
      }
    }
    return BitSet(_words: words)
  }
}
